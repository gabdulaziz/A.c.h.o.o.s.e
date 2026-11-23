"""Local brain: a model running in Ollama on this computer.

Uses only Python's standard library to talk to Ollama's HTTP API
(http://localhost:11434 by default), so there is nothing extra to install.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any, Callable

from .base import BrainError, BrainReply, BrainStats, Message, ToolCall
from .textcalls import extract_text_tool_calls

_NS = 1_000_000_000  # Ollama reports durations in nanoseconds

# Some models write their hidden reasoning into the answer between these tags.
_THINK_TAGS = re.compile(r"<think>(.*?)(?:</think>|$)", re.DOTALL | re.IGNORECASE)


class OllamaBrain:
    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        context_tokens: int = 4096,
        temperature: float = 0.2,
        keep_alive: str = "30m",
        think: bool = False,
        timeout_seconds: float = 600,
    ) -> None:
        self.model = model
        self.name = f"ollama:{model}"
        self.host = host.rstrip("/")
        self.context_tokens = context_tokens
        self.temperature = temperature
        self.keep_alive = keep_alive
        self.think = think
        self.timeout_seconds = timeout_seconds
        # Some models reject the "think" switch. We learn that once and stop sending it.
        self._send_think = True
        # Talk to Ollama directly, never through a system web proxy.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    # ---- public -------------------------------------------------------

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        on_text: Callable[[str], None] | None = None,
    ) -> BrainReply:
        """Ask the model. If on_text is given, the answer is passed to it piece
        by piece while it is being written (streaming)."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [_to_ollama(m) for m in messages],
            "stream": on_text is not None,
            "keep_alive": self.keep_alive,
            "options": {"num_ctx": self.context_tokens, "temperature": self.temperature},
        }
        if tools:
            body["tools"] = tools
        if self._send_think:
            body["think"] = self.think

        send = (lambda: self._post_stream("/api/chat", body, on_text)) if on_text else (
            lambda: self._post("/api/chat", body)
        )
        try:
            data = send()
        except BrainError as err:
            if self._send_think and "think" in str(err).lower():
                self._send_think = False
                body.pop("think", None)
                data = send()
            else:
                raise

        msg = data.get("message") or {}
        content = msg.get("content") or ""
        thinking = msg.get("thinking") or ""
        if "<think>" in content.lower() or "</think>" in content.lower():
            if "<think>" not in content.lower():  # opening tag was eaten by the template
                content = "<think>" + content
            thinking += "".join(_THINK_TAGS.findall(content))
            content = _THINK_TAGS.sub("", content)
        content = content.strip()
        calls = [_from_ollama_call(c) for c in msg.get("tool_calls") or []]
        from_text = False
        if tools and not calls:
            names = {t.get("function", {}).get("name") for t in tools}
            calls, content = extract_text_tool_calls(content, names)
            from_text = bool(calls)

        return BrainReply(
            content=content,
            tool_calls=calls,
            thinking=thinking.strip(),
            tool_calls_from_text=from_text,
            stats=BrainStats(
                prompt_tokens=data.get("prompt_eval_count", 0) or 0,
                output_tokens=data.get("eval_count", 0) or 0,
                prompt_seconds=(data.get("prompt_eval_duration", 0) or 0) / _NS,
                output_seconds=(data.get("eval_duration", 0) or 0) / _NS,
                load_seconds=(data.get("load_duration", 0) or 0) / _NS,
                total_seconds=(data.get("total_duration", 0) or 0) / _NS,
            ),
        )

    def installed_models(self) -> list[str]:
        data = self._get("/api/tags")
        return [m["name"] for m in data.get("models", [])]

    def is_running(self) -> bool:
        try:
            self._get("/api/version")
            return True
        except BrainError:
            return False

    # ---- HTTP helpers -------------------------------------------------

    def _get(self, path: str) -> dict[str, Any]:
        return self._request(urllib.request.Request(self.host + path))

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            self.host + path,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._request(req)

    def _post_stream(
        self, path: str, body: dict[str, Any], on_text: Callable[[str], None]
    ) -> dict[str, Any]:
        """Like _post, but reads the answer piece by piece and passes visible
        text to on_text as it arrives. Returns the same shape as _post."""
        req = urllib.request.Request(
            self.host + path,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        shown = _StreamFilter(on_text)

        def read(resp) -> dict[str, Any]:
            content, thinking, calls, last = [], [], [], {}
            for line in resp:
                if not line.strip():
                    continue
                chunk = json.loads(line.decode("utf-8"))
                if chunk.get("error"):
                    raise BrainError(f"Ollama error: {chunk['error']}")
                msg = chunk.get("message") or {}
                if msg.get("thinking"):
                    thinking.append(msg["thinking"])
                if msg.get("tool_calls"):
                    calls.extend(msg["tool_calls"])
                if msg.get("content"):
                    content.append(msg["content"])
                    shown.feed(msg["content"])
                last = chunk
            last["message"] = {
                "content": "".join(content),
                "thinking": "".join(thinking),
                "tool_calls": calls,
            }
            return last

        return self._request(req, read)

    def _request(self, req: urllib.request.Request, read=None) -> dict[str, Any]:
        try:
            with self._opener.open(req, timeout=self.timeout_seconds) as resp:
                if read:
                    return read(resp)
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", "replace")
            try:
                detail = json.loads(detail).get("error", detail)
            except (ValueError, AttributeError):
                pass
            if err.code == 404 and "not found" in detail.lower():
                raise BrainError(
                    f"Model '{self.model}' is not downloaded yet. Run: ollama pull {self.model}"
                ) from None
            raise BrainError(f"Ollama error {err.code}: {detail}") from None
        except (urllib.error.URLError, ConnectionError, TimeoutError) as err:
            raise BrainError(
                f"Cannot reach Ollama at {self.host} ({err}). Is it running? Try: ollama serve"
            ) from None


class _StreamFilter:
    """Decides what streamed text to show on screen.

    Normal answers are shown as they arrive. If the answer starts like a
    tool request written as text ([, {, ```) or hidden reasoning (<think>),
    nothing is shown, because the brain will clean that up at the end.
    """

    def __init__(self, on_text: Callable[[str], None]) -> None:
        self.on_text = on_text
        self.decided: bool | None = None  # None = not sure yet
        self.buffer = ""

    def feed(self, piece: str) -> None:
        if self.decided is False:
            return
        if self.decided:
            self.on_text(piece)
            return
        self.buffer += piece
        start = self.buffer.lstrip()
        if not start:
            return
        if start[0] in "[{`<":
            self.decided = False
            return
        self.decided = True
        self.on_text(start)


def _to_ollama(m: Message) -> dict[str, Any]:
    out: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.tool_calls:
        out["tool_calls"] = [
            {"function": {"name": c.name, "arguments": c.arguments}} for c in m.tool_calls
        ]
    if m.tool_name:
        out["tool_name"] = m.tool_name
    return out


def _from_ollama_call(raw: dict[str, Any]) -> ToolCall:
    fn = raw.get("function") or {}
    args = fn.get("arguments") or {}
    if isinstance(args, str):  # a few models send arguments as a JSON string
        try:
            args = json.loads(args)
        except ValueError:
            args = {"_raw": args}
    if not isinstance(args, dict):
        args = {"_raw": args}
    return ToolCall(name=str(fn.get("name", "")), arguments=args)
