"""Rescue tool requests that a model wrote as plain text.

Some small models (phi4-mini, sometimes llama) answer with the tool request
written out as JSON text, for example

    [{"type":"function","function":{"name":"list_folder","arguments":{"path":"~"}}}]

instead of using Ollama's proper tool-call field. This module spots that and
turns it into real ToolCalls. Only tools Achoose actually has are accepted,
and rescued calls still go through the same confirmation gate as any other.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .base import ToolCall

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_TAGGED = re.compile(r"<(?:tool_call|function_call|tool_calls)>\s*(.*?)\s*</(?:tool_call|function_call|tool_calls)>", re.DOTALL)


def extract_text_tool_calls(content: str, tool_names: set[str]) -> tuple[list[ToolCall], str]:
    """Return (calls found, content with the JSON removed).

    If nothing valid is found, returns ([], content) unchanged.
    """
    if not content or not tool_names:
        return [], content

    for pattern in (_TAGGED, _FENCE):
        blocks = pattern.findall(content)
        if blocks:
            calls = [c for b in blocks for c in _parse(b, tool_names)]
            if calls:
                return calls, pattern.sub("", content).strip()

    # Plain JSON somewhere in the text: try from each opening bracket.
    decoder = json.JSONDecoder()
    for start, ch in enumerate(content):
        if ch not in "[{":
            continue
        try:
            obj, end = decoder.raw_decode(content, start)
        except ValueError:
            continue
        calls = _from_obj(obj, tool_names)
        if calls:
            rest = (content[:start] + content[end:]).strip()
            return calls, rest
    return [], content


def _parse(text: str, tool_names: set[str]) -> list[ToolCall]:
    try:
        return _from_obj(json.loads(text), tool_names)
    except ValueError:
        return []


def _from_obj(obj: Any, tool_names: set[str]) -> list[ToolCall]:
    items = obj if isinstance(obj, list) else [obj]
    calls: list[ToolCall] = []
    for item in items:
        if not isinstance(item, dict):
            return []
        fn = item.get("function") if isinstance(item.get("function"), dict) else item
        name = fn.get("name")
        args = fn.get("arguments", fn.get("parameters", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                return []
        if name not in tool_names or not isinstance(args, dict):
            return []
        calls.append(ToolCall(name=name, arguments=args))
    return calls
