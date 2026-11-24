"""Meaning fingerprints (embeddings) from a small model running in Ollama."""

from __future__ import annotations

from ..brain import BrainError, OllamaBrain


class OllamaEmbedder:
    """Uses Ollama's /api/embed endpoint.

    EmbeddingGemma works best when told whether a text is a question or a
    document, so we add the small prefixes its makers recommend.
    """

    def __init__(self, model: str = "embeddinggemma", host: str = "http://localhost:11434") -> None:
        self.name = model
        self._http = OllamaBrain(model, host=host)
        self._gemma_style = "embeddinggemma" in model

    def embed_documents(self, texts: list[str], title: str = "none") -> list[list[float]]:
        if self._gemma_style:
            texts = [f"title: {title} | text: {t}" for t in texts]
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        if self._gemma_style:
            text = f"task: search result | query: {text}"
        return self._embed([text])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 16):  # small batches keep memory use low
            data = self._http._post(
                "/api/embed", {"model": self.name, "input": texts[i : i + 16], "keep_alive": "10m"}
            )
            vectors = data.get("embeddings")
            if not vectors:
                raise BrainError(f"Ollama returned no embeddings from {self.name}")
            out.extend(vectors)
        return out
