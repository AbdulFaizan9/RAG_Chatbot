"""Project 60 - Retrieval-Augmented Generation (RAG) Chatbot.

Document ingestion -> overlapping chunks -> BM25 index (saved/loaded as JSON) -> top-k
retrieval -> grounded answer with citations. Abstains when nothing relevant is found.
Plug in any LLM through ``generator(prompt) -> str``; otherwise an extractive answerer is used.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .common.text import chunk_words, split_sentences, tokenize


@dataclass
class Chunk:
    source: str
    index: int
    text: str

    @property
    def cite(self) -> str:
        return f"{self.source}#{self.index}"


@dataclass
class Answer:
    text: str
    citations: list[str]
    grounded: bool


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.chunks: list[Chunk] = []
        self._tf: list[Counter] = []
        self._df: Counter = Counter()
        self._len: list[int] = []

    def add(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            toks = tokenize(c.text, remove_stopwords=True)
            tf = Counter(toks)
            self.chunks.append(c)
            self._tf.append(tf)
            self._len.append(len(toks))
            self._df.update(tf.keys())

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        n = len(self.chunks)
        avg = sum(self._len) / n or 1.0
        q = set(tokenize(query, remove_stopwords=True))
        scores = []
        for i, tf in enumerate(self._tf):
            s = 0.0
            for t in q:
                f = tf.get(t, 0)
                if f:
                    idf = math.log(1 + (n - self._df[t] + 0.5) / (self._df[t] + 0.5))
                    s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self._len[i] / avg))
            scores.append(s)
        order = sorted(range(n), key=lambda i: -scores[i])[:k]
        return [(self.chunks[i], scores[i]) for i in order if scores[i] > 0]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps([asdict(c) for c in self.chunks]), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        idx = cls()
        idx.add([Chunk(**d) for d in json.loads(Path(path).read_text(encoding="utf-8"))])
        return idx


class RAGChatbot:
    NO_ANSWER = "I don't know based on the provided documents."

    def __init__(self, generator: Callable[[str], str] | None = None, chunk_size: int = 80,
                 overlap: int = 15, min_coverage: float = 0.5, top_k: int = 3):
        self.index = BM25Index()
        self.generator = generator
        self.chunk_size, self.overlap = chunk_size, overlap
        self.min_coverage, self.top_k = min_coverage, top_k

    def add_document(self, source: str, text: str) -> int:
        chunks = [Chunk(source, i, t) for i, t in enumerate(chunk_words(text, self.chunk_size, self.overlap))]
        self.index.add(chunks)
        return len(chunks)

    def add_file(self, path: str | Path) -> int:
        p = Path(path)
        return self.add_document(p.name, p.read_text(encoding="utf-8", errors="replace"))

    def build_prompt(self, question: str, hits: list[tuple[Chunk, float]]) -> str:
        ctx = "\n\n".join(f"[{c.cite}] {c.text}" for c, _ in hits)
        return ("Answer ONLY from the context. Cite sources like [file#n]. "
                f"If the answer is missing say you don't know.\n\nContext:\n{ctx}\n\nQuestion: {question}\nAnswer:")

    def ask(self, question: str) -> Answer:
        if not question.strip():
            return Answer("Please ask a question.", [], False)
        q_terms = set(tokenize(question, remove_stopwords=True))
        hits = [h for h in self.index.search(question, self.top_k)
                if q_terms and len(q_terms & set(tokenize(h[0].text, remove_stopwords=True)))
                / len(q_terms) >= self.min_coverage]
        if not hits:
            return Answer(self.NO_ANSWER, [], False)
        cites = [c.cite for c, _ in hits]
        if self.generator:
            return Answer(self.generator(self.build_prompt(question, hits)), cites, True)
        return Answer(self._extract(question, hits), cites, True)

    @staticmethod
    def _extract(question: str, hits: list[tuple[Chunk, float]]) -> str:
        q = set(tokenize(question, remove_stopwords=True))
        scored = []
        for chunk, _ in hits:
            for s in split_sentences(chunk.text):
                overlap = len(q & set(tokenize(s, remove_stopwords=True)))
                if overlap:
                    scored.append((overlap, chunk.cite, s))
        scored.sort(key=lambda t: -t[0])
        best, seen, picked = scored[:2], set(), []
        for _, cite, s in best:
            if s not in seen:
                seen.add(s)
                picked.append(f"{s} [{cite}]")
        return " ".join(picked) if picked else hits[0][0].text[:300] + f" [{hits[0][0].cite}]"


def demo() -> None:
    bot = RAGChatbot()
    bot.add_document("python.txt", "Python is a high-level programming language created by Guido van Rossum. "
                                   "It was first released in 1991. Python emphasises readability.")
    bot.add_document("numpy.txt", "NumPy provides fast multi-dimensional arrays for scientific computing in Python.")
    for q in ["Who created Python?", "What is NumPy used for?", "What is the capital of Mars?"]:
        a = bot.ask(q)
        print(f"> {q}\n{a.text}  {a.citations}")


if __name__ == "__main__":  # pragma: no cover
    demo()
