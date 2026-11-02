"""Shared text utilities (pure Python + NumPy): tokenizing, sentence splitting, TF-IDF."""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Sequence

import numpy as np

STOPWORDS = frozenset(
    """a an and are as at be but by for from has have he her his i if in into is it its
    of on or our she so that the their them then there these they this to was we were
    what when where which who will with you your do does did can could should would may
    might not no yes than too very just also about over under again""".split()
)

_WORD = re.compile(r"[a-z0-9][a-z0-9'+#.\-]*[a-z0-9+#]|[a-z0-9]")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def tokenize(text: str, remove_stopwords: bool = False) -> list[str]:
    """Lower-case word tokens; keeps things like ``c++``, ``node.js`` and ``e-mail``."""
    tokens = [t.strip(".-'") for t in _WORD.findall(text.lower())]
    tokens = [t for t in tokens if t]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def split_sentences(text: str) -> list[str]:
    """Split text into sentences (good enough for English prose)."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


class TfidfVectorizer:
    """Minimal TF-IDF vectorizer producing L2-normalised dense NumPy rows."""

    def __init__(self, remove_stopwords: bool = True, sublinear_tf: bool = True):
        self.remove_stopwords = remove_stopwords
        self.sublinear_tf = sublinear_tf
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = np.zeros(0)

    def fit(self, docs: Sequence[str]) -> "TfidfVectorizer":
        tokenised = [set(tokenize(d, self.remove_stopwords)) for d in docs]
        df = Counter(t for toks in tokenised for t in toks)
        self.vocab = {t: i for i, t in enumerate(sorted(df))}
        n = len(docs)
        self.idf = np.array([np.log((1 + n) / (1 + df[t])) + 1.0 for t in self.vocab])
        return self

    def transform(self, docs: Sequence[str]) -> np.ndarray:
        mat = np.zeros((len(docs), len(self.vocab)))
        for row, doc in enumerate(docs):
            for tok, cnt in Counter(tokenize(doc, self.remove_stopwords)).items():
                col = self.vocab.get(tok)
                if col is not None:
                    mat[row, col] = (1 + np.log(cnt)) if self.sublinear_tf else cnt
        mat *= self.idf
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.where(norms == 0, 1.0, norms)

    def fit_transform(self, docs: Sequence[str]) -> np.ndarray:
        return self.fit(docs).transform(docs)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between rows of ``a`` and rows of ``b`` (safe for zeros)."""
    a = np.atleast_2d(a).astype(float)
    b = np.atleast_2d(b).astype(float)
    an = np.linalg.norm(a, axis=1, keepdims=True)
    bn = np.linalg.norm(b, axis=1, keepdims=True)
    return (a / np.where(an == 0, 1, an)) @ (b / np.where(bn == 0, 1, bn)).T


def chunk_words(text: str, size: int = 120, overlap: int = 20) -> list[str]:
    """Split text into word windows of ``size`` words overlapping by ``overlap``."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    words = text.split()
    if not words:
        return []
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


def top_terms(docs: Iterable[str], k: int = 10) -> list[str]:
    """Top-k terms of a document set by summed TF-IDF weight."""
    docs = list(docs)
    vec = TfidfVectorizer()
    mat = vec.fit_transform(docs)
    if mat.shape[1] == 0:
        return []
    inv = {i: t for t, i in vec.vocab.items()}
    sums = mat.sum(axis=0)
    order = np.argsort(-sums)[:k]
    return [inv[int(i)] for i in order if sums[i] > 0 and not inv[int(i)].isdigit()]
