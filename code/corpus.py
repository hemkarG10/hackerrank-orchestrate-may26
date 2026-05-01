"""Local Markdown corpus loading and deterministic retrieval."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Document:
    path: Path
    company: str
    title: str
    product_area: str
    text: str


@dataclass(frozen=True)
class SearchResult:
    document: Document
    score: float


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def clean_markdown(text: str) -> str:
    text = re.sub(r"---.*?---", " ", text, count=1, flags=re.S)
    text = re.sub(r"!\[[^\]]*]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]+)]\([^)]*\)", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"#+\s*", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def frontmatter_value(raw: str, key: str) -> str:
    match = re.search(rf"^{re.escape(key)}:\s*\"?([^\"\n]+)\"?", raw, flags=re.M)
    return match.group(1).strip() if match else ""


def infer_company(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "hackerrank" in parts:
        return "HackerRank"
    if "claude" in parts:
        return "Claude"
    if "visa" in parts:
        return "Visa"
    return "None"


def infer_product_area(path: Path) -> str:
    parts = list(path.parts)
    for company in ("hackerrank", "claude", "visa"):
        if company in parts:
            index = parts.index(company)
            remainder = parts[index + 1 :]
            if remainder:
                if company == "hackerrank":
                    return remainder[0].replace("-", "_")
                if company == "claude":
                    return remainder[0].replace("-", "_")
                if company == "visa":
                    if len(remainder) > 2:
                        return remainder[-2].replace("-", "_")
                    return remainder[-1].replace(".md", "").replace("-", "_")
    return ""


class CorpusIndex:
    """Small BM25-style index over the shipped Markdown files."""

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self._tokens = [tokenize(f"{doc.title} {doc.product_area} {doc.text}") for doc in documents]
        self._doc_freq: dict[str, int] = {}
        for tokens in self._tokens:
            for token in set(tokens):
                self._doc_freq[token] = self._doc_freq.get(token, 0) + 1
        self._avg_len = sum(len(tokens) for tokens in self._tokens) / max(len(self._tokens), 1)

    @classmethod
    def from_directory(cls, data_dir: Path) -> "CorpusIndex":
        documents: list[Document] = []
        for path in sorted(data_dir.rglob("*.md")):
            raw = path.read_text(encoding="utf-8", errors="ignore")
            title = frontmatter_value(raw, "title")
            if not title:
                heading = re.search(r"^#\s+(.+)$", raw, flags=re.M)
                title = heading.group(1).strip() if heading else path.stem.replace("-", " ")
            documents.append(
                Document(
                    path=path,
                    company=infer_company(path),
                    title=title,
                    product_area=infer_product_area(path),
                    text=clean_markdown(raw),
                )
            )
        return cls(documents)

    def search(self, query: str, company: str | None = None, limit: int = 5) -> list[SearchResult]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        results: list[SearchResult] = []
        total_docs = max(len(self.documents), 1)
        query_set = set(query_tokens)
        for doc, doc_tokens in zip(self.documents, self._tokens):
            if company and company != "None" and doc.company.lower() != company.lower():
                continue
            if not doc_tokens:
                continue
            term_counts: dict[str, int] = {}
            for token in doc_tokens:
                if token in query_set:
                    term_counts[token] = term_counts.get(token, 0) + 1
            if not term_counts:
                continue
            score = 0.0
            length = len(doc_tokens)
            for token, freq in term_counts.items():
                df = self._doc_freq.get(token, 0)
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                denom = freq + 1.2 * (1 - 0.75 + 0.75 * length / max(self._avg_len, 1))
                score += idf * (freq * 2.2) / denom
            title_text = f"{doc.title} {doc.product_area}".lower()
            score += 0.2 * sum(1 for token in query_set if token in title_text)
            if score > 0:
                results.append(SearchResult(doc, score))
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:limit]
