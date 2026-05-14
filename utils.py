from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup


WHITESPACE_RE = re.compile(r"\s+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])\s*")


@dataclass(slots=True)
class DocumentRecord:
    id: str
    title: str
    clean_text: str
    original_html: str | None = None
    segments: tuple[str, ...] = ()


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def extract_title(soup: BeautifulSoup, document_id: str) -> str:
    title_tag = soup.find("title")
    if title_tag:
        title_text = normalize_whitespace(title_tag.get_text(" ", strip=True))
        if title_text:
            return title_text

    h1_tag = soup.find("h1")
    if h1_tag:
        h1_text = normalize_whitespace(h1_tag.get_text(" ", strip=True))
        if h1_text:
            return h1_text

    return f"Document {document_id}"


def extract_visible_segments(soup: BeautifulSoup) -> list[str]:
    for tag in soup(["script", "style"]):
        tag.decompose()

    root = soup.body if soup.body else soup
    segments: list[str] = []
    seen: set[str] = set()

    for element in root.find_all(["h1", "h2", "h3", "p", "li"]):
        text = normalize_whitespace(element.get_text(" ", strip=True))
        if not text or len(text) < 4:
            continue
        if text in seen:
            continue
        seen.add(text)
        segments.append(text)

    if segments:
        return segments

    fallback_text = normalize_whitespace(root.get_text(" ", strip=True))
    return split_text_into_segments(fallback_text)


def split_text_into_segments(text: str, max_length: int = 220) -> list[str]:
    normalized_text = normalize_whitespace(text)
    if not normalized_text:
        return []

    sentences = [part.strip() for part in SENTENCE_SPLIT_RE.split(normalized_text) if part.strip()]
    if not sentences:
        return [normalized_text[:max_length]]

    segments: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = sentence if not current else f"{current} {sentence}"
        if len(candidate) <= max_length:
            current = candidate
            continue

        if current:
            segments.append(current)
        current = sentence

    if current:
        segments.append(current)

    return segments


def html_to_document(document_id: str, html: str, keep_original_html: bool = True) -> DocumentRecord:
    soup = BeautifulSoup(html, "html.parser")
    title = extract_title(soup, document_id)
    segments = extract_visible_segments(soup)
    visible_root = soup.body if soup.body else soup
    clean_text = normalize_whitespace(visible_root.get_text(" ", strip=True))

    return DocumentRecord(
        id=document_id,
        title=title,
        clean_text=clean_text,
        original_html=html if keep_original_html else None,
        segments=tuple(segments),
    )


def count_keyword_occurrences(text: str, query: str) -> int:
    if not query:
        return 0

    normalized_text = text.casefold()
    normalized_query = query.casefold()

    count = 0
    start = 0
    while True:
        index = normalized_text.find(normalized_query, start)
        if index == -1:
            return count
        count += 1
        start = index + len(normalized_query)


def build_snippet(text: str, query: str, max_length: int = 130) -> str:
    if not text:
        return ""

    if not query:
        snippet = text[:max_length]
        return snippet if len(text) <= max_length else f"{snippet}..."

    normalized_text = text.casefold()
    normalized_query = query.casefold()
    match_index = normalized_text.find(normalized_query)

    if match_index == -1:
        snippet = text[:max_length]
        return snippet if len(text) <= max_length else f"{snippet}..."

    half_window = max(max_length - len(query), 40) // 2
    start = max(0, match_index - half_window)
    end = min(len(text), match_index + len(query) + half_window)

    snippet = text[start:end].strip()
    if start > 0:
        snippet = f"...{snippet}"
    if end < len(text):
        snippet = f"{snippet}..."

    if len(snippet) > max_length + 6:
        snippet = snippet[: max_length + 3].rstrip()
        if not snippet.endswith("..."):
            snippet = f"{snippet}..."

    return snippet


def truncate_text(text: str, max_length: int = 220) -> str:
    normalized_text = normalize_whitespace(text)
    if len(normalized_text) <= max_length:
        return normalized_text
    return f"{normalized_text[:max_length].rstrip()}..."


def iter_html_files(data_dir: Path) -> Iterable[Path]:
    if not data_dir.exists() or not data_dir.is_dir():
        return []
    return sorted(data_dir.glob("*.html"))
