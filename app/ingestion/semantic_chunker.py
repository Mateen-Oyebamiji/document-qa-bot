"""
Semantic chunking.

Instead of cutting every N words, split each page into sentences,
embed each sentence (with a little surrounding context), and start a
new chunk wherever the meaning shifts sharply between neighbouring
sentences. A size cap keeps chunks inside the embedding model's input
window, and tiny fragments are merged into their neighbours.

Chunks never cross a page boundary, so every chunk keeps a single page
number, the same as the fixed-size chunker.
"""

import re
from typing import Callable, Sequence

import numpy as np


# Split after . ! ? when the next character looks like the start of a sentence.
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“‘(\[])")

EmbedFn = Callable[[Sequence[str]], np.ndarray]


def split_sentences(text: str, max_sentence_words: int = 60) -> list[str]:
    """
    Split text into sentence-like units.

    PDF tables and charts often have no sentence punctuation at all, so
    any unit longer than `max_sentence_words` is broken into word windows.
    """

    text = " ".join(text.split())

    if not text:
        return []

    units = []

    for sentence in SENTENCE_BOUNDARY.split(text):
        words = sentence.split()

        for start in range(0, len(words), max_sentence_words):
            piece = words[start:start + max_sentence_words]

            if piece:
                units.append(" ".join(piece))

    return units


def _word_count(units: list[str]) -> int:
    return sum(len(unit.split()) for unit in units)


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def semantic_chunk_pages(
    pages: list[dict],
    embed: EmbedFn,
    breakpoint_percentile: float = 75,
    buffer_size: int = 1,
    max_words: int = 200,
    min_words: int = 40,
    max_sentence_words: int = 60
) -> list[dict]:
    """
    Chunk a document (list of {"page", "text"} dicts) semantically.

    breakpoint_percentile: a new chunk starts where the cosine distance
        between neighbouring sentences is above this percentile of all
        distances in the document. Lower = more, smaller chunks.
    buffer_size: sentences on each side included when embedding a
        sentence, which smooths out noisy one-line sentences.
    max_words: hard cap on chunk size. all-MiniLM-L6-v2 reads at most
        256 tokens (~180-200 words), so larger chunks are truncated.
    min_words: chunks smaller than this are merged into a neighbour.
    """

    if max_words <= 0:
        raise ValueError("max_words must be greater than 0")
    if min_words < 0 or min_words > max_words:
        raise ValueError("min_words must be between 0 and max_words")

    page_units = []

    for page in pages:
        units = split_sentences(page["text"], max_sentence_words)

        if units:
            page_units.append((page["page"], units))

    if not page_units:
        return []

    # 1. Embed every sentence together with its neighbours.
    windows = []

    for _, units in page_units:
        for i in range(len(units)):
            low = max(0, i - buffer_size)
            high = min(len(units), i + buffer_size + 1)
            windows.append(" ".join(units[low:high]))

    vectors = _normalise(np.asarray(embed(windows), dtype=float))

    # 2. Cosine distance between consecutive sentences on the same page.
    page_distances = []
    position = 0

    for _, units in page_units:
        page_vectors = vectors[position:position + len(units)]
        position += len(units)

        distances = 1 - np.sum(page_vectors[:-1] * page_vectors[1:], axis=1)
        page_distances.append(distances)

    all_distances = np.concatenate(
        [d for d in page_distances if len(d)] or [np.array([])]
    )

    if len(all_distances):
        threshold = float(np.percentile(all_distances, breakpoint_percentile))
    else:
        threshold = float("inf")

    # 3. Group sentences, breaking on large distances or the size cap.
    chunks = []

    for (page_number, units), distances in zip(page_units, page_distances):
        groups = []
        current = [units[0]]

        for i in range(1, len(units)):
            too_big = (
                _word_count(current) + len(units[i].split()) > max_words
            )

            if distances[i - 1] > threshold or too_big:
                groups.append(current)
                current = [units[i]]
            else:
                current.append(units[i])

        groups.append(current)

        # 4. Merge fragments that are too small to stand on their own.
        merged = []

        for group in groups:
            if (
                merged
                and (
                    _word_count(group) < min_words
                    or _word_count(merged[-1]) < min_words
                )
                and _word_count(merged[-1]) + _word_count(group) <= max_words
            ):
                merged[-1] = merged[-1] + group
            else:
                merged.append(group)

        for chunk_number, group in enumerate(merged, start=1):
            chunks.append({
                "page": page_number,
                "chunk": chunk_number,
                "text": " ".join(group)
            })

    return chunks