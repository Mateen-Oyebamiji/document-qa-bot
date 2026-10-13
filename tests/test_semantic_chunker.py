import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
#--------------------------------------------------------------------------------------------
import numpy as np
import pytest

from app.ingestion.semantic_chunker import semantic_chunk_pages, split_sentences


def fake_embed(texts):
    """Embed by topic keyword so the semantic breakpoints are predictable."""
    vectors = []
    for text in texts:
        revenue = text.lower().count("revenue")
        factory = text.lower().count("factory")
        vectors.append([revenue + 1e-6, factory + 1e-6])
    return np.array(vectors)


def test_split_sentences_handles_punctuation_and_long_units():
    text = "Revenue grew. Profit rose!  " + " ".join(["123"] * 25)
    units = split_sentences(text, max_sentence_words=10)

    assert units[0] == "Revenue grew."
    assert units[1].startswith("Profit rose!")
    assert all(len(unit.split()) <= 10 for unit in units)


def test_breaks_where_topic_changes():
    text = (
        "Revenue grew strongly this year. Revenue rose in every region. "
        "Revenue was a record. The new factory opened in Vietnam. "
        "The factory runs on solar power. The factory employs many people."
    )

    chunks = semantic_chunk_pages(
        [{"page": 3, "text": text}],
        fake_embed,
        breakpoint_percentile=50,
        buffer_size=0,
        min_words=0,
    )

    assert len(chunks) == 2
    assert "factory" not in chunks[0]["text"].lower()
    assert "revenue" not in chunks[1]["text"].lower()
    assert [chunk["chunk"] for chunk in chunks] == [1, 2]
    assert all(chunk["page"] == 3 for chunk in chunks)


def test_respects_max_words_and_keeps_all_text():
    text = " ".join(f"Revenue sentence number {i}." for i in range(60))

    chunks = semantic_chunk_pages(
        [{"page": 1, "text": text}],
        fake_embed,
        max_words=40,
        min_words=0,
    )

    assert all(len(chunk["text"].split()) <= 40 for chunk in chunks)
    joined = " ".join(chunk["text"] for chunk in chunks)
    assert joined.split() == text.split()


def test_never_crosses_pages_and_skips_empty_pages():
    pages = [
        {"page": 1, "text": "Revenue grew. Revenue rose."},
        {"page": 2, "text": "   "},
        {"page": 3, "text": "The factory opened."},
    ]

    chunks = semantic_chunk_pages(pages, fake_embed, min_words=0)

    assert {chunk["page"] for chunk in chunks} == {1, 3}


def test_small_fragments_are_merged():
    text = (
        "Revenue grew strongly this year in all regions. "
        "Factory. "
        "Revenue rose again in the second half of the year."
    )

    chunks = semantic_chunk_pages(
        [{"page": 1, "text": text}],
        fake_embed,
        breakpoint_percentile=0,
        buffer_size=0,
        min_words=5,
    )

    assert all(len(chunk["text"].split()) >= 5 for chunk in chunks)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        semantic_chunk_pages([], fake_embed, max_words=0)
    with pytest.raises(ValueError):
        semantic_chunk_pages([], fake_embed, max_words=10, min_words=20)