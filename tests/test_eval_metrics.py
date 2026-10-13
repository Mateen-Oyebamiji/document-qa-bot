import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
#--------------------------------------------------------------------------------------------

import json

import pytest

from evaluation.metrics import (
    first_evidence_hit_rank,
    first_page_hit_rank,
    load_dataset,
    summarise,
)


RETRIEVED = [
    {"page": 10, "text": "Nothing useful here."},
    {"page": 51, "text": "Sale of goods per region: Americas 38,429 35,408"},
    {"page": 52, "text": "Marketing\u00a010,151 9,372"},
]


def test_page_rank():
    assert first_page_hit_rank(RETRIEVED, [51]) == 2
    assert first_page_hit_rank(RETRIEVED, [52, 10]) == 1
    assert first_page_hit_rank(RETRIEVED, [99]) is None


def test_evidence_rank_normalises_whitespace_and_case():
    assert first_evidence_hit_rank(RETRIEVED, ["americas  38,429"]) == 2
    assert first_evidence_hit_rank(RETRIEVED, ["Marketing 10,151"]) == 3
    assert first_evidence_hit_rank(RETRIEVED, ["not there"]) is None
    assert first_evidence_hit_rank(RETRIEVED, []) is None


def test_summarise():
    rows = [
        {"page_rank": 1, "evidence_rank": 1, "has_evidence": True},
        {"page_rank": 4, "evidence_rank": None, "has_evidence": True},
        {"page_rank": None, "evidence_rank": None, "has_evidence": False},
        {"page_rank": 2, "evidence_rank": 2, "has_evidence": True},
    ]

    summary = summarise(rows, [1, 3, 5])

    assert summary["n"] == 4
    assert summary["hit@1"] == 0.25
    assert summary["hit@3"] == 0.5
    assert summary["hit@5"] == 0.75
    assert summary["mrr"] == pytest.approx((1 + 0.25 + 0.5) / 4)
    assert summary["n_evidence"] == 3
    assert summary["evidence@1"] == pytest.approx(1 / 3)
    assert summary["evidence@3"] == pytest.approx(2 / 3)


def test_load_dataset_validates(tmp_path):
    good = tmp_path / "good.jsonl"
    good.write_text(
        json.dumps({"id": "a", "question": "Q?", "gold_pages": [1]}) + "\n"
        + "# comment line\n\n"
    )
    examples = load_dataset(good)
    assert examples[0]["evidence"] == []
    assert examples[0]["category"] == "uncategorised"

    duplicate = tmp_path / "dup.jsonl"
    row = json.dumps({"id": "a", "question": "Q?", "gold_pages": [1]})
    duplicate.write_text(row + "\n" + row + "\n")
    with pytest.raises(ValueError):
        load_dataset(duplicate)

    no_pages = tmp_path / "empty.jsonl"
    no_pages.write_text(
        json.dumps({"id": "a", "question": "Q?", "gold_pages": []}) + "\n"
    )
    with pytest.raises(ValueError):
        load_dataset(no_pages)


def test_shipped_dataset_is_valid():
    examples = load_dataset("evaluation/datasets/lego_2025_retrieval.jsonl")
    assert 20 <= len(examples) <= 50