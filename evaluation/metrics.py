"""
Retrieval metrics.

Two notions of a "hit" are tracked for each question:

- page hit: a retrieved chunk comes from one of the gold pages.
  Independent of how the page was chunked, so it is fair across
  chunking strategies, but lenient: a page can hold many facts.
- evidence hit: a retrieved chunk contains one of the question's
  evidence strings (e.g. "Total remuneration 282"). Stricter: it fails
  if the chunk boundary cut the fact in half or the wrong half of a
  long page was retrieved.
"""

import json
from pathlib import Path


def normalise(text: str) -> str:
    """Collapse whitespace (incl. non-breaking spaces) and lowercase."""
    return " ".join(text.split()).lower()


def load_dataset(path: str | Path) -> list[dict]:
    examples = []

    with open(path, encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            example = json.loads(line)

            for field in ("id", "question", "gold_pages"):
                if field not in example:
                    raise ValueError(
                        f"{path}:{line_number} is missing '{field}'"
                    )

            if not example["gold_pages"]:
                raise ValueError(
                    f"{path}:{line_number} has no gold_pages; "
                    "unanswerable questions don't belong in a hit-rate set"
                )

            example.setdefault("category", "uncategorised")
            example.setdefault("evidence", [])
            examples.append(example)

    ids = [example["id"] for example in examples]

    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} has duplicate ids")

    return examples


def first_page_hit_rank(
    retrieved: list[dict],
    gold_pages: list[int]
) -> int | None:
    """1-based rank of the first chunk from a gold page, or None."""
    gold = set(gold_pages)

    for rank, chunk in enumerate(retrieved, start=1):
        if chunk["page"] in gold:
            return rank

    return None


def first_evidence_hit_rank(
    retrieved: list[dict],
    evidence: list[str]
) -> int | None:
    """1-based rank of the first chunk containing any evidence string."""
    if not evidence:
        return None

    needles = [normalise(item) for item in evidence]

    for rank, chunk in enumerate(retrieved, start=1):
        text = normalise(chunk["text"])

        if any(needle in text for needle in needles):
            return rank

    return None


def summarise(rows: list[dict], ks: list[int]) -> dict:
    """
    Aggregate per-question rows into hit@k / MRR.

    Each row needs `page_rank` (int|None), `evidence_rank` (int|None)
    and `has_evidence` (bool).
    """

    n = len(rows)
    summary = {"n": n}

    if n == 0:
        return summary

    for k in ks:
        summary[f"hit@{k}"] = sum(
            1 for row in rows
            if row["page_rank"] is not None and row["page_rank"] <= k
        ) / n

    summary["mrr"] = sum(
        1 / row["page_rank"] for row in rows if row["page_rank"]
    ) / n

    with_evidence = [row for row in rows if row["has_evidence"]]
    summary["n_evidence"] = len(with_evidence)

    if with_evidence:
        for k in ks:
            summary[f"evidence@{k}"] = sum(
                1 for row in with_evidence
                if row["evidence_rank"] is not None
                and row["evidence_rank"] <= k
            ) / len(with_evidence)

    return summary