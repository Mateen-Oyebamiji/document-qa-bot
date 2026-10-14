"""
Sanity-check a retrieval dataset against the PDF before trusting any scores.

For every question, confirms that each evidence string really appears in
the extracted text, and on which pages. Run from the project root:

    python -m evaluation.check_dataset
"""

import sys

from app.ingestion.pdf_parser import extract_text_from_pdf
from evaluation.metrics import load_dataset, normalise


PDF_PATH = "app/data/report.pdf"
DATASET_PATH = "evaluation/datasets/lego_2025_retrieval.jsonl"


def main():
    pages = {
        page["page"]: normalise(page["text"])
        for page in extract_text_from_pdf(PDF_PATH)
    }
    dataset = load_dataset(DATASET_PATH)
    problems = 0

    for example in dataset:
        gold = set(example["gold_pages"])

        for evidence in example["evidence"]:
            found = sorted(
                number for number, text in pages.items()
                if normalise(evidence) in text
            )

            if not found:
                problems += 1
                print(f"✗ {example['id']}: evidence not found anywhere: {evidence!r}")
            elif not gold & set(found):
                problems += 1
                print(
                    f"✗ {example['id']}: evidence found on {found} "
                    f"but gold pages are {sorted(gold)}: {evidence!r}"
                )
            else:
                print(f"✓ {example['id']}: {evidence!r} on pages {found}")

        if not example["evidence"]:
            print(f"- {example['id']}: no evidence string (page-only check)")

    print(f"\n{len(dataset)} questions, {problems} problem(s)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()