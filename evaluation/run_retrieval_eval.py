"""
Retrieval evaluation harness.

Indexes the PDF once per chunking strategy (in-memory Qdrant, so no
Docker needed), runs every labelled question through each retriever,
and reports hit@k, MRR and evidence hit rate.

Run from the project root:

    python -m evaluation.run_retrieval_eval

Options:

    --pdf app/data/report.pdf
    --dataset evaluation/datasets/lego_2025_retrieval.jsonl
    --chunkers fixed_500_50,semantic_p90     (default: all)
    --retrievers vector,hybrid               (default: vector,keyword,hybrid)
    --k 1,3,5,10
    --out evaluation/results

Add or tweak strategies in CHUNKERS below.
"""

import argparse
import csv
import json
import statistics
import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# The app's module-level Qdrant client points at localhost:6333; the
# harness uses an in-memory client instead, so its version check is noise.
warnings.filterwarnings("ignore", message="Failed to obtain server version")

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.embeddings.embedder import MODEL_NAME, model
from app.ingestion.chunker import chunk_text
from app.ingestion.pdf_parser import extract_text_from_pdf
from app.ingestion.semantic_chunker import semantic_chunk_pages
from app.retrieval.hybrid_search import hybrid_search
from app.retrieval.keyword_search import search_by_keywords
from app.retrieval.search import search_documents
from evaluation.metrics import (
    first_evidence_hit_rank,
    first_page_hit_rank,
    load_dataset,
    summarise,
)


DEFAULT_PDF = "app/data/report.pdf"
DEFAULT_DATASET = "evaluation/datasets/lego_2025_retrieval.jsonl"
DEFAULT_OUT = "evaluation/results"


def embed(texts):
    return model.encode(
        list(texts),
        batch_size=64,
        show_progress_bar=False
    )


def fixed_chunks(pages, chunk_size, overlap):
    chunks = []

    for page in pages:
        chunks.extend(
            chunk_text(
                text=page["text"],
                page_number=page["page"],
                chunk_size=chunk_size,
                overlap=overlap
            )
        )

    return chunks


# name -> (description, function(pages) -> chunks)
CHUNKERS = {
    "fixed_500_50": (
        "Fixed 500 words / 50 overlap (current app setting)",
        lambda pages: fixed_chunks(pages, 500, 50),
    ),
    "fixed_250_50": (
        "Fixed 250 words / 50 overlap",
        lambda pages: fixed_chunks(pages, 250, 50),
    ),
    "fixed_150_30": (
        "Fixed 150 words / 30 overlap (fits MiniLM's 256-token window)",
        lambda pages: fixed_chunks(pages, 150, 30),
    ),
    "semantic_p90": (
        "Semantic, break at 90th-percentile distance, max 200 words",
        lambda pages: semantic_chunk_pages(
            pages, embed, breakpoint_percentile=90, max_words=200
        ),
    ),
    "semantic_p75": (
        "Semantic, break at 75th-percentile distance, max 200 words",
        lambda pages: semantic_chunk_pages(
            pages, embed, breakpoint_percentile=75, max_words=200
        ),
    ),
}

RETRIEVERS = ("vector", "keyword", "hybrid")


def token_stats(chunks):
    """Share of chunks longer than the embedding model's input window."""
    tokenizer = getattr(model, "tokenizer", None)
    max_tokens = getattr(model, "max_seq_length", None)

    if tokenizer is None or not max_tokens:
        return None, None

    lengths = [
        len(tokenizer(chunk["text"], add_special_tokens=True)["input_ids"])
        for chunk in chunks
    ]
    truncated = sum(1 for length in lengths if length > max_tokens)

    return truncated / len(lengths), max_tokens


def build_index(client, collection_name, chunks):
    vectors = embed([chunk["text"] for chunk in chunks])

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=len(vectors[0]),
            distance=Distance.COSINE
        )
    )

    client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=index,
                vector=vector.tolist(),
                payload={
                    "page": chunk["page"],
                    "chunk": chunk["chunk"],
                    "text": chunk["text"]
                }
            )
            for index, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]
    )


def retrieve(retriever, question, limit, client, collection_name):
    """Run one retriever and normalise results to page/chunk/text dicts."""
    if retriever == "vector":
        points = search_documents(
            question,
            limit=limit,
            qdrant_client=client,
            collection_name=collection_name
        )
        return [
            {
                "page": point.payload["page"],
                "chunk": point.payload["chunk"],
                "text": point.payload["text"]
            }
            for point in points
        ]

    if retriever == "keyword":
        results = search_by_keywords(
            question,
            limit=limit,
            qdrant_client=client,
            collection_name=collection_name
        )
    elif retriever == "hybrid":
        results = hybrid_search(
            question,
            limit=limit,
            candidate_limit=max(20, limit),
            qdrant_client=client,
            collection_name=collection_name
        )
    else:
        raise ValueError(f"Unknown retriever: {retriever}")

    return [
        {"page": r["page"], "chunk": r["chunk"], "text": r["text"]}
        for r in results
    ]


def pct(value):
    return "–" if value is None else f"{value * 100:.0f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pdf", default=DEFAULT_PDF)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--chunkers", default=",".join(CHUNKERS))
    parser.add_argument("--retrievers", default=",".join(RETRIEVERS))
    parser.add_argument("--k", default="1,3,5,10")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    ks = sorted(int(k) for k in args.k.split(","))
    k_max = max(ks)
    context_k = 5 if 5 in ks else ks[-1]
    chunker_names = [name.strip() for name in args.chunkers.split(",")]
    retrievers = [name.strip() for name in args.retrievers.split(",")]

    for name in chunker_names:
        if name not in CHUNKERS:
            parser.error(f"unknown chunker '{name}'. Options: {', '.join(CHUNKERS)}")
    for name in retrievers:
        if name not in RETRIEVERS:
            parser.error(f"unknown retriever '{name}'. Options: {', '.join(RETRIEVERS)}")

    dataset = load_dataset(args.dataset)
    pages = extract_text_from_pdf(args.pdf)

    print(f"Dataset: {len(dataset)} questions from {args.dataset}")
    print(f"Document: {len(pages)} pages with text from {args.pdf}")
    print(f"Embedding model: {MODEL_NAME}\n")

    client = QdrantClient(":memory:")

    config_rows = []      # one per chunker x retriever
    question_rows = []    # one per chunker x retriever x question

    for chunker_name in chunker_names:
        description, chunk_fn = CHUNKERS[chunker_name]

        started = time.perf_counter()
        chunks = chunk_fn(pages)
        word_counts = [len(chunk["text"].split()) for chunk in chunks]
        truncated_share, max_tokens = token_stats(chunks)

        collection_name = f"eval_{chunker_name}"
        build_index(client, collection_name, chunks)
        index_seconds = time.perf_counter() - started

        print(
            f"[{chunker_name}] {len(chunks)} chunks, "
            f"median {statistics.median(word_counts):.0f} words, "
            f"{pct(truncated_share)} over the {max_tokens}-token limit "
            f"(indexed in {index_seconds:.1f}s)"
        )

        for retriever in retrievers:
            rows = []

            for example in dataset:
                retrieved = retrieve(
                    retriever,
                    example["question"],
                    k_max,
                    client,
                    collection_name
                )

                row = {
                    "chunker": chunker_name,
                    "retriever": retriever,
                    "id": example["id"],
                    "category": example["category"],
                    "question": example["question"],
                    "gold_pages": example["gold_pages"],
                    "retrieved_pages": [chunk["page"] for chunk in retrieved],
                    "page_rank": first_page_hit_rank(
                        retrieved, example["gold_pages"]
                    ),
                    "has_evidence": bool(example["evidence"]),
                    "evidence_rank": first_evidence_hit_rank(
                        retrieved, example["evidence"]
                    ),
                    "context_words": sum(
                        len(chunk["text"].split())
                        for chunk in retrieved[:context_k]
                    ),
                }
                rows.append(row)

            summary = summarise(rows, ks)
            summary.update({
                "chunker": chunker_name,
                "description": description,
                "retriever": retriever,
                "chunks": len(chunks),
                "median_words": statistics.median(word_counts),
                "truncated_share": truncated_share,
                "context_words": statistics.mean(
                    row["context_words"] for row in rows
                ),
                "by_category": {},
            })

            by_category = defaultdict(list)
            for row in rows:
                by_category[row["category"]].append(row)
            for category, category_rows in sorted(by_category.items()):
                summary["by_category"][category] = summarise(
                    category_rows, ks
                )

            config_rows.append(summary)
            question_rows.extend(rows)

            print(
                f"    {retriever:<8} hit@{context_k} "
                f"{pct(summary[f'hit@{context_k}'])}  "
                f"MRR {summary['mrr']:.2f}  "
                f"evidence@{context_k} "
                f"{pct(summary.get(f'evidence@{context_k}'))}"
            )


if __name__ == "__main__":
    main()