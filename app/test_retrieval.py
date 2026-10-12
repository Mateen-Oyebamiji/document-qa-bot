
from app.retrieval.search import search_documents

questions = [
    "What was LEGO group's profit before tax in 2025?",
    "Profit before tax 2025 16,809 mDKK",
    "What was the profit before tax reported in the Parent Company Statement of Profit or Loss for 2025?",
]

for question in questions:
    print("\n" + "=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)

    results = search_documents(
        question=question,
        limit=10
    )

    for rank, result in enumerate(results, start=1):
        print(
            f"Rank {rank}: "
            f"Page {result.payload['page']} | "
            f"Chunk {result.payload['chunk']} | "
            f"Score {result.score:.4f}"
        )
