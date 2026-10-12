
from app.retrieval.hybrid_search import hybrid_search

questions = [
    "What was LEGO A/S Parent Company profit before tax in 2025?",
    "What was the profit before tax in the Parent Company Statement of Profit or Loss for 2025?"
]

for question in questions:
    results = hybrid_search(
        question=question,
        limit=10,
        candidate_limit=20
    )

    print("\n" + "=" * 90)
    print(f"QUESTION: {question}")
    print("=" * 90)

    for rank, result in enumerate(results, start=1):
        print(
            f"\nFinal rank: {rank} | "
            f"Page: {result['page']} | "
            f"Chunk: {result['chunk']} | "
            f"RRF: {result['rrf_score']:.5f}"
        )

        print(f"Vector rank: {result['vector_rank']}")
        print(f"Keyword rank: {result['keyword_rank']}")
        print(f"Text preview: {result['text'][:250]}")
