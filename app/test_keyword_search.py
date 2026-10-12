
from app.retrieval.keyword_search import search_by_keywords

question = "What was LEGO group's profit before tax in 2025?"

results = search_by_keywords(
    question=question,
    limit=10
)

print(f"\nQUESTION: {question}")
print("=" * 80)

for rank, result in enumerate(results, start=1):
    print(
        f"\nRank {rank}: "
        f"Page {result['page']} | "
        f"Chunk {result['chunk']} | "
        f"Score {result['score']:.4f}"
    )

    print(f"Matched terms: {result['matched_terms']}")
    print("-" * 80)
    print(result["text"][:500])
