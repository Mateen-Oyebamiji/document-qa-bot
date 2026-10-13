
from app.retrieval.search import search_documents
from app.retrieval.keyword_search import search_by_keywords


def hybrid_search(
    question: str,
    limit: int = 5,
    candidate_limit: int = 20,
    rrf_k: int = 60,
    qdrant_client=None,
    collection_name: str | None = None
) -> list[dict]:
    """
    Combine vector and keyword rankings using
    Reciprocal Rank Fusion (RRF).
    """

    vector_results = search_documents(
        question=question,
        limit=candidate_limit,
        qdrant_client=qdrant_client,
        collection_name=collection_name
    )

    keyword_results = search_by_keywords(
        question=question,
        limit=candidate_limit,
        qdrant_client=qdrant_client,
        collection_name=collection_name
    )

    combined = {}

    # Add vector-search rankings.
    for rank, result in enumerate(vector_results, start=1):
        point_id = result.id

        if point_id not in combined:
            combined[point_id] = {
                "page": result.payload["page"],
                "chunk": result.payload["chunk"],
                "text": result.payload["text"],
                "point_id": point_id,
                "rrf_score": 0.0,
                "vector_rank": None,
                "keyword_rank": None
            }

        combined[point_id]["rrf_score"] += (
            1 / (rrf_k + rank)
        )

        combined[point_id]["vector_rank"] = rank

    # Add keyword-search rankings.
    for rank, result in enumerate(keyword_results, start=1):
        point_id = result["point_id"]

        if point_id not in combined:
            combined[point_id] = {
                "page": result["page"],
                "chunk": result["chunk"],
                "text": result["text"],
                "point_id": point_id,
                "rrf_score": 0.0,
                "vector_rank": None,
                "keyword_rank": None
            }

        combined[point_id]["rrf_score"] += (
            1 / (rrf_k + rank)
        )

        combined[point_id]["keyword_rank"] = rank

    ranked_results = sorted(
        combined.values(),
        key=lambda item: item["rrf_score"],
        reverse=True
    )

    return ranked_results[:limit]
