from app.embeddings.embedder import create_embedding
from app.vector_store.qdrant_store import client, COLLECTION_NAME


def search_documents(
    question: str,
    limit: int = 5,
    qdrant_client=None,
    collection_name: str | None = None
):
    """
    Vector search over the indexed chunks.

    `qdrant_client` and `collection_name` default to the app's
    collection; the evaluation harness passes its own so it can
    compare several chunking strategies side by side.
    """

    question_embedding = create_embedding(question)

    results = (qdrant_client or client).query_points(
        collection_name=collection_name or COLLECTION_NAME,
        query=question_embedding,
        limit=limit,
        with_payload=True
    )

    return results.points