from app.embeddings.embedder import create_embedding
from app.vector_store.qdrant_store import client, COLLECTION_NAME


def search_documents(
    question: str,
    limit: int = 5
):

    question_embedding = create_embedding(question)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=question_embedding,
        limit=limit,
        with_payload=True
    )

    return results.points