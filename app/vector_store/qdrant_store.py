from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


COLLECTION_NAME = "document_chunks"


client = QdrantClient(
    url="http://localhost:6333"
)


def create_collection(vector_size: int, recreate: bool = False):
    """
    Create the collection if it doesn't exist.

    recreate=True deletes any existing collection first, so re-indexing
    with a different chunking strategy can't leave stale chunks behind.
    """

    exists = client.collection_exists(COLLECTION_NAME)

    if exists and recreate:
        client.delete_collection(COLLECTION_NAME)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )