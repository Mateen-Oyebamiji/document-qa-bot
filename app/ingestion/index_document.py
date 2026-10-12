import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from qdrant_client.models import PointStruct

from app.ingestion.document_processor import process_document
from app.embeddings.embedder import create_embedding
from app.vector_store.qdrant_store import client, create_collection, COLLECTION_NAME


PDF_PATH = "app/data/report.pdf"


chunks = process_document(
    file_path=PDF_PATH,
    chunk_size=500,
    overlap=50
)


if not chunks:
    raise ValueError("No chunks were created.")


first_embedding = create_embedding(chunks[0]["text"])

create_collection(
    vector_size=len(first_embedding)
)


points = []


for index, chunk in enumerate(chunks):

    embedding = create_embedding(chunk["text"])

    point = PointStruct(
        id=index,
        vector=embedding,
        payload={
            "page": chunk["page"],
            "chunk": chunk["chunk"],
            "text": chunk["text"]
        }
    )

    points.append(point)


client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)


print(f"Indexed {len(points)} chunks.")