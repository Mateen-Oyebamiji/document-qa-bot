import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from qdrant_client.models import PointStruct

from app.embeddings.embedder import model
from app.ingestion.pdf_parser import extract_text_from_pdf
from app.ingestion.semantic_chunker import semantic_chunk_pages
from app.vector_store.qdrant_store import client, create_collection, COLLECTION_NAME


PDF_PATH = "app/data/report.pdf"

# Chosen with the retrieval evaluation (python -m evaluation.run_retrieval_eval):
# semantic_p75 + hybrid search scored best. Re-run the eval before changing these.
BREAKPOINT_PERCENTILE = 75
MAX_WORDS = 200


def embed(texts):
    return model.encode(
        list(texts),
        batch_size=64,
        show_progress_bar=False
    )


pages = extract_text_from_pdf(PDF_PATH)

chunks = semantic_chunk_pages(
    pages,
    embed,
    breakpoint_percentile=BREAKPOINT_PERCENTILE,
    max_words=MAX_WORDS
)


if not chunks:
    raise ValueError("No chunks were created.")


vectors = embed([chunk["text"] for chunk in chunks])

# Start from an empty collection so no chunks from an older
# chunking strategy are left behind.
create_collection(
    vector_size=len(vectors[0]),
    recreate=True
)


points = [
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


client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)


print(f"Indexed {len(points)} chunks.")