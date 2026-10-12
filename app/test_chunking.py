import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.ingestion.document_processor import process_document


PDF_PATH = "app/data/report.pdf"

for chunk_size, overlap in [
    (250, 0),
    (250, 50),
    (500, 50),
    (500, 100),
]:
    chunks = process_document(
        file_path=PDF_PATH,
        chunk_size=chunk_size,
        overlap=overlap
    )

    print(
        f"chunk_size={chunk_size}, "
        f"overlap={overlap} "
        f"→ {len(chunks)} chunks"
    )


for chunk in chunks[:5]:
    print("=" * 70)
    print(
        f"Page: {chunk['page']} | "
        f"Chunk: {chunk['chunk']}"
    )
    print("=" * 70)
    print(chunk["text"][:1000])
    print()