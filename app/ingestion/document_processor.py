from app.ingestion.pdf_parser import extract_text_from_pdf
from app.ingestion.chunker import chunk_text


def process_document(
    file_path: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> list[dict]:

    pages = extract_text_from_pdf(file_path)

    all_chunks = []

    for page in pages:
        chunks = chunk_text(
            text=page["text"],
            page_number=page["page"],
            chunk_size=chunk_size,
            overlap=overlap
        )

        all_chunks.extend(chunks)

    return all_chunks