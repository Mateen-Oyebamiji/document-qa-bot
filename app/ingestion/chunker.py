def chunk_text(
    text: str,
    page_number: int,
    chunk_size: int = 500,
    overlap: int = 100
) -> list[dict]:

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    words = text.split()

    chunks = []

    start = 0
    chunk_number = 0

    while start < len(words):
        end = start + chunk_size

        chunk_words = words[start:end]

        if not chunk_words:
            break

        chunk_number += 1

        chunks.append({
            "page": page_number,
            "chunk": chunk_number,
            "text": " ".join(chunk_words)
        })

        start = end - overlap

    return chunks