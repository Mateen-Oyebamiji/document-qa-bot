import fitz


def extract_text_from_pdf(file_path: str) -> list[dict]:
    document = fitz.open(file_path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text().strip()

        if text:
            pages.append({
                "page": page_number,
                "text": text
            })


    document.close()

    return pages