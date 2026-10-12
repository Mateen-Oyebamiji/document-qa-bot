from ingestion.pdf_parser import extract_text_from_pdf


PDF_PATH = "app/data/report.pdf"


pages = extract_text_from_pdf(PDF_PATH)

print(f"Extracted {len(pages)} pages\n")

for page in pages:
    print(
        f"Page {page['page']}: "
        f"{len(page['text'])} characters"
    )