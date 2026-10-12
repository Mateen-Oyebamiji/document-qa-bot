from ingestion.pdf_parser import extract_text_from_pdf


PDF_PATH = "app/data/report.pdf"


pages = extract_text_from_pdf(PDF_PATH)

print(f"Extracted {len(pages)} pages\n")

for page in pages[:3]:
    print("=" * 60)
    print(f"PAGE {page['page']}")
    print("=" * 60)
    print(page["text"][:1000])
    print()