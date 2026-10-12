import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
#--------------------------------------------------------------------------------------------
from app.ingestion.chunker import chunk_text


text = """
The company was founded in 1998 and initially operated
in three markets. During the following decade, the company
expanded into international markets and significantly
increased its workforce. Revenue also increased as demand
for its products grew.
"""


chunks = chunk_text(
    text=text,
    page_number=1,
    chunk_size=20,
    overlap=5
)


for chunk in chunks:
    print("=" * 60)
    print(f"Page: {chunk['page']}")
    print(f"Chunk: {chunk['chunk']}")
    print(chunk["text"])