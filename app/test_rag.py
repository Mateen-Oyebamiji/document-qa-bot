from app.llm.rag_answer import answer_question

question = "What was LEGO group's profit before tax in 2025?"

result = answer_question(question)

print("\nANSWER")
print("=" * 80)

print(result["answer"])

print("\nRETRIEVED SOURCES")
print("=" * 80)

for source in result["sources"]:
    print(
        f"\nPage {source.payload['page']} "
        f"| Chunk {source.payload['chunk']} "
        f"| Score {source.score:.4f}"
    )

    print("-" * 80)
    print(source.payload["text"])
    print("-" * 80)