from app.llm.client import client
from app.retrieval.search import search_documents


def answer_question(question: str):

    results = search_documents(
        question=question,
        limit=5
    )

    if not results:
        return {
            "answer": "I could not find relevant information in the document.",
            "sources": []
        }

    context_parts = []

    for index, result in enumerate(results, start=1):

        context_parts.append(
            f"""
SOURCE {index}
Page: {result.payload['page']}
Chunk: {result.payload['chunk']}

{result.payload['text']}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are a document question-answering assistant.

Your job is to answer the user's question using ONLY
the information contained in the provided sources.

Do not use outside knowledge.

If the sources do not contain enough information to answer
the question, say:

"I could not find the answer in the provided document."

When making a claim, cite the page that supports it.

Use this citation format:

[Page X]

You may cite multiple pages when necessary.

USER QUESTION:
{question}

SOURCES:
{context}
"""

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt
    )

    return {
        "answer": response.text,
        "sources": results
    }