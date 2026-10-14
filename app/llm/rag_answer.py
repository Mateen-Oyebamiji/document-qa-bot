import time

from google.genai import errors

from app.llm.client import client
from app.retrieval.hybrid_search import hybrid_search


# Tried in order. If a model is still overloaded after its retries,
# the next one is used. Run `python gemini_check.py` to see which
# models your API key can reach.
MODEL_NAMES = [
    "gemini-flash-lite-latest",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
]

# Gemini status codes worth retrying: rate limited (429) or
# temporarily overloaded/unavailable (500, 503).
RETRYABLE_CODES = {429, 500, 503}
FALLBACK_CODES = RETRYABLE_CODES | {404}


def generate_with_retry(
    prompt: str,
    model: str,
    attempts: int = 2,
    base_delay: float = 2.0
):
    """
    Call one Gemini model, retrying temporary failures with exponential
    backoff (wait 2s, then 4s, ...). Other errors are raised immediately.
    """

    for attempt in range(1, attempts + 1):
        try:
            return client.models.generate_content(
                model=model,
                contents=prompt
            )
        except errors.APIError as error:
            if error.code not in RETRYABLE_CODES or attempt == attempts:
                raise

            delay = base_delay * 2 ** (attempt - 1)
            print(
                f"{model} returned {error.code} "
                f"(attempt {attempt}/{attempts}); retrying in {delay:.0f}s"
            )
            time.sleep(delay)


def generate_with_fallback(prompt: str):
    """
    Try each model in MODEL_NAMES until one answers.

    Model-specific errors (overloaded, rate limited, retired) move on
    to the next model; anything else, like a bad API key, is raised
    at once because every model would fail the same way.
    """

    last_error = None

    for model in MODEL_NAMES:
        try:
            response = generate_with_retry(prompt, model)
            print(f"Answered by {model}")
            return response
        except errors.APIError as error:
            if error.code not in FALLBACK_CODES:
                raise

            print(f"{model} unavailable ({error.code}); trying the next model")
            last_error = error

    raise last_error

def answer_question(question: str):

    # Hybrid (vector + keyword) retrieval scored best in the retrieval
    # evaluation, especially on table and number questions.
    results = hybrid_search(
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
Page: {result['page']}
Chunk: {result['chunk']}

{result['text']}
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