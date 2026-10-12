

import re

from app.vector_store.qdrant_store import client, COLLECTION_NAME


STOP_WORDS = {
    "what", "was", "is", "the", "a", "an", "in",
    "on", "of", "to", "for", "and", "how", "much",
    "did", "do", "does", "were", "are", "with",
    "s"
}


def tokenize(text: str) -> list[str]:
    """Extract lowercase words and numbers."""
    return re.findall(r"[a-z]+|\d[\d,.-]*", text.lower())


def search_by_keywords(
    question: str,
    limit: int = 5
) -> list[dict]:

    query_tokens = [
        token
        for token in tokenize(question)
        if token not in STOP_WORDS
    ]

    if not query_tokens:
        return []

    query_terms = set(query_tokens)
    matching_chunks = []
    offset = None

    # Generate consecutive phrases from the question.
    phrases = []

    for length in range(len(query_tokens), 1, -1):
        for start in range(len(query_tokens) - length + 1):
            phrase = query_tokens[start:start + length]

            # Ignore phrases containing only numbers.
            if any(re.search(r"[a-z]", term) for term in phrase):
                phrases.append(phrase)

    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )

        for point in points:
            text = point.payload.get("text", "")
            chunk_tokens = tokenize(text)

            if not chunk_tokens:
                continue

            chunk_terms = set(chunk_tokens)
            matched_terms = query_terms & chunk_terms

            if not matched_terms:
                continue

            # Reward coverage of the important question terms.
            coverage = len(matched_terms) / len(query_terms)

            # Reward matching phrases, especially longer ones.
            phrase_score = 0

            for phrase in phrases:
                phrase_length = len(phrase)

                for i in range(
                    len(chunk_tokens) - phrase_length + 1
                ):
                    if chunk_tokens[i:i + phrase_length] == phrase:
                        phrase_score += phrase_length * 0.5
                        break

            score = coverage + phrase_score

            matching_chunks.append({
                "page": point.payload.get("page"),
                "chunk": point.payload.get("chunk"),
                "text": text,
                "score": score,
                "matched_terms": sorted(matched_terms),
                "point_id": point.id
            })

        if next_offset is None:
            break

        offset = next_offset

    matching_chunks.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return matching_chunks[:limit]

