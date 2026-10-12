from embeddings.embedder import create_embedding


text = "The company reported strong revenue growth."

embedding = create_embedding(text)

print(f"Number of dimensions: {len(embedding)}")
print(f"First 10 numbers: {embedding[:10]}")