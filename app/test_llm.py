from app.llm.client import client


interaction = client.interactions.create(
    model="gemini-3.1-flash-lite",
    input="Explain what a database is in one sentence."
)


print(interaction.output_text)