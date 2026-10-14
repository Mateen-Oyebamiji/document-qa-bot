# Retrieval evaluation

Generated 2026-10-13 12:45 · 43 questions · embedding model `all-MiniLM-L6-v2`

## Chunking strategies

- `fixed_500_50`: Fixed 500 words / 50 overlap (current app setting)
- `fixed_250_50`: Fixed 250 words / 50 overlap
- `fixed_150_30`: Fixed 150 words / 30 overlap (fits MiniLM's 256-token window)
- `semantic_p90`: Semantic, break at 90th-percentile distance, max 200 words
- `semantic_p75`: Semantic, break at 75th-percentile distance, max 200 words

## Results

| Chunking | Retriever | Chunks | Median words | Truncated | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | Evidence@5 | Words in top 5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed_500_50 | vector | 124 | 281 | 68% | 65% | 79% | 84% | 91% | 0.74 | 78% | 1602 |
| fixed_500_50 | keyword | 124 | 281 | 68% | 60% | 84% | 95% | 98% | 0.74 | 95% | 1885 |
| fixed_500_50 | hybrid | 124 | 281 | 68% | 67% | 86% | 93% | 98% | 0.79 | 90% | 1721 |
| fixed_250_50 | vector | 214 | 250 | 62% | 49% | 74% | 86% | 88% | 0.64 | 80% | 957 |
| fixed_250_50 | keyword | 214 | 250 | 62% | 58% | 84% | 91% | 95% | 0.73 | 88% | 1090 |
| fixed_250_50 | hybrid | 214 | 250 | 62% | 63% | 84% | 88% | 98% | 0.75 | 88% | 1055 |
| fixed_150_30 | vector | 328 | 150 | 5% | 51% | 79% | 84% | 88% | 0.65 | 73% | 631 |
| fixed_150_30 | keyword | 328 | 150 | 5% | 58% | 88% | 91% | 98% | 0.74 | 85% | 707 |
| fixed_150_30 | hybrid | 328 | 150 | 5% | 70% | 81% | 86% | 100% | 0.77 | 78% | 679 |
| semantic_p90 | vector | 258 | 143 | 13% | 63% | 81% | 84% | 88% | 0.73 | 80% | 691 |
| semantic_p90 | keyword | 258 | 143 | 13% | 63% | 88% | 93% | 98% | 0.77 | 88% | 757 |
| semantic_p90 | hybrid | 258 | 143 | 13% | 67% | 84% | 88% | 95% | 0.77 | 83% | 719 |
| semantic_p75 | vector | 300 | 107 | 10% | 53% | 84% | 88% | 93% | 0.69 | 85% | 604 |
| semantic_p75 | keyword | 300 | 107 | 10% | 60% | 88% | 93% | 98% | 0.76 | 90% | 676 |
| semantic_p75 | hybrid | 300 | 107 | 10% | 74% | 86% | 95% | 98% | 0.82 | 93% | 629 |

- **Hit@k**: share of questions where a chunk from a gold page is in the top k.
- **MRR**: mean of 1/rank of the first gold-page chunk (0 if not in the top 10).
- **Evidence@5**: share of questions where the exact evidence text is inside a top-5 chunk (stricter; 41 questions have evidence strings).
- **Truncated**: chunks longer than the embedding model's input window. The model never sees the text past that point.
- **Words in top 5**: how much text the LLM would receive. Larger chunks buy hit rate with more context; compare at similar budgets.

## Hit@5 by question category

| Chunking | Retriever | chart (n=2) | fact (n=13) | multi_hop (n=5) | sustainability (n=9) | table (n=14) |
|---|---|---|---|---|---|---|
| fixed_500_50 | vector | 100% | 92% | 60% | 100% | 71% |
| fixed_500_50 | keyword | 100% | 92% | 80% | 100% | 100% |
| fixed_500_50 | hybrid | 100% | 100% | 60% | 100% | 93% |
| fixed_250_50 | vector | 100% | 92% | 60% | 100% | 79% |
| fixed_250_50 | keyword | 100% | 92% | 60% | 89% | 100% |
| fixed_250_50 | hybrid | 100% | 92% | 60% | 100% | 86% |
| fixed_150_30 | vector | 100% | 92% | 80% | 89% | 71% |
| fixed_150_30 | keyword | 100% | 100% | 60% | 78% | 100% |
| fixed_150_30 | hybrid | 100% | 92% | 60% | 89% | 86% |
| semantic_p90 | vector | 100% | 100% | 60% | 89% | 71% |
| semantic_p90 | keyword | 100% | 100% | 60% | 89% | 100% |
| semantic_p90 | hybrid | 100% | 100% | 40% | 100% | 86% |
| semantic_p75 | vector | 100% | 100% | 60% | 89% | 86% |
| semantic_p75 | keyword | 100% | 100% | 60% | 89% | 100% |
| semantic_p75 | hybrid | 100% | 100% | 60% | 100% | 100% |

## Hardest questions

Missed at k=5 by this many of the 15 configurations:

| Id | Question | Misses |
|---|---|---|
| q31 | Who owns LEGO A/S and in what proportions? | 11/15 |
| q14 | What was the LEGO Group's revenue in 2025? | 9/15 |
| q28 | Which region had the fastest growth in sale of goods in 2025? | 9/15 |
| q32 | How is free cash flow defined? | 9/15 |
| q15 | What were sales of goods in the Americas in 2025? | 7/15 |
| q12 | What is the outlook for 2026? | 5/15 |
| q17 | How much was spent on marketing in 2025? | 5/15 |
| q38 | How many tonnes of waste went to landfill from production sites in 2025, versus the target? | 5/15 |
| q16 | How much licence income did the LEGO Group earn in 2025? | 3/15 |
| q36 | What was the lost time injury rate in 2025? | 3/15 |

Per-question ranks are in `per_question.csv`.
