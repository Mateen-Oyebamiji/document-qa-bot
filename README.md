# Document QA — Retrieval-Augmented Generation

A question-answering application that lets users ask questions about a long PDF report and receive answers grounded in retrieved document passages.

## Overview

Document QA uses a Retrieval-Augmented Generation (RAG) pipeline to find relevant sections of a document and provide them to a language model as context for answering questions.

The project explores how document chunking, semantic search, keyword search, and hybrid retrieval affect answer quality.

https://github.com/user-attachments/assets/77281d7e-f759-4c4c-87d3-69ced63f8a03

## Features

- Extract text from PDF documents.
- Split pages into **semantic chunks** that break where the topic changes, with page metadata preserved.
- Generate local text embeddings using Sentence Transformers (`all-MiniLM-L6-v2`).
- Store and retrieve document chunks with Qdrant.
- **Hybrid retrieval**: semantic and keyword search combined with Reciprocal Rank Fusion.
- Generate context-grounded answers with the Gemini API, with retries and fallback across models when one is overloaded.
- Inspect retrieved sources in a Streamlit interface, including whether each passage was found by vector search, keyword search, or both.
- **Retrieval evaluation harness** that compares chunking and retrieval strategies on 43 labelled questions.

## Architecture

```
PDF → Text extraction → Semantic chunking → Embeddings → Qdrant
                                                            │
Question ──► Vector search ─┐                               │
         └─► Keyword search ┴─► Reciprocal Rank Fusion ◄────┘
                                        │
                                  Top 5 passages ──► Gemini ──► Answer + page citations
```



## Technology Stack

- Python
- Streamlit
- PyMuPDF
- Sentence Transformers
- Qdrant
- Google Gemini API

## Setup

### 1. Install dependencies

Create and activate a Python virtual environment, then run:

```bash
pip install -r requirements.txt
```

### 2. Configure the API key

Copy the example environment file:

```bash
cp .env.example .env
```

Add your own Gemini API key to `.env`. Never commit this file.

### 3. Start Qdrant

Ensure Docker Desktop is running, then start Qdrant:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Keep the container running while using the application.

### 4. Add a PDF document

Place the PDF you are authorized to use at:

`app/data/report.pdf`

The report itself is not included in this repository.

### 5. Index the document

From the project root, run:

```bash
python -m app.ingestion.index_document
```

### 6. Launch the application

```bash
python -m streamlit run app/ui.py
```

## Results

The retrieval setup was chosen by measurement. Every configuration was scored on the same 43 labelled questions (see [Evaluation](#evaluation)).

| Setup | Hit@5 | MRR | Evidence@5 |
|---|---|---|---|
| Original: fixed 500-word chunks + vector search | 84% | 0.74 | 78% |
| Fixed 500-word chunks + hybrid search | 93% | 0.79 | 90% |
| **Current: semantic chunks (p75) + hybrid search** | **95%** | **0.82** | **93%** |

- **Hit@5**: the right page is among the 5 passages sent to the LLM.
- **MRR**: how high the first correct passage ranks (1.0 means always first).
- **Evidence@5**: the exact supporting text, e.g. `"Marketing 10,151"`, is inside one of those 5 passages. This is the strictest measure.

Compared question by question, the current setup **fixes 6 questions** the original missed and **breaks 1**. Full results for all 15 configurations are in [`evaluation/results/summary.md`](evaluation/results/summary.md).

## Problems Found and How They Were Addressed

Each problem below was found through the evaluation harness or while running the app, and each fix was checked by re-running the evaluation.

### 1. Vector search missed financial tables

**Evidence:** for "How much licence income did the LEGO Group earn in 2025?", vector search returned pages 11, 12 and 13, the Financial Review prose that *talks about* revenue, instead of page 51, which holds the revenue table. Five of the six questions fixed by the new setup were table lookups (revenue, Americas sales, licence income, marketing spend, regional growth).

**Cause:** table text is mostly bare numbers (`Licence income 749 670`), which embedding models represent poorly. Narrative text about the same topic looks more relevant.

**Fix:** hybrid search. The keyword half matches exact row labels such as "licence income" and "marketing", and RRF merges both rankings. This raised Evidence@5 from 78% to 90% with the same chunks.

### 2. Most chunks were longer than the embedding model can read

**Evidence:** `all-MiniLM-L6-v2` reads at most 256 tokens. **68%** of the original 500-word chunks were longer than that, measured with the model's own tokenizer, so the text past the limit was never embedded.

**Fix:** semantic chunks are capped at 200 words, and only 10% now go over the limit. Shrinking chunks alone did not fix vector search, though: blind 150-word chunks scored 84% Hit@5, no better than 500-word ones. See the next problem.

### 3. Short answers got lost inside long chunks

**Evidence:** "What is the outlook for 2026?" failed with fixed chunks. The outlook paragraph starts around word 524 of page 13 and shared a chunk with a long passage on compliance policies, so that chunk's single vector represented "policies" rather than "outlook".

**Fix:** semantic chunking splits where the meaning changes, so the outlook paragraph gets its own chunk. At a similar chunk size, semantic chunking beat blind fixed-size cutting on Evidence@5 by **12 points** (85% vs 73%, vector search), because facts stay together with their labels.

### 4. Re-indexing could leave stale chunks behind

**Cause:** `upsert` overwrites points with matching ids but never deletes. Re-indexing with a strategy that produces fewer chunks would leave old chunks in the collection, silently mixing two chunking strategies in search results.

**Fix:** indexing now deletes and recreates the collection (`create_collection(..., recreate=True)`).

### 5. The LLM API was unavailable for long periods

**Evidence:** `gemini-3.1-flash-lite` returned `503 UNAVAILABLE` on every call for an extended period. Two models that still appeared in the API's model list, `gemini-2.5-flash` and `gemini-2.5-flash-lite`, returned `404` because they had been retired.

**Fix:** temporary errors (429/500/503) are retried with exponential backoff. If a model is still unavailable, or has been retired (404), the next model in a fallback list is tried. Errors that would affect every model, such as an invalid API key, are raised immediately.

### 6. Development environment issues

- Streamlit's file watcher scanned every module in `transformers` and logged hundreds of `torchvision` import errors. It is now disabled in `.streamlit/config.toml`, so restart the app after editing code.
- With auto-reload off, an old Streamlit process kept serving outdated code on port 8501 while the new one started on 8502. Stop old processes before restarting.

## Current Limitations

These are known weaknesses, with the evidence for each:

- **Keyword search has no stemming.** "Who owns LEGO A/S?" fails because the question says "owns" and the page says "owned by". Keyword search never finds the right page, which drags it from rank 2 (vector only) down to rank 8 (hybrid).
- **Keyword search over-counts repeated terms.** "How is free cash flow defined?" got worse with hybrid search: page 47, which repeats "free cash flow" several times, now outranks the definitions table on page 49.
- **The evaluation set is small.** With 43 questions, each question is worth about 2.3 percentage points, so differences under about 5 points (1–2 questions) may be noise.
- **The questions favour keyword search.** Many were written using the report's own wording ("licence income", "equity ratio"). Keyword search scores 91–95%, which probably overstates how it would do on real users' paraphrased questions.
- **The settings were chosen and measured on the same questions.** Picking the best of 15 configurations on one set risks overfitting to it. A separate held-out set would give a fairer final score.
- **Only retrieval is evaluated.** The harness checks that the right passages reach the LLM, not that the answer or its page citations are correct.
- **Values that only appear in charts** have no reliable text to match, so two questions are scored on page only.
- **Gemini's free tier** can be unavailable or rate-limited; the fallback chain reduces but does not remove this.

## Future Improvements

In rough priority order:

1. **Add stemming or lemmatisation to keyword search** to fix the "owns"/"owned" blind spot, then re-run the evaluation to confirm it helps.
2. **Add paraphrased questions** to test retrieval when users don't use the document's wording.
3. **Split the dataset into tuning and held-out questions**, so the final score isn't measured on the questions used to choose settings.
4. **Evaluate answer quality and citation accuracy**, including questions the document cannot answer, where the correct behaviour is to say so.
5. **Try a reranker** to fix ordering problems like the free cash flow definition.
6. **Table-aware chunking**, keeping table rows together with their column headers.
7. **Lower the semantic chunk cap** (e.g. 150 words) so fewer chunks go over the embedding model's limit.

## Evaluation

`evaluation/` measures how often retrieval finds the right passage.

```bash
python -m evaluation.run_retrieval_eval
```

The harness indexes `app/data/report.pdf` once per chunking strategy in an in-memory Qdrant, so it needs neither Docker nor the Gemini key. It runs every labelled question through vector, keyword and hybrid retrieval and writes reports to `evaluation/results/`:

- `summary.md`: all configurations, hit rate by question category, and the hardest questions.
- `summary.json`: the same numbers in machine-readable form.
- `per_question.csv`: gold and retrieved pages for every question, for digging into misses.

Useful flags: `--chunkers fixed_500_50,semantic_p75`, `--retrievers vector,hybrid`, `--k 1,3,5,10`.

**Dataset:** `evaluation/datasets/lego_2025_retrieval.jsonl` contains 43 questions across five categories (facts, table lookups, multi-part, sustainability KPIs, chart values). Each question has the PDF page(s) where the answer is stated and a short evidence quote copied from that page. Verify the labels against the PDF with:

```bash
python -m evaluation.check_dataset
```

**Strategies compared:** fixed-size chunks of 500, 250 and 150 words, and semantic chunks breaking at the 90th and 75th percentile of sentence-to-sentence distance, each with vector, keyword and hybrid retrieval. The 150-word fixed chunker is a size-matched control for the semantic chunker, so their comparison isolates the effect of cutting by meaning from the effect of chunk size.
## Learning Goals

This project explores the practical components of a RAG application, including document processing, vector databases, hybrid retrieval, LLM integration, and evaluation. In particular, it practises using measurement to choose between design options and to find out *why* a pipeline fails.

