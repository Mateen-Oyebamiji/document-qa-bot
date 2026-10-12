# Document QA — Retrieval-Augmented Generation

A question-answering application that lets users ask questions about a long PDF report and receive answers grounded in retrieved document passages.

## Overview

Document QA uses a Retrieval-Augmented Generation (RAG) pipeline to find relevant sections of a document and provide them to a language model as context for answering questions.

The project explores how document chunking, semantic search, keyword search, and hybrid retrieval affect answer quality.

## Features

- Extract text from PDF documents.
- Split document text into chunks while preserving page metadata.
- Generate local text embeddings using Sentence Transformers.
- Store and retrieve document chunks with Qdrant.
- Combine semantic and keyword search using Reciprocal Rank Fusion.
- Generate context-grounded answers using the Gemini API.
- Inspect retrieved source passages through a Streamlit interface.

## Architecture

PDF → Text Extraction → Chunking → Embeddings → Qdrant → Retrieval → Gemini → Answer and Sources

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

## Current Limitations

- Retrieval can struggle with financial tables and exact numeric questions.
- Answer quality depends on the relevance of retrieved passages.
- The application depends on the availability and rate limits of the configured LLM API.
- Automated retrieval and answer evaluation is still being developed.

## Future Improvements

- Build a labelled evaluation dataset and measure retrieval performance.
- Improve retrieval for financial tables and numeric questions.
- Add automated tests and more robust API error handling.
- Evaluate answer accuracy and citation correctness.
- Improve the user interface and deployment experience.

## Learning Goals

This project explores the practical components of a RAG application, including document processing, vector databases, hybrid retrieval, LLM integration, and evaluation.
