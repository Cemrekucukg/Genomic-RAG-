# Genomic-RAG for RARS1 (PubMed + Chroma + LLM)

This project implements a Genomic-RAG system for extracting structured information about the **RARS1 gene** from PubMed abstracts.

The system dynamically retrieves recent publications, stores them in a vector database, retrieves relevant snippets, and uses an LLM to generate structured, citation-grounded outputs.

---

## Design Decisions & Justifications

### 1. PubMed API Rate Limits Handling

The PubMed ingestion is implemented using Biopython's Entrez module.  
To comply with NCBI rate limits:

- An email address is provided via `Entrez.email`.
- Requests are batched (search → fetch pattern).
- The ingestion process avoids aggressive parallel calls.
- The system is designed to fetch a limited number of recent abstracts per run.

This ensures compliance with NCBI usage policies and prevents temporary IP blocking.

---

### 2. Embedding Model Choice

We selected `BAAI/bge-base-en-v1.5` for the following reasons:

- Strong semantic retrieval performance in scientific/technical text.
- Good balance between embedding quality and computational efficiency.
- Compatible with CPU environments (no GPU required).
- Performs well on biomedical-style terminology and mutation patterns.

The model enables reliable retrieval of abstracts mentioning specific gene variants.

---

### 3. Phenotype vs Variant Identification

We ensured correct separation between genetic variants and phenotypes using:

1. **Chunking Strategy**
   - Abstracts are chunked carefully to preserve mutation patterns such as:
     - `c.5A>G`
     - `p.Asp2Gly`
   - Chunking avoids splitting variant strings across boundaries.

2. **LLM Prompt Constraints**
   - The LLM is explicitly instructed to:
     - Extract variant identifiers separately.
     - Extract associated diseases and phenotypes separately.
     - Use only information present in retrieved snippets.

3. **Strict JSON Schema**
   - Variants are structured under:
     - `"variant"`
   - Clinical manifestations are structured under:
     - `"phenotypes"`
   - Diseases are structured under:
     - `"associated_diseases"`

4. **Hallucination Guardrail**
   - Validates that:
     - The extracted variant string appears in the retrieved context.
     - Citations exist in snippet headers.
   - Unsupported extractions are removed automatically.

This multi-layer approach minimizes misclassification between mutation identifiers and clinical features.

---

## System Architecture

1. **Dynamic Data Ingestion**
   - PubMed API (Entrez) used to fetch RARS1-related abstracts.
   - Metadata includes PMID and DOI.
   - Output stored as JSONL.

2. **Knowledge Processing & Storage**
   - Abstracts are chunked carefully to preserve variant patterns (e.g., `c.5A>G`, `p.Asp2Gly`).
   - Chunks embedded using `BAAI/bge-base-en-v1.5`.
   - Stored in ChromaDB (persistent local vector database).

3. **LLM Extraction Layer**
   - Retrieval returns top-K relevant snippets.
   - OpenAI model (`gpt-4o-mini`) generates structured JSON output.
   - Strict JSON Schema enforced.
   - Every extracted claim must include PMID/DOI citations.

4. **Hallucination Guardrail**
   - Validates:
     - Citations exist in retrieved snippets.
     - Variants actually appear in retrieved text.
   - Removes unsupported items.
   - Prevents uncited summary claims.

5. **Evaluation**
   - `--eval` runs:
     - Normal biomedical question
     - Trick question (e.g., Alzheimer’s association)
   - Results written to `eval_results.json`.

---

## Installation

```bash
pip install -r requirements.txt
```
## Usage

### 1. Ingest PubMed Data

```bash
python ingest.py
```

### 2. Chunk and Index

```bash
python chunk_and_index.py
```

### 3. Query

```bash
python main.py --query "What are the most recently reported variants in RARS1 and their associated symptoms?" --k 5
```

### 4. Run Evaluation

```bash
python main.py --eval --k 5
```

This generates `eval_results.json`.

---

## Design Choices

- **Embedding Model:** BGE-base for strong semantic retrieval.
- **Strict JSON Schema:** Prevents malformed outputs.
- **Guardrail Layer:** Ensures no hallucinated claims.
- **Citation Enforcement:** Every variant/disease claim must reference PMID or DOI.

---

## Deliverables

- `main.py`
- `ingest.py`
- `requirements.txt`
- `README.md`
- `eval_results.json`
