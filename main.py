import argparse
import json
import os
import re
from typing import List, Dict, Any, Tuple

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer


PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "rars1_pubmed"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"

OPENAI_MODEL = "gpt-4o-mini"

DNA_VAR = re.compile(r"\bc\.\d+[A-Za-z0-9_+-]+>[A-Za-z0-9_+-]+\b")
PROT_VAR = re.compile(r"\bp\.[A-Za-z]+\d+[A-Za-z]+\b")

def load_collection():
    client = chromadb.PersistentClient(
        path=PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(name=COLLECTION_NAME)


def retrieve(query: str, k: int = 5) -> List[Dict[str, Any]]:
    model = SentenceTransformer(EMBED_MODEL)
    q_emb = model.encode([query]).tolist()[0]

    col = load_collection()
    res = col.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for i in range(len(res["ids"][0])):
        hits.append(
            {
                "id": res["ids"][0][i],
                "distance": res["distances"][0][i],
                "text": res["documents"][0][i],
                "pmid": res["metadatas"][0][i].get("pmid"),
                "doi": res["metadatas"][0][i].get("doi"),
            }
        )
    return hits


def build_context(hits: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Dict[str, str]]]:
    """
    Returns:
      - context string to feed LLM
      - citation_map: allowed citations -> {"pmid":..., "doi":...}
    """
    lines = []
    citation_map: Dict[str, Dict[str, str]] = {}

    for idx, h in enumerate(hits, 1):
        pmid = h.get("pmid") or ""
        doi = h.get("doi") or ""
        cite_tokens = []
        if pmid:
            cite_tokens.append(f"PMID:{pmid}")
            citation_map[f"PMID:{pmid}"] = {"pmid": pmid, "doi": doi}
        if doi:
            cite_tokens.append(f"DOI:{doi}")
            citation_map[f"DOI:{doi}"] = {"pmid": pmid, "doi": doi}

        cite_header = " | ".join(cite_tokens) if cite_tokens else "NO_CITATION"
        lines.append(f"[SNIPPET {idx}] {cite_header}\n{h['text']}\n")

    return "\n".join(lines).strip(), citation_map


def llm_extract_structured(query: str, context: str) -> Dict[str, Any]:
    """
    LLM must:
      - Use ONLY the provided snippets
      - Every clinical/variant/disease claim must include PMID or DOI (from snippet headers)
      - Output JSON only (strict schema)
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment/.env")

    client = OpenAI(api_key=api_key)

    # JSON Schema (ONLY schema here; name/strict will be provided in text.format)
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "gene": {"type": "string"},
            "question": {"type": "string"},
            "summary": {"type": "string"},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "variant": {"type": "string"},
                        "associated_diseases": {"type": "array", "items": {"type": "string"}},
                        "phenotypes": {"type": "array", "items": {"type": "string"}},
                        "citations": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                        "evidence_snippet_numbers": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 1
                        }
                    },
                    "required": [
                        "variant",
                        "associated_diseases",
                        "phenotypes",
                        "citations",
                        "evidence_snippet_numbers"
                    ]
                }
            },
            "not_found": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "message": {"type": "string"},
                    "citations": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["message", "citations"]
            }
        },
        "required": ["gene", "question", "summary", "items", "not_found"]
    }

    instructions = (
        "You are a biomedical extraction assistant.\n"
        "TASK: Answer about the gene RARS1 using ONLY the provided SNIPPETS.\n"
        "Rules:\n"
        "1) Do NOT use outside knowledge.\n"
        "2) Extract specific variants (e.g., c.5A>G, p.Asp2Gly), associated diseases, and phenotypes.\n"
        "3) EVERY claim must include citations as PMID:<id> and/or DOI:<id> that appear in the snippet headers.\n"
        "4) If the question asks something not supported by snippets, put it in not_found.message and leave items empty.\n"
        "5) evidence_snippet_numbers must reference the SNIPPET numbers that support the claim.\n"
        "6) Output must strictly match the JSON schema.\n"
    )

    resp = client.responses.create(
        model=OPENAI_MODEL,
        instructions=instructions,
        input=f"QUESTION:\n{query}\n\nSNIPPETS:\n{context}\n",
        text={
            "format": {
                "type": "json_schema",
                "name": "rars1_rag_answer",
                "schema": schema,
                "strict": True
            }
        }
    )

    # The SDK returns the model output as text; for json_schema it should be valid JSON.
    out_text = resp.output_text
    return json.loads(out_text)


def guardrail_validate(
    llm_json: Dict[str, Any],
    allowed_citations: Dict[str, Dict[str, str]],
    context_text: str
) -> Dict[str, Any]:
    """
    Hallucination guardrail:
      - Remove items whose citations are not in allowed set
      - Remove items whose variant string does not appear in retrieved context
    """
    items = llm_json.get("items", [])
    kept = []
    removed = []

    for it in items:
        citations = it.get("citations", [])
        variant = (it.get("variant") or "").strip()

        # citations must be from retrieved snippet headers
        if not citations or any(c not in allowed_citations for c in citations):
            removed.append({"reason": "invalid_citation", "item": it})
            continue

        # variant must appear in retrieved text (simple but effective)
        if variant:
            v_ok = variant in context_text
            # if variant is like "p.Asp2Gly" it might appear as "(p.Asp2Gly)" in text
            if not v_ok and variant.startswith("p."):
                v_ok = f"({variant})" in context_text
            if not v_ok:
                removed.append({"reason": "variant_not_in_context", "item": it})
                continue

        kept.append(it)

    llm_json["items"] = kept
    llm_json["guardrail_removed_items"] = removed
    llm_json["guardrail_pass"] = (len(removed) == 0)

    if not kept:
        llm_json["items"] = []
        llm_json["summary"] = "No supported variant claims after guardrail validation."
        llm_json["not_found"] = {
            "message": "Not found in retrieved literature.",
            "citations": []
        }
    else:
        llm_json["summary"] = (
            "Summary omitted to avoid uncited claims. "
            "See items for cited evidence."
        )

        llm_json["not_found"] = {
            "message": "N/A",
            "citations": []
        }

    return llm_json

def run_once(query: str, k: int) -> Dict[str, Any]:
    hits = retrieve(query, k=k)
    context, citation_map = build_context(hits)
    llm_json = llm_extract_structured(query, context)
    final = guardrail_validate(llm_json, citation_map, context)
    final["_retrieval_debug"] = {
        "k": k,
        "snippets": [
            {"pmid": h["pmid"], "doi": h["doi"], "id": h["id"], "distance": h["distance"]}
            for h in hits
        ]
    }
    return final


def write_eval(eval_path: str, results: List[Dict[str, Any]]) -> None:
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="RARS1 Genomic-RAG (PubMed + Chroma + LLM + Guardrail)")
    parser.add_argument("--query", type=str, help="User question")
    parser.add_argument("--k", type=int, default=5, help="Top-K retrieval")
    parser.add_argument("--eval", action="store_true", help="Run normal + trick question and write eval_results.json")
    args = parser.parse_args()

    if args.eval:
        tests = [
            {
                "name": "normal_variants_and_symptoms",
                "question": "What are the most recently reported variants in RARS1 and their associated symptoms?"
            },
            {
                "name": "trick_unrelated_disease",
                "question": "Is RARS1 associated with Alzheimer's disease?"
            }
        ]
        out = []
        for t in tests:
            result = run_once(t["question"], k=args.k)
            out.append({"test": t["name"], "question": t["question"], "result": result})

        write_eval("eval_results.json", out)
        print("Wrote eval_results.json")
        return

    if not args.query:
        raise SystemExit("Provide --query or use --eval")

    result = run_once(args.query, k=args.k)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()