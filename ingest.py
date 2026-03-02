import json
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Dict

from Bio import Entrez
from dotenv import load_dotenv
from tqdm import tqdm


@dataclass
class PubMedArticle:
    pmid: str
    title: str
    abstract: str
    journal: str
    pub_year: str
    doi: Optional[str]


def _setup_entrez() -> None:
    load_dotenv()
    email = os.getenv("NCBI_EMAIL")
    api_key = os.getenv("NCBI_API_KEY")

    if not email:
        raise RuntimeError(
            "NCBI_EMAIL is required. Put it in .env as NCBI_EMAIL=you@example.com"
        )

    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key


def _sleep(api_key_present: bool) -> None:
    time.sleep(0.12 if api_key_present else 0.35)


def search_pmids(query: str, retmax: int = 30) -> List[str]:
    api_key_present = bool(getattr(Entrez, "api_key", None))
    _sleep(api_key_present)

    with Entrez.esearch(db="pubmed", term=query, sort="pub+date", retmax=retmax) as h:
        rec = Entrez.read(h)
    return rec.get("IdList", [])


def fetch_articles(pmids: List[str]) -> List[PubMedArticle]:
    if not pmids:
        return []

    api_key_present = bool(getattr(Entrez, "api_key", None))
    _sleep(api_key_present)

    with Entrez.efetch(
        db="pubmed",
        id=",".join(pmids),
        rettype="abstract",
        retmode="xml",
    ) as h:
        recs = Entrez.read(h)

    out: List[PubMedArticle] = []
    for item in recs.get("PubmedArticle", []):
        med = item.get("MedlineCitation", {})
        art = med.get("Article", {})

        pmid = str(med.get("PMID", "")).strip()
        title = str(art.get("ArticleTitle", "")).strip()

        abstract_text = ""
        abs_obj = art.get("Abstract")
        if abs_obj and "AbstractText" in abs_obj:
            abstract_text = " ".join([str(p) for p in abs_obj["AbstractText"]]).strip()

        journal = ""
        journal_info = art.get("Journal")
        if journal_info and "Title" in journal_info:
            journal = str(journal_info.get("Title", "")).strip()

        pub_year = ""
        ji = journal_info.get("JournalIssue") if journal_info else None
        pd = ji.get("PubDate") if ji else None
        if pd and "Year" in pd:
            pub_year = str(pd.get("Year", "")).strip()

        doi = None
        pubmed_data = item.get("PubmedData", {})
        for aid in pubmed_data.get("ArticleIdList", []):
            try:
                if aid.attributes.get("IdType") == "doi":
                    doi = str(aid)
                    break
            except Exception:
                pass

        out.append(
            PubMedArticle(
                pmid=pmid,
                title=title,
                abstract=abstract_text,
                journal=journal,
                pub_year=pub_year,
                doi=doi,
            )
        )

    return out


def write_jsonl(articles: List[PubMedArticle], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for a in articles:
            f.write(
                json.dumps(
                    {
                        "pmid": a.pmid,
                        "doi": a.doi,
                        "title": a.title,
                        "abstract": a.abstract,
                        "journal": a.journal,
                        "pub_year": a.pub_year,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def ingest_rars1(retmax: int = 30, out_path: str = "data/pubmed_rars1_abstracts.jsonl") -> Dict[str, int]:
    _setup_entrez()
    pmids = search_pmids("RARS1", retmax=retmax)
    articles = fetch_articles(pmids)
    write_jsonl(articles, out_path)

    return {"pmids": len(pmids), "articles": len(articles)}


if __name__ == "__main__":
    stats = ingest_rars1(retmax=30)
    print(stats)