from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path


ARXIV_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    published: str
    updated: str
    primary_category: str
    abs_url: str
    pdf_url: str


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _safe_id(raw: str) -> str:
    # Example IDs:
    # - http://arxiv.org/abs/2501.12345v2
    # - http://arxiv.org/abs/cs/9901001v1
    parts = raw.rstrip("/").split("/")
    return parts[-1] if parts else raw


def search_arxiv(topic: str, max_results: int = 24, timeout: int = 40) -> list[ArxivPaper]:
    q = _clean_text(topic)
    if not q:
        q = "reinforcement learning large language model"
    encoded = urllib.parse.quote_plus(f"all:{q}")
    url = (
        "http://export.arxiv.org/api/query"
        f"?search_query={encoded}&start=0&max_results={max(1, int(max_results))}"
        "&sortBy=relevance&sortOrder=descending"
    )

    with urllib.request.urlopen(url, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")

    root = ET.fromstring(body)
    papers: list[ArxivPaper] = []
    for entry in root.findall("atom:entry", ARXIV_ATOM_NS):
        abs_url = _clean_text(entry.findtext("atom:id", default="", namespaces=ARXIV_ATOM_NS))
        arxiv_id = _safe_id(abs_url)
        title = _clean_text(entry.findtext("atom:title", default="", namespaces=ARXIV_ATOM_NS))
        abstract = _clean_text(entry.findtext("atom:summary", default="", namespaces=ARXIV_ATOM_NS))
        published = _clean_text(entry.findtext("atom:published", default="", namespaces=ARXIV_ATOM_NS))
        updated = _clean_text(entry.findtext("atom:updated", default="", namespaces=ARXIV_ATOM_NS))

        authors = []
        for author in entry.findall("atom:author", ARXIV_ATOM_NS):
            name = _clean_text(author.findtext("atom:name", default="", namespaces=ARXIV_ATOM_NS))
            if name:
                authors.append(name)

        primary_category = ""
        primary = entry.find("arxiv:primary_category", ARXIV_ATOM_NS)
        if primary is not None:
            primary_category = _clean_text(primary.attrib.get("term", ""))

        pdf_url = ""
        for link in entry.findall("atom:link", ARXIV_ATOM_NS):
            href = _clean_text(link.attrib.get("href", ""))
            title_attr = _clean_text(link.attrib.get("title", ""))
            rel = _clean_text(link.attrib.get("rel", ""))
            typ = _clean_text(link.attrib.get("type", ""))
            if title_attr == "pdf" or (rel == "related" and typ == "application/pdf"):
                pdf_url = href
                break
        if not pdf_url and abs_url:
            pdf_url = abs_url.replace("/abs/", "/pdf/") + ".pdf"

        papers.append(
            ArxivPaper(
                arxiv_id=arxiv_id,
                title=title,
                abstract=abstract,
                authors=authors,
                published=published,
                updated=updated,
                primary_category=primary_category,
                abs_url=abs_url,
                pdf_url=pdf_url,
            )
        )
    return papers


def save_papers_json(papers: list[ArxivPaper], out_path: Path) -> None:
    payload = [asdict(p) for p in papers]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def papers_to_digest_markdown(topic: str, papers: list[ArxivPaper], top_k: int = 12) -> str:
    lines: list[str] = []
    lines.append(f"# Literature Digest: {topic}")
    lines.append("")
    lines.append(f"- Total retrieved from arXiv: {len(papers)}")
    lines.append(f"- Included in digest: {min(len(papers), max(1, top_k))}")
    lines.append("")
    lines.append("## Papers")
    lines.append("")

    for idx, p in enumerate(papers[: max(1, int(top_k))], start=1):
        author_text = ", ".join(p.authors[:4])
        if len(p.authors) > 4:
            author_text += ", et al."
        lines.append(f"### {idx}. {p.title}")
        lines.append(f"- arXiv: `{p.arxiv_id}`")
        lines.append(f"- Category: `{p.primary_category or 'unknown'}`")
        lines.append(f"- Updated: {p.updated or 'unknown'}")
        lines.append(f"- Authors: {author_text or 'unknown'}")
        lines.append(f"- URL: {p.abs_url}")
        lines.append(f"- PDF: {p.pdf_url}")
        lines.append(f"- Abstract: {p.abstract}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def _safe_filename(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip())
    return s[:120] or "paper"


def download_pdfs(
    papers: list[ArxivPaper],
    out_dir: Path,
    max_count: int = 6,
    timeout: int = 60,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for p in papers[: max(0, int(max_count))]:
        if not p.pdf_url:
            continue
        fn = _safe_filename(f"{p.arxiv_id}.pdf")
        dst = out_dir / fn
        try:
            with urllib.request.urlopen(p.pdf_url, timeout=timeout) as resp:
                data = resp.read()
            dst.write_bytes(data)
            paths.append(dst)
        except Exception:
            # Keep pipeline resilient; best-effort download.
            continue
    return paths
