from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ARXIV_ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}
RL_TERMS = {"rl", "reinforcement", "policy", "reward", "ppo", "grpo", "dpo", "preference", "preferences"}
LLM_TERMS = {"llm", "language", "languages", "reasoning", "transformer", "chatgpt", "instruction", "model", "models"}


@dataclass
class LiteraturePaper:
    source: str
    paper_id: str
    title: str
    abstract: str
    authors: list[str]
    published: str
    updated: str
    primary_category: str
    abs_url: str
    pdf_url: str
    citation_count: int | None = None


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _safe_id(raw: str) -> str:
    parts = (raw or "").rstrip("/").split("/")
    return parts[-1] if parts else raw


def _normalize_title(title: str) -> str:
    t = _clean_text(title).lower()
    return re.sub(r"[^a-z0-9]+", "", t)


def _topic_terms(topic: str) -> list[str]:
    stop = {
        "the",
        "and",
        "for",
        "with",
        "under",
        "from",
        "into",
        "using",
        "that",
        "this",
        "these",
        "those",
        "are",
        "was",
        "were",
        "have",
        "has",
        "had",
        "not",
        "you",
        "your",
    }
    out: list[str] = []
    for t in re.findall(r"[a-z0-9]+", topic.lower()):
        if (len(t) <= 2 and t != "rl") or t in stop:
            continue
        if t not in out:
            out.append(t)
    return out[:24]


def _relevance_score(p: LiteraturePaper, terms: list[str]) -> float:
    title = f" {p.title.lower()} "
    abstract = f" {p.abstract.lower()} "
    score = 0.0
    for t in terms:
        token = f" {t} "
        if token in title:
            score += 3.0
        elif token in abstract:
            score += 1.0

    if p.source == "arxiv":
        score += 0.15
    elif p.source == "semantic_scholar":
        score += 0.10
    elif p.source == "openalex":
        score += 0.05
    return score


def _has_any_term(text: str, terms: set[str]) -> bool:
    for t in terms:
        if f" {t} " in text:
            return True
    return False


def _query_demands_rl_llm(terms: list[str]) -> bool:
    return any(t in RL_TERMS for t in terms) and any(t in LLM_TERMS for t in terms)


def _paper_matches_rl_llm(p: LiteraturePaper) -> bool:
    text = f" {p.title.lower()} {p.abstract.lower()} "
    return _has_any_term(text, RL_TERMS) and _has_any_term(text, LLM_TERMS)


def _json_get(url: str, timeout: int = 40) -> dict:
    headers = {
        "User-Agent": "vibe-research/0.1 (+literature-multi-source)",
        "Accept": "application/json",
    }
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    return json.loads(body or "{}")


def _decode_openalex_abstract(inv: dict | None) -> str:
    if not isinstance(inv, dict) or not inv:
        return ""
    max_idx = -1
    for positions in inv.values():
        if isinstance(positions, list) and positions:
            max_idx = max(max_idx, max(positions))
    if max_idx < 0:
        return ""
    tokens = [""] * (max_idx + 1)
    for token, positions in inv.items():
        if not isinstance(positions, list):
            continue
        for pos in positions:
            if isinstance(pos, int) and 0 <= pos <= max_idx:
                tokens[pos] = token
    return _clean_text(" ".join(tokens))


def _search_arxiv(topic: str, max_results: int = 24, timeout: int = 40) -> list[LiteraturePaper]:
    q = _clean_text(topic) or "reinforcement learning large language model"
    encoded = urllib.parse.quote_plus(f"all:{q}")
    url = (
        "http://export.arxiv.org/api/query"
        f"?search_query={encoded}&start=0&max_results={max(1, int(max_results))}"
        "&sortBy=relevance&sortOrder=descending"
    )

    with urllib.request.urlopen(url, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")

    root = ET.fromstring(body)
    papers: list[LiteraturePaper] = []
    for entry in root.findall("atom:entry", ARXIV_ATOM_NS):
        abs_url = _clean_text(entry.findtext("atom:id", default="", namespaces=ARXIV_ATOM_NS))
        paper_id = _safe_id(abs_url)
        title = _clean_text(entry.findtext("atom:title", default="", namespaces=ARXIV_ATOM_NS))
        abstract = _clean_text(entry.findtext("atom:summary", default="", namespaces=ARXIV_ATOM_NS))
        published = _clean_text(entry.findtext("atom:published", default="", namespaces=ARXIV_ATOM_NS))
        updated = _clean_text(entry.findtext("atom:updated", default="", namespaces=ARXIV_ATOM_NS))

        authors: list[str] = []
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
            LiteraturePaper(
                source="arxiv",
                paper_id=paper_id,
                title=title,
                abstract=abstract,
                authors=authors,
                published=published,
                updated=updated,
                primary_category=primary_category,
                abs_url=abs_url,
                pdf_url=pdf_url,
                citation_count=None,
            )
        )
    return papers


def _search_semantic_scholar(topic: str, max_results: int = 24, timeout: int = 40) -> list[LiteraturePaper]:
    q = _clean_text(topic) or "reinforcement learning large language model"
    fields = [
        "paperId",
        "title",
        "abstract",
        "authors",
        "externalIds",
        "url",
        "publicationDate",
        "year",
        "fieldsOfStudy",
        "openAccessPdf",
        "citationCount",
    ]
    params = urllib.parse.urlencode(
        {
            "query": q,
            "limit": max(1, min(int(max_results), 100)),
            "fields": ",".join(fields),
        }
    )
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?{params}"
    payload = _json_get(url, timeout=timeout)

    papers: list[LiteraturePaper] = []
    for item in payload.get("data", []) or []:
        if not isinstance(item, dict):
            continue
        external = item.get("externalIds") or {}
        arxiv_id = _clean_text(str(external.get("ArXiv", ""))) if isinstance(external, dict) else ""
        pid = arxiv_id or _clean_text(str(item.get("paperId", ""))) or "unknown"
        title = _clean_text(str(item.get("title", "")))
        abstract = _clean_text(str(item.get("abstract", "")))

        authors: list[str] = []
        for a in item.get("authors", []) or []:
            if isinstance(a, dict):
                n = _clean_text(str(a.get("name", "")))
                if n:
                    authors.append(n)

        pub_date = _clean_text(str(item.get("publicationDate", "")))
        if not pub_date:
            y = _clean_text(str(item.get("year", "")))
            pub_date = y

        fields_of_study = item.get("fieldsOfStudy", []) or []
        cat = ""
        if isinstance(fields_of_study, list) and fields_of_study:
            cat = _clean_text(str(fields_of_study[0]))

        abs_url = _clean_text(str(item.get("url", "")))
        pdf_url = ""
        oa = item.get("openAccessPdf")
        if isinstance(oa, dict):
            pdf_url = _clean_text(str(oa.get("url", "")))

        cite = item.get("citationCount")
        try:
            citation_count = int(cite) if cite is not None else None
        except (TypeError, ValueError):
            citation_count = None

        papers.append(
            LiteraturePaper(
                source="semantic_scholar",
                paper_id=pid,
                title=title,
                abstract=abstract,
                authors=authors,
                published=pub_date,
                updated=pub_date,
                primary_category=cat,
                abs_url=abs_url,
                pdf_url=pdf_url,
                citation_count=citation_count,
            )
        )
    return papers


def _search_openalex(topic: str, max_results: int = 24, timeout: int = 40) -> list[LiteraturePaper]:
    q = _clean_text(topic) or "reinforcement learning large language model"
    params = urllib.parse.urlencode(
        {
            "search": q,
            "per-page": max(1, min(int(max_results), 200)),
            "sort": "relevance_score:desc",
        }
    )
    url = f"https://api.openalex.org/works?{params}"
    payload = _json_get(url, timeout=timeout)

    papers: list[LiteraturePaper] = []
    for item in payload.get("results", []) or []:
        if not isinstance(item, dict):
            continue
        display_name = _clean_text(str(item.get("display_name", "")))
        abstract = _decode_openalex_abstract(item.get("abstract_inverted_index"))

        authors: list[str] = []
        for auth in item.get("authorships", []) or []:
            if not isinstance(auth, dict):
                continue
            author = auth.get("author")
            if isinstance(author, dict):
                name = _clean_text(str(author.get("display_name", "")))
                if name:
                    authors.append(name)

        published = _clean_text(str(item.get("publication_date", "")))
        updated = _clean_text(str(item.get("updated_date", ""))) or published

        cat = ""
        pt = item.get("primary_topic")
        if isinstance(pt, dict):
            field = pt.get("field")
            if isinstance(field, dict):
                cat = _clean_text(str(field.get("display_name", "")))

        openalex_id = _safe_id(_clean_text(str(item.get("id", ""))))
        doi = _clean_text(str(item.get("doi", "")))
        pid = doi or openalex_id or "unknown"

        abs_url = ""
        primary_location = item.get("primary_location")
        if isinstance(primary_location, dict):
            abs_url = _clean_text(str(primary_location.get("landing_page_url", "")))

        if not abs_url:
            abs_url = doi or _clean_text(str(item.get("id", "")))

        pdf_url = ""
        oa = item.get("open_access")
        if isinstance(oa, dict):
            pdf_url = _clean_text(str(oa.get("oa_url", "")))
        if not pdf_url:
            best_oa = item.get("best_oa_location")
            if isinstance(best_oa, dict):
                pdf_url = _clean_text(str(best_oa.get("pdf_url", "")))

        cited_by = item.get("cited_by_count")
        try:
            citation_count = int(cited_by) if cited_by is not None else None
        except (TypeError, ValueError):
            citation_count = None

        papers.append(
            LiteraturePaper(
                source="openalex",
                paper_id=pid,
                title=display_name,
                abstract=abstract,
                authors=authors,
                published=published,
                updated=updated,
                primary_category=cat,
                abs_url=abs_url,
                pdf_url=pdf_url,
                citation_count=citation_count,
            )
        )
    return papers


def _dedupe_by_title(papers: Iterable[LiteraturePaper]) -> list[LiteraturePaper]:
    seen: dict[str, LiteraturePaper] = {}
    for p in papers:
        key = _normalize_title(p.title)
        if not key:
            key = f"{p.source}:{p.paper_id}".lower()
        if key in seen:
            old = seen[key]
            old_has_pdf = bool(old.pdf_url)
            new_has_pdf = bool(p.pdf_url)
            old_cite = old.citation_count or -1
            new_cite = p.citation_count or -1
            if (new_has_pdf and not old_has_pdf) or (new_cite > old_cite):
                seen[key] = p
            continue
        seen[key] = p
    return list(seen.values())


def search_literature(
    topic: str,
    max_results: int = 24,
    sources: list[str] | None = None,
    timeout: int = 40,
) -> list[LiteraturePaper]:
    srcs = [s.strip().lower() for s in (sources or ["arxiv"]) if s.strip()]
    if not srcs:
        srcs = ["arxiv"]
    expanded_topic = (
        f"{_clean_text(topic)} reinforcement learning large language model reasoning preference optimization"
    ).strip()

    per_source = max(1, int(max_results) // max(1, len(srcs)))
    all_papers: list[LiteraturePaper] = []
    for src in srcs:
        try:
            if src == "arxiv":
                all_papers.extend(_search_arxiv(topic=expanded_topic, max_results=per_source, timeout=timeout))
            elif src in {"semantic_scholar", "semanticscholar"}:
                all_papers.extend(
                    _search_semantic_scholar(topic=expanded_topic, max_results=per_source, timeout=timeout)
                )
            elif src == "openalex":
                all_papers.extend(_search_openalex(topic=expanded_topic, max_results=per_source, timeout=timeout))
        except Exception:
            # Keep search resilient; one source failure should not block the cycle.
            continue

    deduped = _dedupe_by_title(all_papers)
    terms = _topic_terms(topic)
    if terms:
        ranked = sorted(deduped, key=lambda p: _relevance_score(p, terms), reverse=True)
        if _query_demands_rl_llm(terms):
            strict = [p for p in ranked if _paper_matches_rl_llm(p)]
            if len(strict) >= max(3, int(max_results) // 3):
                ranked = strict
        if ranked and _relevance_score(ranked[0], terms) > 0:
            ranked = [p for p in ranked if _relevance_score(p, terms) > 0]
        return ranked[: max(1, int(max_results))]
    return deduped[: max(1, int(max_results))]


def save_papers_json(papers: list[LiteraturePaper], out_path: Path) -> None:
    payload = [asdict(p) for p in papers]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def papers_to_digest_markdown(topic: str, papers: list[LiteraturePaper], top_k: int = 12) -> str:
    source_counts: dict[str, int] = {}
    for p in papers:
        source_counts[p.source] = source_counts.get(p.source, 0) + 1

    lines: list[str] = []
    lines.append(f"# Literature Digest: {topic}")
    lines.append("")
    lines.append(f"- Total retrieved: {len(papers)}")
    lines.append(f"- Included in digest: {min(len(papers), max(1, int(top_k)))}")
    if source_counts:
        parts = [f"{k}={v}" for k, v in sorted(source_counts.items())]
        lines.append(f"- Sources: {', '.join(parts)}")
    lines.append("")
    lines.append("## Papers")
    lines.append("")

    for idx, p in enumerate(papers[: max(1, int(top_k))], start=1):
        author_text = ", ".join(p.authors[:4])
        if len(p.authors) > 4:
            author_text += ", et al."
        lines.append(f"### {idx}. {p.title}")
        lines.append(f"- ID: `{p.source}:{p.paper_id}`")
        lines.append(f"- Source: `{p.source}`")
        lines.append(f"- Category: `{p.primary_category or 'unknown'}`")
        lines.append(f"- Published: {p.published or 'unknown'}")
        lines.append(f"- Updated: {p.updated or 'unknown'}")
        if p.citation_count is not None:
            lines.append(f"- CitationCount: {p.citation_count}")
        lines.append(f"- Authors: {author_text or 'unknown'}")
        lines.append(f"- URL: {p.abs_url}")
        lines.append(f"- PDF: {p.pdf_url or 'n/a'}")
        lines.append(f"- Abstract: {p.abstract}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def _safe_filename(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip())
    return s[:160] or "paper"


def download_pdfs(
    papers: list[LiteraturePaper],
    out_dir: Path,
    max_count: int = 6,
    timeout: int = 60,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for p in papers[: max(0, int(max_count))]:
        if not p.pdf_url:
            continue
        fn = _safe_filename(f"{p.source}_{p.paper_id}.pdf")
        dst = out_dir / fn
        try:
            with urllib.request.urlopen(p.pdf_url, timeout=timeout) as resp:
                data = resp.read()
            if not data:
                continue
            dst.write_bytes(data)
            paths.append(dst)
        except Exception:
            continue
    return paths
