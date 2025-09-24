"""
Simple script to run a ScholarQA query and render results to an HTML file.

Usage:
  python display_results.py --query "What is RAG?"

This script constructs a minimal pipeline similar to api/test.py, executes
the query, converts the response sections into an HTML report, and opens it
in the default web browser. It maps custom <Paper ...></Paper> tags inside
the section text to clickable chips and prefers arXiv links when available.
"""

from __future__ import annotations

import argparse
import html
import os
import re
import tempfile
import webbrowser
from typing import Any, Dict, List

from scholarqa import ScholarQA
from scholarqa.rag.retrieval import PaperFinder
from scholarqa.rag.retriever_base import FullTextRetriever
from scholarqa.llms.constants import GEMINI_25_FLASH


def build_scholar_qa() -> ScholarQA:
    """Create a ScholarQA instance using a simple FullText retriever.

    Mirrors the minimal setup from api/test.py without external reranker.
    """
    retriever = FullTextRetriever(n_retrieval=20, n_keyword_srch=10)
    paper_finder = PaperFinder(retriever, n_rerank=-1, context_threshold=0.0)
    return ScholarQA(paper_finder=paper_finder, llm_model=GEMINI_25_FLASH)


def is_arxiv_id(value: str) -> bool:
    """Return True if value looks like an arXiv identifier (e.g., 2409.06679 or 2409.06679v2)."""
    return bool(re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", str(value or ""), flags=re.IGNORECASE))


def build_paper_href(arxiv_id: str | None, corpus_id: str | None) -> str:
    """Return a link to arXiv when arxiv_id looks valid, else Semantic Scholar by corpus id."""
    if arxiv_id and is_arxiv_id(arxiv_id):
        return f"https://arxiv.org/abs/{arxiv_id}"
    if corpus_id:
        return f"https://www.semanticscholar.org/p/{corpus_id}"
    return "#"


def parse_paper_tags(text: str, citations: List[Dict[str, Any]]) -> str:
    """Replace <Paper ...></Paper> tags with anchor chips.

    The function prefers arXiv links when the id pattern matches, otherwise falls back to S2 links.
    """
    # Prepare citation map by corpus_id for resolving titles when paperTitle is missing
    citation_map: Dict[str, Dict[str, Any]] = {}
    for c in citations or []:
        cid = str((c.get("paper") or {}).get("corpus_id") or "")
        if cid:
            citation_map[cid] = c

    def _replace(match: re.Match) -> str:
        attrs = match.group(1)
        attr_map: Dict[str, str] = {}
        for k, v in re.findall(r'(\w+)="([^"]*)"', attrs):
            attr_map[k] = v
        corpus_id = attr_map.get("corpusId") or attr_map.get("corpus_id") or ""
        arxiv_id = attr_map.get("arxivId") or ""
        label = attr_map.get("paperTitle") or (citation_map.get(corpus_id, {}).get("id") if corpus_id else None) or corpus_id or "Paper"
        full_title = (citation_map.get(corpus_id, {}).get("paper") or {}).get("title") if corpus_id else None
        href = build_paper_href(arxiv_id, corpus_id)
        return (
            f"<a class=\"paper-chip\" href=\"{html.escape(href)}\" target=\"_blank\" rel=\"noreferrer\" "
            f"title=\"{html.escape(full_title or label)}\">{html.escape(label)}</a>"
        )

    # Clean stray closing brackets the LLM may have produced near chips
    cleaned = re.sub(r"\s*\]\s*", " ", text or "")
    # Replace chips
    rendered = re.sub(r"<Paper\s+([^>]+)><\/Paper>", _replace, cleaned)
    # Collapse extra spaces
    rendered = re.sub(r"[ ]+", " ", rendered)
    return rendered


def render_html(payload: Dict[str, Any]) -> str:
    """Convert a ScholarQA response payload to a standalone HTML document string."""
    sections: List[Dict[str, Any]] = list(payload.get("sections") or [])
    head = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>ScholarQA Answer</title>
    <style>
      :root { --bg:#0a3235; --bg2:#0f2a31; --text:#faf2e9; --muted:#c6d6d7; --chip:#184a4f; --chipb:#24666d; }
      html, body { margin:0; padding:0; background:var(--bg); color:var(--text); font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }
      .container { max-width: 980px; margin: 24px auto; padding: 0 16px; }
      .section { background: var(--bg2); border: 1px solid #163940; border-radius: 10px; padding: 16px; margin: 16px 0; }
      .title { margin: 0 0 8px; font-size: 20px; }
      .tldr { margin: 0 0 12px; color: var(--muted); font-style: italic; }
      .text { line-height: 1.55; white-space: pre-wrap; }
      .citations { margin-top: 14px; border-top: 1px dashed #215058; padding-top: 12px; }
      .citations h4 { margin: 0 0 8px; font-size: 14px; color: var(--muted); font-weight: 600; letter-spacing: 0.02em; }
      .citation { background: #0c383f; border: 1px solid #1b4f56; border-radius: 8px; padding: 10px; margin: 8px 0; }
      .citation a { color: #8be1ff; text-decoration: none; }
      .citation a:hover { text-decoration: underline; }
      .meta { color: var(--muted); font-size: 12px; margin-top: 2px; }
      .snip { color: #cfe8e6; font-size: 12px; margin-top: 6px; }
      .paper-chip { display:inline-block; margin: 0 4px; padding: 2px 6px; border-radius: 999px; border: 1px solid var(--chipb); background: var(--chip); color: #c7f2ff; font-size: 12px; text-decoration: none; white-space: nowrap; }
    </style>
  </head>
  <body>
    <div class="container">
      <h2>ScholarQA Answer</h2>
    """
    body_parts: List[str] = []
    for section in sections:
        title = html.escape(str(section.get("title") or "Untitled section"))
        tldr = section.get("tldr")
        citations = section.get("citations") or []
        text_html = parse_paper_tags(str(section.get("text") or ""), citations)

        body_parts.append("<div class=\"section\">")
        body_parts.append(f"<h3 class=\"title\">{title}</h3>")
        if tldr:
            body_parts.append(f"<div class=\"tldr\">TLDR; {html.escape(str(tldr))}</div>")
        body_parts.append(f"<div class=\"text\">{text_html}</div>")

        if citations:
            body_parts.append("<div class=\"citations\"><h4>Citations</h4>")
            for c in citations:
                paper = c.get("paper") or {}
                cid = str(paper.get("corpus_id") or "")
                arx = cid if is_arxiv_id(cid) else None
                href = build_paper_href(arx, cid)
                title_p = paper.get("title") or c.get("id") or cid
                authors = ", ".join(a.get("name") for a in (paper.get("authors") or []) if a.get("name"))
                year = paper.get("year")
                venue = paper.get("venue")
                cites = paper.get("n_citations") or 0
                snip = (c.get("snippets") or [None])[0]
                body_parts.append(
                    """
                    <div class="citation">
                      <div><a href="%s" target="_blank" rel="noreferrer">%s</a></div>
                      <div class="meta">%s%s%s%s</div>
                      %s
                    </div>
                    """
                    % (
                        html.escape(href),
                        html.escape(str(title_p)),
                        html.escape(authors) + (". " if authors else ""),
                        html.escape(str(year)) if year else "",
                        (". " + html.escape(venue)) if venue else "",
                        (f" · {cites} citations" if cites else ""),
                        (f"<div class=\"snip\">{html.escape(snip)}</div>" if snip else ""),
                    )
                )
            body_parts.append("</div>")

        body_parts.append("</div>")

    tail = """
    </div>
  </body>
  </html>
    """
    return head + "\n".join(body_parts) + tail


def main() -> None:
    """Entry point: run query, render HTML, and open in browser."""
    parser = argparse.ArgumentParser(description="Run ScholarQA and render HTML output.")
    parser.add_argument("--query", required=True, help="User question to answer")
    args = parser.parse_args()

    sqa = build_scholar_qa()
    result = sqa.answer_query(args.query)

    html_doc = render_html(result)
    with tempfile.NamedTemporaryFile(prefix="scholarqa_", suffix=".html", delete=False, mode="w", encoding="utf-8") as f:
        f.write(html_doc)
        out_path = f.name

    print(f"Wrote HTML to: {out_path}")
    try:
        webbrowser.open(f"file://{out_path}")
    except Exception:
        pass


if __name__ == "__main__":
    main()


