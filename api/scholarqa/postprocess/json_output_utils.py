import re
from typing import Optional, Dict, Any, List
from scholarqa.utils import make_int
from langsmith import traceable
import logging
from anyascii import anyascii

logger = logging.getLogger(__name__)


def safe_json_parse(raw_output: Any) -> Any:
    """
    Best-effort JSON parser for LLM outputs.

    This function attempts to robustly parse JSON-like content produced by LLMs.
    It handles common issues such as:
    - Markdown code fences (```json ... ``` or ``` ... ```)
    - Leading/trailing commentary around the JSON
    - JSON arrays or objects embedded in text
    - Accidental single quotes for strings
    - Trailing commas in arrays/objects
    - CompletionCost objects from LLM calls

    Returns the parsed Python object on success, otherwise raises a ValueError.
    """
    import json
    import ast
    
    # Handle CompletionCost objects
    if hasattr(raw_output, 'content'):
        raw_output = raw_output.content
    elif not isinstance(raw_output, str):
        raw_output = str(raw_output)

    def strip_code_fences(text: str) -> str:
        text = text.strip()
        if text.startswith("```") and text.endswith("```"):
            # Remove the outermost fence and optional language tag
            inner = text[3:-3].strip()
            # If first line is a language tag like 'json' remove it
            first_newline = inner.find("\n")
            if first_newline != -1:
                lang = inner[:first_newline].strip().lower()
                if lang in {"json", "javascript", "ts", "python"}:
                    return inner[first_newline + 1 :].strip()
            return inner
        return text

    def extract_json_slice(text: str) -> str:
        # Try to find the first top-level JSON array or object slice
        stack = []
        start = None
        for i, ch in enumerate(text):
            if ch in "[{":
                if not stack:
                    start = i
                stack.append(ch)
            elif ch in "]}":
                if stack:
                    stack.pop()
                    if not stack and start is not None:
                        return text[start : i + 1]
        return text

    def try_json_loads(text: str):
        return json.loads(text)

    def try_literal_eval(text: str):
        # Fallback: handle single quotes, trailing commas sometimes tolerated by ast.literal_eval
        return ast.literal_eval(text)

    if raw_output is None:
        raise ValueError("No content to parse")

    if isinstance(raw_output, (dict, list)):
        return raw_output

    text = str(raw_output)
    # 1) Strip fences
    text = strip_code_fences(text)
    # 2) Extract likely JSON slice
    candidate = extract_json_slice(text)

    # 3) Try strict json
    try:
        return try_json_loads(candidate)
    except Exception:
        pass

    # 4) Soft fixes: replace single quotes with double quotes where safe
    soft = candidate
    if "'" in soft and '"' not in soft:
        soft = soft.replace("'", '"')
        try:
            return try_json_loads(soft)
        except Exception:
            pass

    # 5) Fallback to literal_eval for python-like structures
    try:
        return try_literal_eval(candidate)
    except Exception:
        pass

    # 6) Last attempt: parse the whole text
    try:
        return try_json_loads(text)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON output: {e}; raw={raw_output}")


def find_tldr_super_token(text: str) -> Optional[str]:
    # First, find the first instance of any token that has text "tldr" or "TLDR" in it, considering word boundaries
    tldr_token = re.search(r"\b\w*tldr\w*\b", text, re.IGNORECASE)

    if tldr_token:
        tldr_token = tldr_token.group(0)
        # Now find the word or token that the tldr_token is a subtoken in. This includes punctuation and markdown symbols
        tldr_super_token_pattern = re.compile(rf"[^\s]*{re.escape(tldr_token)}[^\s]*", re.IGNORECASE)
        tldr_super_token = re.search(tldr_super_token_pattern, text)

        if tldr_super_token:
            return tldr_super_token.group(0)
        else:
            return None
    else:
        return None


def get_section_text(gen_text: str) -> Dict[str, Any]:
    # Assume each section starts with 'Section X' followed by 'TLDR:'
    # find the first instance of any token surrounded by spaces or newlines that has text "tldr" or "TLDR" in it
    tldr_token = find_tldr_super_token(gen_text)
    curr_section = dict()
    if tldr_token is not None:
        parts = gen_text.split(tldr_token)
    else:
        parts = [gen_text]
    try:
        if len(parts) > 1:
            title = parts[0].strip()
            title = re.sub(r"\s*\(list\)", "", title)
            title = re.sub(r"\s*\(synthesis\)", "", title)
            curr_section["title"] = title.strip('#').strip()
            if tldr_token is not None:
                text_parts = parts[1].strip().split("\n", 1)
                tldr = text_parts[0]  # Assume TLDR is a single line
                text = text_parts[1]
                curr_section["tldr"] = tldr.strip('#').strip()
            else:
                text = parts[1].strip()
            curr_section["text"] = text
        else:
            raise Exception("Invalid content generated for the query by the LLM")
    except Exception as e:
        logger.exception(f"Error while parsing llm gen text: {gen_text} - {e}")
        raise e

    return curr_section


def resolve_ref_id(ref_str, ref_corpus_id, citation_ids):
    # in case of multiple papers from same author in the same year, add a count suffix
    if ref_str not in citation_ids:
        citation_ids[ref_str] = dict()
    if ref_corpus_id not in citation_ids[ref_str]:
        if citation_ids[ref_str]:
            rfsplits = ref_str.split(",")
            # in case of 2 (Doe et al., 2024), the one found later becomes (Doe et al._1, 2024) and so on...
            if len(rfsplits) > 1:
                ref_str_id = f"{rfsplits[0]}_{len(citation_ids[ref_str])},{rfsplits[1]}"
            else:
                ref_str_id = f"{ref_str}_{len(citation_ids[ref_str])}"
        else:
            ref_str_id = ref_str
        citation_ids[ref_str][ref_corpus_id] = ref_str_id
    else:
        ref_str_id = citation_ids[ref_str][ref_corpus_id]
    return ref_str_id


def pop_ref_data(ref_str_id, ref_corpus_id, fixed_quote, curr_paper_metadata) -> Dict[str, Any]:
    curr_ref = dict()
    curr_ref["id"] = ref_str_id
    curr_ref["snippets"] = [fq.strip() for fq in fixed_quote.split("...")]
    curr_ref["paper"] = dict()
    # Keep IDs as strings to support arXiv-style IDs like "2509.01324"
    curr_ref["paper"]["corpus_id"] = str(ref_corpus_id) if ref_corpus_id is not None else ""
    if curr_paper_metadata:
        #Commenting out the open access check as we switch to s2 api for the open access logic

        # if not (curr_paper_metadata.get("isOpenAccess") and curr_paper_metadata.get("openAccessPdf")):
        #     if curr_paper_metadata.get("abstract"):
        #         curr_ref["snippets"] = [s for s in curr_ref["snippets"] if
        #                                 s[:100] not in curr_paper_metadata["abstract"]]
        #     if not curr_ref["snippets"]:
        #         curr_ref["snippets"] = ["Please click on the paper title to read the abstract on Semantic Scholar."]

        curr_ref["score"] = curr_paper_metadata.get("relevance_judgement", 0)
        curr_ref["paper"]["title"] = curr_paper_metadata["title"]
        curr_ref["paper"]["authors"] = curr_paper_metadata["authors"]
        curr_ref["paper"]["year"] = make_int(curr_paper_metadata.get("year", 0))
        curr_ref["paper"]["venue"] = curr_paper_metadata["venue"]
        curr_ref["paper"]["n_citations"] = curr_paper_metadata["citationCount"]
    return curr_ref


@traceable(name="Postprocessing: Converted LLM generated output to json summary")
def get_json_summary(llm_model: str, summary_sections: List[str], summary_quotes: Dict[str, Any],
                     paper_metadata: Dict[str, Any], citation_ids: Dict[str, Dict[int, str]],
                     inline_tags=False) -> List[Dict[str, Any]]:
    # Include both corpusId and arxivId for forward-compat UI usage; values are identical for now
    text_ref_format = '<Paper corpusId="{corpus_id}" arxivId="{corpus_id}" paperTitle="{ref_str}" isShortName></Paper>'
    sections = []
    llm_name_parts = llm_model.split("/", maxsplit=1)
    llm_ref_format = f'<Model name="{llm_name_parts[0].capitalize()}" version="{llm_name_parts[1]}">'
    summary_quotes = {anyascii(k): v for k, v in summary_quotes.items()}
    inline_citation_quotes = {anyascii(k): v for incite in summary_quotes.values() for k, v in
                              incite["inline_citations"].items()}

    def normalize_id(id_token: str) -> str:
        """Normalize paper ID tokens for robust matching.
        - Strip leading labels like 'paperId ' or 'arXiv:'
        - Trim surrounding whitespace and trailing punctuation like '.' or ','
        - Keep dots inside (for arXiv IDs), keep case and digits
        """
        if id_token is None:
            return ""
        norm = anyascii(str(id_token)).strip()
        # Remove common prefixes
        for prefix in ["paperid ", "paperId ", "arxiv:", "arXiv:"]:
            if norm.startswith(prefix):
                norm = norm[len(prefix):]
        # Trim trailing punctuation
        norm = norm.rstrip(".,;: ]")
        # Trim stray leading '[' if present
        norm = norm.lstrip("[")
        return norm

    # Build maps from normalized ID -> canonical bracket key
    def key_to_norm_id(key: str) -> str:
        try:
            token = key[1:-1].split(" | ")[0]
            return normalize_id(token)
        except Exception:
            return ""

    id_to_main_key = {key_to_norm_id(k): k for k in summary_quotes.keys()}
    id_to_inline_key = {key_to_norm_id(k): k for k in inline_citation_quotes.keys()}
    for sec in summary_sections:
        curr_section = get_section_text(sec)
        text = curr_section["text"]
        if curr_section:
            # Allow dotted IDs (e.g., arXiv formats) in adjacency fix
            pattern = r"(?:; )?(\S+ \| [A-Za-z. ]+ \| \d+ \| Citations: \d+)"
            replacement = r"] [\1"
            text = re.sub(pattern, replacement, text)
            text = re.sub(r"\[\]", "", text)
            curr_section["text"] = text.replace("[LLM MEMORY | 2024]", llm_ref_format)
            refs_list = []
            # tool tips inserted via span tags
            references = re.findall(r"\[.*?\]", text)
            refs_done = set()

            for ref in references:
                ref = anyascii(ref)
                resolved_key = None
                resolved_source = None  # 'main' or 'inline'
                if ref in summary_quotes or ref in inline_citation_quotes:
                    # Exact match
                    resolved_key = ref
                    resolved_source = 'main' if ref in summary_quotes else 'inline'
                else:
                    # Try ID-based resolution
                    try:
                        ref_token = ref[1:-1].split(" | ")[0]
                    except Exception:
                        ref_token = ref.strip("[]")
                    norm_ref_id = normalize_id(ref_token)

                    # Direct ID match first
                    if norm_ref_id in id_to_main_key:
                        resolved_key = id_to_main_key[norm_ref_id]
                        resolved_source = 'main'
                    elif norm_ref_id in id_to_inline_key:
                        resolved_key = id_to_inline_key[norm_ref_id]
                        resolved_source = 'inline'
                    else:
                        # Unique prefix match among available IDs
                        main_candidates = [k for k in id_to_main_key.keys() if k.startswith(norm_ref_id)] if norm_ref_id else []
                        inline_candidates = [k for k in id_to_inline_key.keys() if k.startswith(norm_ref_id)] if norm_ref_id else []
                        if len(main_candidates) == 1:
                            resolved_key = id_to_main_key[main_candidates[0]]
                            resolved_source = 'main'
                        elif len(inline_candidates) == 1:
                            resolved_key = id_to_inline_key[inline_candidates[0]]
                            resolved_source = 'inline'

                if resolved_key:
                    ref_parts = resolved_key[1:-1].split(" | ")
                    ref_corpus_id, ref_str = ref_parts[0], f"({ref_parts[1]}, {make_int(ref_parts[2])})".replace(
                        "NULL, ", "")
                    norm_done_id = normalize_id(ref_corpus_id)
                    if norm_done_id not in refs_done:
                        if resolved_source == 'main':
                            fixed_quote = summary_quotes[resolved_key]["quote"]
                        else:
                            # abstract for inline citation
                            fixed_quote = inline_citation_quotes[resolved_key]
                        fixed_quote = fixed_quote.strip().replace("“", '"').replace("”", '"')
                        if fixed_quote.startswith("..."):
                            fixed_quote = fixed_quote[3:]
                        if fixed_quote.endswith("..."):
                            fixed_quote = fixed_quote[:-3]
                        # dict to save reference strings as there is a possibility of having multiple papers in the same year from the same author
                        refs_done.add(norm_done_id)
                        ref_str_id = resolve_ref_id(ref_str, ref_corpus_id, citation_ids)
                        ref_data = pop_ref_data(ref_str_id, ref_corpus_id, fixed_quote,
                                                paper_metadata.get(ref_corpus_id))
                        if inline_tags:
                            curr_section["text"] = curr_section["text"].replace(ref, text_ref_format.format(
                                corpus_id=ref_data["paper"]["corpus_id"], ref_str=ref_data["id"]))
                        else:
                            curr_section["text"] = curr_section["text"].replace(ref, ref_data["id"])
                        refs_list.append(ref_data)
                else:
                    curr_section["text"] = curr_section["text"].replace(ref, "")
                    logger.warning(f"Reference not found in the summary quotes: {ref}")
            curr_section["text"] = re.sub(r"[ ]+", " ", curr_section["text"])
            # curr_section["text"] = curr_section["text"].replace(") ; (", "]; [")
            curr_section["citations"] = refs_list
            # add number of unique citations to section tldr
            if curr_section["tldr"]:
                if refs_list:
                    curr_section["tldr"] += (
                        f" ({len(refs_list)} sources)" if len(refs_list) > 1 else " (1 source)")
                else:
                    curr_section["tldr"] += " (LLM Memory)"
            sections.append(curr_section)
    return sections
