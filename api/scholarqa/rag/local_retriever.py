import requests
import logging
from typing import List, Dict, Any, Union
from scholarqa.rag.retriever_base import AbstractRetriever
from scholarqa.utils import make_int, METADATA_FIELDS, NUMERIC_META_FIELDS

logger = logging.getLogger(__name__)


class LocalhostRetriever(AbstractRetriever):
    def __init__(self, base_url: str = "http://localhost:8001/graph/v1/", n_retrieval: int = 256, n_keyword_srch: int = 20):
        self.base_url = base_url.rstrip("/") + "/"
        self.n_retrieval = n_retrieval
        self.n_keyword_srch = n_keyword_srch

    def retrieve_passages(self, query: str, **filter_kwargs) -> List[Dict[str, Any]]:
        """Query the localhost API snippet search endpoint to retrieve papers based on the query."""
        snippets_list = self.snippet_search(query, **filter_kwargs)
        snippets_list = [
            snippet for snippet in snippets_list if len(snippet["text"].split(" ")) > 20
        ]
        return snippets_list

    def snippet_search(self, query: str, **filter_kwargs) -> List[Dict[str, Any]]:
        if not self.n_retrieval:
            return []

        query_params = {fkey: fval for fkey, fval in filter_kwargs.items() if fval}
        query_params.update({"query": query, "limit": self.n_retrieval})

        try:
            response = requests.get(
                f"{self.base_url}snippet/search",
                params=query_params,
                timeout=30
            )
            response.raise_for_status()
            snippets = response.json()
        except Exception as e:
            logger.error(f"Error calling localhost snippet search: {e}")
            return []

        snippets_list = []
        res_data = snippets.get("data", snippets.get("snippets", []))

        if res_data:
            for fields in res_data:
                res_map = dict()
                snippet, paper = fields["snippet"], fields["paper"]
                res_map["corpus_id"] = str(paper["corpusId"])
                res_map["title"] = paper["title"]
                res_map["text"] = snippet["text"]
                res_map["score"] = fields["score"]
                res_map["section_title"] = snippet["snippetKind"]

                if snippet["snippetKind"] == "body":
                    if section := snippet.get("section"):
                        res_map["section_title"] = section

                if "snippetOffset" in snippet and snippet["snippetOffset"].get("start"):
                    res_map["char_start_offset"] = snippet["snippetOffset"]["start"]
                else:
                    res_map["char_start_offset"] = 0

                if "annotations" in snippet and "sentences" in snippet["annotations"] and snippet["annotations"]["sentences"]:
                    res_map["sentence_offsets"] = snippet["annotations"]["sentences"]
                else:
                    res_map["sentence_offsets"] = []

                if snippet.get("annotations") and snippet["annotations"].get("refMentions"):
                    res_map["ref_mentions"] = [rmen for rmen in
                                              snippet["annotations"]["refMentions"] if rmen.get("matchedPaperCorpusId")
                                              and rmen.get("start") and rmen.get("end")]
                else:
                    res_map["ref_mentions"] = []

                res_map["pdf_hash"] = snippet.get("extractionPdfHash", "")
                res_map["stype"] = "localhost"

                if res_map:
                    snippets_list.append(res_map)

        return snippets_list

    def retrieve_additional_papers(self, query: str, **filter_kwargs) -> List[Dict[str, Any]]:
        return self.keyword_search(query, **filter_kwargs) if self.n_keyword_srch else []

    def keyword_search(self, kquery: str, **filter_kwargs) -> List[Dict[str, Any]]:
        """Query the localhost API paper search endpoint and return top n papers with full metadata including authors."""
        paper_data = []
        query_params = {fkey: fval for fkey, fval in filter_kwargs.items() if fval}
        query_params.update({
            "query": kquery,
            "limit": self.n_keyword_srch,
            "fields": "authors,title,abstract,year,venue,citationCount,referenceCount,influentialCitationCount,corpusId"
        })

        try:
            response = requests.get(
                f"{self.base_url}paper/search",
                params=query_params,
                timeout=30
            )
            response.raise_for_status()
            res = response.json()
        except Exception as e:
            logger.error(f"Error calling localhost paper search: {e}")
            return []

        if "data" in res:
            paper_data = res["data"]
            paper_data = [pd for pd in paper_data if pd.get("corpusId") and pd.get("title") and pd.get("abstract")]

            for pd in paper_data:
                # Keep corpusId as string to handle IDs like "2504.05220"
                pd["corpus_id"] = str(pd["corpusId"])
                pd["text"] = pd["abstract"]
                pd["section_title"] = "abstract"
                pd["char_start_offset"] = 0
                pd["sentence_offsets"] = []
                pd["ref_mentions"] = []
                pd["score"] = 0.0
                pd["stype"] = "localhost_api"
                pd["pdf_hash"] = ""

                # Convert numeric fields
                for field in NUMERIC_META_FIELDS:
                    if field in pd:
                        pd[field] = make_int(pd[field])

        return paper_data

    def get_paper_metadata(self, corpus_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch metadata for papers by corpus IDs using the batch endpoint."""
        if not corpus_ids:
            return {}

        try:
            response = requests.post(
                f"{self.base_url}paper/batch",
                json={"ids": corpus_ids},
                params={"fields": "authors,title,abstract,year,venue,citationCount,referenceCount,influentialCitationCount,corpusId"},
                timeout=30
            )
            response.raise_for_status()
            res = response.json()
        except Exception as e:
            logger.error(f"Error calling localhost paper batch: {e}")
            return {}

        metadata = {}
        for paper in res:
            if paper:
                corpus_id = str(paper["corpusId"])
                # Convert numeric fields
                for field in NUMERIC_META_FIELDS:
                    if field in paper:
                        paper[field] = make_int(paper[field])
                metadata[corpus_id] = paper

        return metadata