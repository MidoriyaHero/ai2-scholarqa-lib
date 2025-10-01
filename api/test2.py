"""
Test script for step_draft_outline function only.
Tests outline generation from rewritten queries.
"""

from scholarqa.rag.retrieval import PaperFinder
from scholarqa.rag.retriever_base import FullTextRetriever
from scholarqa import ScholarQA
from scholarqa.llms.constants import GEMINI_25_FLASH
from scholarqa.llms.litellm_helper import CostReportingArgs
# Minimal setup
retriever = FullTextRetriever(n_retrieval=10, n_keyword_srch=5)
paper_finder = PaperFinder(retriever, n_rerank=-1, context_threshold=0.0)
scholar_qa = ScholarQA(paper_finder=paper_finder, llm_model=GEMINI_25_FLASH)

test_queries = [
    "What is Computer Vision?",
    "What is the difference between stemming and lemmatization in NLP?",
    "How are large language models used for sentiment analysis of social media data?",
    "Compare sequence-to-sequence models and transformers for machine translation tasks.",
    "Discuss recent advances in natural language processing research.",
    "What are the current limitations and future directions of named entity recognition in NLP?",
    "How is NLP applied in healthcare, especially for processing clinical notes?",
    "Explain how BERT differs from GPT in natural language understanding tasks.",
    "What challenges exist in applying NLP to low-resource languages?",
    "How can reinforcement learning improve natural language generation models?",
    "Survey recent approaches to question answering in natural language processing."
]

print("=== Testing step_draft_outline ===\n")

for i, query in enumerate(test_queries, 1):
    print(f"\n{'='*80}")
    print(f"Test {i}/{len(test_queries)}")
    print(f"Query: {query}\n")
    
    try:
        # Create minimal cost args
        cost_args = CostReportingArgs(
            task_id=f"test_{i}",
            user_id="test_user",
            description="Test outline generation",
            model=GEMINI_25_FLASH,
            msg_id=f"test_{i}"
        )
        
        # Call step_draft_outline
        outline = scholar_qa.step_draft_outline(
            rewritten_query=query,
            cost_args=cost_args,
            max_sections=5,
            min_sections=2
        )
        
        print(f"Generated outline ({len(outline)} sections):")
        for j, section in enumerate(outline, 1):
            print(f"  {j}. {section}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
