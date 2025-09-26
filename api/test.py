from scholarqa.rag.reranker.reranker_base import FlagEmbeddingScores
from scholarqa.rag.reranker.modal_engine import ModalReranker
from scholarqa.rag.retrieval import PaperFinderWithReranker, PaperFinder
from scholarqa.rag.retriever_base import FullTextRetriever
from scholarqa import ScholarQA
from scholarqa.llms.constants import GEMINI_25_FLASH

#Retrieval class/steps
retriever = FullTextRetriever(n_retrieval=20, n_keyword_srch=10) #full text and keyword search
reranker = FlagEmbeddingScores(model_name_or_path="mixedbread-ai/mxbai-rerank-large-v1") # BGE reranker


#Reranker if deployed on Modal, modal_app_name and modal_api_name are modal specific arguments.
#Please refer https://github.com/allenai/ai2-scholarqa-lib/blob/aps/readme_fixes/docs/MODAL.md for more info 
reranker = ModalReranker(app_name='<modal_app_name>', api_name='<modal_api_name>', batch_size=256, gen_options=dict())

#wraps around the retriever with `retrieve_passages()` and `retrieve_additional_papers()`, and reranker with rerank()
#any modifications to the retrieval output can be made here
paper_finder =  PaperFinder(retriever, n_rerank=-1, context_threshold=0.0)

#For wrapper class with MultiStepQAPipeline integrated
scholar_qa = ScholarQA(paper_finder=paper_finder, llm_model=GEMINI_25_FLASH) #llm_model can be any litellm model
print(scholar_qa.answer_query("what is rag? what is it user for? how it works?"))
