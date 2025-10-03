# Ties together retriever and LLM for a chat-based RAG pipeline.

from .prompt import render_prompt

class ChatRAG:
    def __init__(self, retriever, llm, min_score=0.25):
        self.retriever = retriever
        self.llm = llm
        self.min_score = min_score
    
    def answer(self, question: str): # Answer a question using retrieval and LLM
        docs, scores = self.retriever.retrieve(question) # retrieve documents
        if not docs or max(scores) < self.min_score: # if no good docs
            return {"answer": "I don't know based on my resources.", "sources": []} # return default answer
        prompt, urls = render_prompt(question, docs) # render prompt
        out = self.llm.generate(prompt) # generate answer
        return {"answer": out, "sources": urls} # return answer and sources