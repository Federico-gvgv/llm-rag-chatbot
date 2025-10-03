import streamlit as st
from raglab.config import Settings
from raglab.embed import Embedder
from raglab.index import FaissIndex
from raglab.retrieve import Retriever
from raglab.pipeline import ChatRAG
from raglab.llm import LLM

cfg = Settings.load("config/config.yaml")

st.set_page_config(page_title=cfg.ui["title"], layout="wide")
st.title(cfg.ui["title"])

# Quick guardrails
banned = set(map(str.lower, cfg.ui.get("banned_queries", [])))

@st.cache_resource # Cache the retriever and LLM to avoid reloading
def load_stack():
    embedder = Embedder(cfg.embeddings["model_name"], cfg.embeddings["batch_size"])
    ix = FaissIndex(cfg.index["faiss_path"], cfg.index["meta_path"])
    ix.load()
    retriever = Retriever(ix, embedder, cfg.retriever["top_k"], cfg.retriever["use_mmr"], cfg.retriever["mmr_lambda"])
    llm = LLM(**cfg.llm)
    rag = ChatRAG(retriever, llm, cfg.retriever["min_score"])
    return rag

rag = load_stack() # Load the RAG stack

if "history" not in st.session_state:
    st.session_state.history = [] # Initialize chat history

q = st.chat_input("Ask a question...") # Chat input box

if q:
    if any(word in q.lower() for word in banned):
        st.warning("Your query contains banned words. Please modify and try again.")
    else:
        st.session_state.history.append(("user", q)) # Add user message to history
        out = rag.answer(q) # Get answer from RAG
        st.session_state.history.append(("assistant", out["answer"], out["sources"])) # Add assistant message to history

for turn in st.session_state.history: # Display chat history
    role = turn[0] # role: user or assistant
    if role == "user":
        with st.chat_message("user"):
            st.write(turn[1]) # user message
    else:
        with st.chat_message("assistant"):
            st.markdown(turn[1]) # assistant message
            sources = turn[2] # sources
            if sources: # if there are sources
                st.caption("Sources:") # caption
                for i, u in enumerate(sources, 1):
                    st.write(f"[{i}] {u}") # list sources