import json, csv, re
from pathlib import Path
import argparse

from raglab.config import Settings
from raglab.embed import Embedder
from raglab.index import FaissIndex
from raglab.retrieve import Retriever
from raglab.pipeline import ChatRAG
from raglab.llm import LLM


# ---------- Smarter matching utilities ----------

def normalize(s: str) -> str:
    """
    Normalize text for robust substring matching:
    - lowercasing
    - remove backticks
    - drop empty parens '()'
    - collapse whitespace
    """
    s = s.lower()
    s = re.sub(r"`+", "", s)        # remove backticks
    s = s.replace("()", "")         # drop empty parens
    s = re.sub(r"\s+", " ", s)      # collapse whitespace
    return s.strip()


def match_gold(answer: str, gold) -> bool:
    """
    Determine if 'answer' satisfies 'gold'.

    gold may be:
      - str: normalized substring match
          * if it starts with 're:', treat remainder as a regex (case-insensitive)
      - list[str]: any-of variants (True if any variant matches)
      - dict: {"all": [k1, k2, ...]} require all normalized substrings to appear
    """
    if isinstance(gold, list):
        return any(match_gold(answer, g) for g in gold)

    if isinstance(gold, dict) and "all" in gold:
        ans_norm = normalize(answer)
        return all(normalize(g) in ans_norm for g in gold["all"])

    if isinstance(gold, str) and gold.startswith("re:"):
        pattern = gold[3:]
        return re.search(pattern, answer, flags=re.IGNORECASE) is not None

    # default: simple normalized substring
    return normalize(str(gold)) in normalize(answer)


# ---------- Main script ----------

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True, help="Path to config YAML (e.g., config/config.yaml)")
args = ap.parse_args()

cfg = Settings.load(args.config)

# Load retriever + LLM stack
embedder = Embedder(cfg.embeddings["model_name"], cfg.embeddings["batch_size"])
ix = FaissIndex(cfg.index["faiss_path"], cfg.index["meta_path"])
ix.load()
retriever = Retriever(ix, embedder, cfg.retriever["top_k"], cfg.retriever["use_mmr"], cfg.retriever["mmr_lambda"])
llm = LLM(**cfg.llm)
rag = ChatRAG(retriever, llm, cfg.retriever["min_score"])

# Load eval set
qas_path = Path("eval/qas.jsonl")
qas = [json.loads(line) for line in qas_path.open("r", encoding="utf-8")]

rows = [("id", "strategy", "question", "expected", "answer", "correct")]

for i, qa in enumerate(qas, 1):
    q = qa["q"]
    gold = qa["a"]

    # Baseline: no retrieval context
    base_answer = llm.generate(q).strip()
    base_ok = match_gold(base_answer, gold)
    rows.append((i, "no_ctx", q, json.dumps(gold, ensure_ascii=False), base_answer, base_ok))

    # RAG: with retrieved context
    out = rag.answer(q)
    rag_answer = out["answer"].strip()
    rag_ok = match_gold(rag_answer, gold)
    rows.append((i, "rag", q, json.dumps(gold, ensure_ascii=False), rag_answer, rag_ok))

# Save results
Path("eval").mkdir(exist_ok=True)
with open("eval/results.csv", "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(rows)

print("Saved eval/results.csv")
