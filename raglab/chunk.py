# Utility functions for chunking documents into smaller pieces based on token counts

from typing import List, Dict
import re
import tiktoken

# Split paragraphs based on double newlines
def split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n{2,}", text)
    return [p.strip() for p in parts if p.strip()]

# Class for counting tokens in text
class TokenSizer:
    def __init__(self, model: str = "cl100k_base"): # Default model
        self.enc = tiktoken.get_encoding(model) # Initialize tokenizer for the specified model
    def count(self, s: str) -> int:
        return len(self.enc.encode(s)) # Return token count for the string
    
def chunk_docs(docs: List[Dict], target_tokens=800, overlap_tokens=120):
    sizer = TokenSizer()
    chunks = []
    for d in docs:
        paras = split_paragraphs(d['text'])
        buf, buf_tokens = [], 0
        for para in paras:
            t = sizer.count(para)
            if buf_tokens + t > target_tokens and buf: # If adding this paragraph exceeds target and buffer is not empty
                chunk_text = "\n\n".join(buf) # Join paragraphs with double newlines
                chunks.append({"url": d["url"], "title": d.get("title", ""), "text": chunk_text})
                # Start new buffer with overlap from the end
                if overlap_tokens > 0 and buf:
                    tail = buf[-1]
                    buf = [tail] if sizer.count(tail) <= overlap_tokens else [] # Keep last paragraph if it fits in overlap
                    buf_tokens = sum(sizer.count(p) for p in buf)
                else:
                    buf, buf_tokens = [], 0 # Reset buffer
            buf.append(para) # Add paragraph to buffer
            buf_tokens += t # Update token count
        if buf: # Add any remaining paragraphs as a final chunk
            chunk_text = "\n\n".join(buf)
            chunks.append({"url": d["url"], "title": d.get("title", ""), "text": chunk_text})
    return chunks