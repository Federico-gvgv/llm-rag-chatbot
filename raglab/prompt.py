# Builds prompts for LLMs using provided context documents.

# Uses a system prompt and a template to format the final prompt.

SYSTEM = (
    "You are a helpful assistant specialized in the configured domain. "
    "Use ONLY the provided context to answer. If the answer is not in the context, say you don't know. "
    "Cite sources with [n] markers that map to the provided URLs"
)

TEMPLATE = (
    "<system>\n{system}\n</system>\n"
    "<context>\n{context}\n</context>\n"
    "<question>\n{question}\n</question>\n"
    "Reply with: answer first, then 'Sources:' followed by [n]->URL pairs."
)

def render_prompt(question: str, docs: list[dict]) -> tuple[str, list[str]]: # Render prompt with context documents
    ctx_lines, urls = [], [] # context lines and URLs
    for i, d in enumerate(docs, 1):
        ctx_lines.append(f"[{i}] {d['text']}") # add source marker
        urls.append(d["url"]) # collect URLs
    context = "\n\n".join(ctx_lines) # join context lines
    return TEMPLATE.format(system=SYSTEM, context=context, question=question), urls # return prompt and URLs