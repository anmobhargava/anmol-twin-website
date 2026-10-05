"""
HyDE (Hypothetical Document Embeddings).

The problem it solves: a short, casual recruiter question ("what's your AI
experience?") is written very differently from how the corpus itself is
written (structured, detailed, resume-style bullets). Embedding the raw
question and comparing it against embeddings of resume-style prose can miss
good matches purely due to this style/vocabulary gap — even when the content
is exactly what's needed.

HyDE's trick: instead of embedding the question directly, first ask an LLM
to write a HYPOTHETICAL answer to the question — even though we know it might
be wrong or generic — then embed THAT hypothetical answer instead. A fabricated
but stylistically-similar answer embeds much closer to the real matching
corpus content than the bare question would, because it's now written in
roughly the same "register" as what we're searching over.

Note this hypothetical document is used ONLY to drive retrieval — it is never
shown to the user or treated as fact. The final answer still comes from the
REAL retrieved chunks, not from the hypothetical one.
"""

import os
import sys

from anthropic import Anthropic

# chassis/ lives at backend/chassis/ (one level up from this file, in rag/)
# -- a vendored copy of the shared chassis's tracing.py, not an installed
# package. Adding backend/ to sys.path here makes `from chassis.tracing
# import ...` resolve whether this file is run standalone (`python
# hyde.py`, which only puts rag/ on sys.path automatically) or imported via
# pipeline.py (which only adds rag/ itself, not backend/).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from chassis.tracing import record_usage, traced_llm_call

HYDE_PROMPT = """You are helping retrieve information from a person's professional \
background corpus (resume, work history, skills). Given a question a recruiter \
might ask, write a brief, plausible-sounding hypothetical answer — 2-3 sentences, \
resume/bio style — as if it were a real excerpt from that person's background. \
It's fine if the specific facts you invent are wrong; this is used only to \
improve semantic search, not shown to anyone. Do not add disclaimers or caveats.

Question: {question}

Hypothetical answer excerpt:"""


class HyDEGenerator:
    def __init__(self, client: Anthropic | None = None, model: str = "claude-sonnet-4-6"):
        self.client = client or Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    @traced_llm_call("hyde")
    def generate(self, question: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=150,
            messages=[{"role": "user", "content": HYDE_PROMPT.format(question=question)}],
        )
        record_usage(response)
        return response.content[0].text.strip()


if __name__ == "__main__":
    hyde = HyDEGenerator()
    for q in ["What's your AI experience?", "Are you good with data?"]:
        hypothetical = hyde.generate(q)
        print(f"Question: {q!r}")
        print(f"Hypothetical doc (for retrieval only): {hypothetical!r}\n")