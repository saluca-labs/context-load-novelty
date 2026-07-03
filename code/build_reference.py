#!/usr/bin/env python3
"""Build per-domain RELEVANT reference corpora (for the frontier dose-response).

The isolation test showed relevant material as neutral reference moves ideation
off-mainstream. To test the FRONTIER (does the effect grow/peak as load rises to
a large fraction of the window?), we need a lot of relevant, neutral reference
text to inject in increasing amounts. Generate it locally on gemma2 ($0), save to
references.json. The frontier runner truncates this to N tokens per load level.
"""

from __future__ import annotations
import json, sys, time
from pathlib import Path
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
TOPICS = {
    "science":  "the biology and science of why humans and animals sleep",
    "product":  "note-taking apps and personal knowledge-management software",
    "strategy": "the specialty-coffee industry and the business of coffee roasting",
}
ANGLES = ["core mechanisms and established findings", "history and background",
          "open questions and debates", "methods and how it is studied",
          "notable examples and case studies", "related adjacent concepts",
          "common misconceptions", "practical and applied aspects"]

def main():
    c = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1", timeout=180)
    refs = {}
    for dom, topic in TOPICS.items():
        chunks = []
        for i, angle in enumerate(ANGLES):
            p = (f"Write a detailed, factual, encyclopedic overview of {topic}, focusing on "
                 f"{angle}. About 400 words. Neutral reference tone, no lists, no preamble.")
            r = c.chat.completions.create(model="gemma2:9b", temperature=0.6, max_tokens=700,
                messages=[{"role": "user", "content": p}])
            chunks.append((r.choices[0].message.content or "").strip())
            print(f"  {dom} chunk {i+1}/{len(ANGLES)} ({len(chunks[-1])} chars)")
        refs[dom] = "\n\n".join(chunks)
        print(f"{dom}: {len(refs[dom])} chars (~{len(refs[dom])//4} tokens)")
    out = ROOT / "saluca" / "references.json"
    out.write_text(json.dumps(refs, ensure_ascii=False), encoding="utf-8")
    print(f"-> {out}")

if __name__ == "__main__":
    t=time.time(); main(); print(f"done in {time.time()-t:.0f}s")
