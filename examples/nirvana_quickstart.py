"""Nirvana + sembaker quickstart.

Requires: pip install sembaker[nirvana]   and OPENAI_API_KEY in the environment.

The pipeline below is 100% native Nirvana; the only sembaker addition is the
cxnv.optimize(ndf) call before .execute(). sembaker fills the OPTIONAL UDF slot
Nirvana already exposes on every semantic operator — Nirvana runs the compiled
function per row at zero token cost, and automatically falls back to its
per-row LLM path if the function raises.
"""

import os

# Optional but recommended: score each compile draw against LLM-labeled sample
# rows and keep the best (tames temperature-1 variance). Must be set before
# the sembaker import.
os.environ.setdefault("CX_VALIDATE", "1")

import pandas as pd
import nirvana as nv

import sembaker.backends.nirvana as cxnv

nv.configure_llm_backbone(model_name="gpt-5-mini", api_key=os.environ["OPENAI_API_KEY"])

# 20 rows: enough for the cost model to pick the compiled path (the one-shot
# compile only amortizes above ~11 records; at N=4 it would stay native).
df = pd.DataFrame({
    "reviewText": [
        "Absolutely loved it — a masterpiece.",
        "Terrible pacing, I walked out halfway.",
        "One of the best films of the year.",
        "Dull, predictable, and far too long.",
    ] * 5,
})

ndf = nv.DataFrame(df)
ndf.semantic_filter("the review is clearly positive", input_columns=["reviewText"])

# Walk the lineage plan, decide per semantic node, inject the compiled
# function into node.operator.tool.
report = cxnv.optimize(ndf)
print(report)

out, cost, secs = ndf.execute()
print(out)
print(f"tokens cost={cost}  wall={secs:.1f}s")
