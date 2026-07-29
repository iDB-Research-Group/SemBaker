"""LOTUS + sembaker quickstart.

Requires: pip install sembaker[lotus]   and OPENAI_API_KEY in the environment.

The pipeline below is 100% native LOTUS (LazyFrame plan API); the only sembaker
addition is the cxlotus.optimize(lf) call before execution. Rewritten nodes
never touch lotus.settings.lm — the LM config is only used by ops that stay
native.
"""

import os

# Optional but recommended: score each compile draw against LLM-labeled sample
# rows and keep the best (tames temperature-1 variance). Must be set before
# the sembaker import.
os.environ.setdefault("CX_VALIDATE", "1")

import pandas as pd
import lotus
from lotus.ast import LazyFrame
from lotus.models import LM

import sembaker.backends.lotus as cxlotus

lotus.settings.configure(lm=LM(model="gpt-5-mini"))

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

lf = LazyFrame(df).sem_filter("The {reviewText} is clearly positive.")

# Walk lf's node list, decide per semantic node, and replace
# SemFilterNode/SemMapNode with CX nodes that run the compiled function
# row-locally (compiled lazily on first execution, one LLM call total).
report = cxlotus.optimize(lf)
print(report)

out = lf.execute(df)
print(out)
