"""Palimpzest + sembaker quickstart.

Requires: pip install sembaker[pz]   and OPENAI_API_KEY in the environment.

The pipeline below is 100% native Palimpzest; the only sembaker addition is the
cxpz.optimize(ds) call before .run().
"""

import os

# Optional but recommended: score each compile draw against LLM-labeled sample
# rows and keep the best (tames temperature-1 variance). Must be set before
# the sembaker import.
os.environ.setdefault("CX_VALIDATE", "1")

import pandas as pd
import palimpzest as pz

import sembaker.backends.pz as cxpz

# 20 rows: enough for the cost model to pick the compiled path (the one-shot
# compile only amortizes above ~11 records; at N=4 it would stay native).
texts = [
    "Absolutely loved it — a masterpiece.",
    "Terrible pacing, I walked out halfway.",
    "One of the best films of the year.",
    "Dull, predictable, and far too long.",
] * 5
df = pd.DataFrame({
    "reviewId": [f"r{i}" for i in range(len(texts))],
    "reviewText": texts,
})

ds = (pz.MemoryDataset(id="reviews", vals=df)
        .sem_filter("the review is clearly positive"))

# Walk the plan, decide per semantic op (compile vs native), gate PZ's
# implementation rules so the chosen side wins.
report = cxpz.optimize(ds)
print(report)

config = pz.QueryProcessorConfig(
    policy=pz.MaxQuality(),
    available_models=[pz.Model.GPT_5_MINI],
    progress=True,
)
out = ds.run(config)
for record in out.data_records or []:
    print(record.to_dict(include_bytes=False))
