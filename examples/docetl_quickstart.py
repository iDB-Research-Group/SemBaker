"""DocETL + sembaker quickstart.

Requires: pip install sembaker[docetl]   and OPENAI_API_KEY in the environment.

The pipeline below is 100% native DocETL; the only sembaker addition is the
one-time cxdoc.apply() call, which wraps Frame.map / Frame.filter. Each
wrapped call decides compile-vs-native; on compile it rewrites the op into
DocETL's own code_map / code_filter (a local transform(doc), zero LLM calls
per document).
"""

import os

# Optional but recommended: score each compile draw against LLM-labeled sample
# docs and keep the best (tames temperature-1 variance). Must be set before
# the sembaker import.
os.environ.setdefault("CX_VALIDATE", "1")

import docetl

import sembaker.backends.docetl as cxdoc

cxdoc.apply()

# 20 docs: enough for the cost model to pick the compiled path (the one-shot
# compile only amortizes above ~11 records; at N=4 it would stay native).
docs = [
    {"reviewText": "Absolutely loved it — a masterpiece."},
    {"reviewText": "Terrible pacing, I walked out halfway."},
    {"reviewText": "One of the best films of the year."},
    {"reviewText": "Dull, predictable, and far too long."},
] * 5

f = docetl.from_list(docs)
f = f.filter(prompt="Keep clearly positive reviews. {{ input.reviewText }}")
out = f.collect()
print(out)
