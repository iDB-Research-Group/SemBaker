<p align="center">
  <img src="https://raw.githubusercontent.com/iDB-Research-Group/SemBaker/main/assets/sembaker-logo.svg" width="440" alt="SemBaker — bake semantics into code">
</p>

# SemBaker: A Compilation-Based Optimizer for Semantic Operators

**An external optimizer for accelerating semantic operator systems, relying on *compilation-based execution* of semantic operators.**

<p align="center">
  <img src="https://raw.githubusercontent.com/iDB-Research-Group/SemBaker/main/assets/sembaker-flow.svg" alt="SemBaker inspects each semantic operator, compiles suitable operators once into cached Python functions, and leaves the rest on the engine's native path." width="100%">
</p>

Semantic operator systems let you write data operations in natural language —
`sem_filter("clearly positive review")`, `sem_map("extract the patient's
age")`, `sem_join("opposite sentiment")`. The usual way to execute these is
*interpretation*: for every row (or candidate pair, or document) the system
issues an LLM call to evaluate the predicate. Expressive, but it puts an
expensive LLM call inside the data loop — high latency, high cost, poor
scaling.

SemBaker accelerates the execution in another way: **compilation**. It calls the LLM *once* to
translate a semantic operator into a Python function, then runs
that function locally over the whole dataset — no per-item LLM call at
execution.

```
interpretation:   N items  ->  N LLM calls           (cost/latency grow with N)
compilation:      N items  ->  1 compile call + N local function calls (~0)
```

The compiled function is a reusable, cacheable artifact; the LLM cost is a
fixed one-shot "compile" charge that amortizes across all the input items.

**Performance** (measured on gpt-5-mini): one compile call costs ~$0.005 and
~25s, versus ~$0.0003 and ~2.5s for every native per-row call — so compilation
breaks even at roughly a dozen rows, and beyond that cost and latency stay
flat while the compiled function executes in microseconds per row (e.g. 10,000
rows: 1 LLM call instead of 10,000).

SemBaker implements the approach introduced in our vision paper
[*From Interpretation to Compilation: Compilation-Based Execution of Semantic
Operators*](https://arxiv.org/abs/2607.13407) (Dong & Wang, 2026) — see
[Citation](#citation).

## Install

```bash
pip install sembaker                 # core only (compile / cache / decide)
pip install sembaker[pz]             # + Palimpzest adapter deps
pip install sembaker[lotus]          # + LOTUS (lotus-ai)
pip install sembaker[nirvana]        # + Nirvana (nirvana-ai)
pip install sembaker[docetl]         # + DocETL
pip install sembaker[all]            # everything
```

Set `OPENAI_API_KEY` in your environment (or a `.env` you load yourself). The
default compile model is `gpt-5-mini`; override with `CX_COMPILE_MODEL`.

## Usage: keep your engine's native API

Users keep writing each engine's (Palimpzest, LOTUS, etc.) pipeline with their **native** API; 
SemBaker walks the pipeline, uses a cost model to decide per operator whether to compile or
stay native, and builds external scheduler and executor to execute compiled operators — without modifying the engine's
source code.

Note: the cost model only picks the compiled path when the input is large
enough to amortize the one-shot compile cost (roughly a dozen rows; see
"Compile-vs-native decision" below). On toy-sized data everything routes
native — the runnable scripts in [examples/](https://github.com/iDB-Research-Group/SemBaker/tree/main/examples) use 20 rows so the
compiled path actually engages.

**Palimpzest**

```python
import palimpzest as pz
import sembaker.backends.pz as cxpz

ds = (pz.MemoryDataset(id="reviews", vals=df)
        .sem_filter("the review is clearly positive"))
report = cxpz.optimize(ds)        # walk plan, decide per op, gate rules
out = ds.run(config)              # PZ executes; compiled ops run locally
```

**LOTUS**

```python
import lotus
from lotus.ast import LazyFrame
import sembaker.backends.lotus as cxlotus

lf = LazyFrame(df).sem_filter("The {reviewText} is clearly positive.")
cxlotus.optimize(lf)              # SemFilterNode -> CXFilterNode, in place
out = lf.execute(df)
```

**Nirvana**

```python
import nirvana as nv
import sembaker.backends.nirvana as cxnv

ndf = nv.DataFrame(df)
ndf.semantic_filter("the review is clearly positive", input_columns=["reviewText"])
cxnv.optimize(ndf)                # inject compiled fn into the native UDF slot
out, cost, secs = ndf.execute()   # automatic LLM fallback if the fn raises
```

**DocETL**

```python
import docetl
import sembaker.backends.docetl as cxdoc
cxdoc.apply()                     # wrap Frame.map / Frame.filter

f = docetl.from_list(docs)
f = f.filter(prompt="Keep clearly positive reviews. {{ input.reviewText }}")
out = f.collect()                 # rewritten to native code_filter, 0 LLM/doc
```

| Backend | How it's wired (no engine-source edits) | Module |
|---|---|---|
| Palimpzest | `Compiled*` physical operators + implementation rules | `sembaker.backends.pz`, `sembaker.backends.pz_ops` |
| LOTUS | LazyFrame node rewrite (`Sem*Node` → `CX*Node`) | `sembaker.backends.lotus` |
| Nirvana | native UDF-slot injection | `sembaker.backends.nirvana` |
| DocETL | `Frame.map`/`filter` → native `code_map`/`code_filter` | `sembaker.backends.docetl` |

**Your own engine**: the compile core is public API; an adapter is typically
~100 lines. See [docs/writing_a_backend.md](https://github.com/iDB-Research-Group/SemBaker/blob/main/docs/writing_a_backend.md).

## Operator support

**Every pipeline still runs in full.** SemBaker's decision is strictly
per-operator: an operator it can't (or shouldn't) compile is simply left on
the engine's native path — it is reported in the rewrite report, never
rewritten, never broken. What CAN be compiled currently is **filter / map /
join**:

| Backend | filter | map | join | everything else |
|---|---|---|---|---|
| Palimpzest | ✅ | ✅ | ✅ (semantic condition; `on=` equi-joins already run natively without LLM) | native (e.g. `sem_agg`) |
| LOTUS | ✅ | ✅ | ✅ inner joins only | native (`sem_agg`, `sem_topk`, `sem_extract`, ...) |
| Nirvana | ✅ | ✅ | ✅ | native (`rank`, `reduce`) |
| DocETL | ✅ | ✅ | — native (DocETL's `equijoin` has no code slot to inject into) | native (`reduce`, `resolve`, ...) |

Also left native by design: operators where the user already supplied their
own UDF (never touched), and predicates the codifiability judge rejects
(e.g. tone/sarcasm/nuance judgments that keyword-or-regex logic can't
faithfully capture).

## The compile pipeline & knobs

Every backend routes its compiles through one entry point
(`sembaker.optimizer.compile_op.compile_operator`), so these environment switches
apply uniformly across all backends:

| Env switch | Values | Meaning |
|---|---|---|
| `CX_COMPILE` | `e2e` (default) · `ir1` · `ir2` | compile method (see IR section) |
| `CX_CACHE` | `1` (default) · `0` | reuse/store compiled artifacts |
| `CX_REFINE` | `0` (default) · `1` | rewrite the predicate into a concrete, column-grounded one first |
| `CX_VALIDATE` | `0` (default) · `1` | pre-cache validation gate: score draws on LLM-labeled samples, keep the best |
| `CX_IR_FALLBACK` | `1` | if the IR can't express an operator, fall back to `e2e` |
| `CX_WARM` | `0` (default) · `1` | warm phase: pre-compile operators in parallel before execution |
| `CX_COMPILE_MODEL` | model id | LLM used for the one-shot compile (default `gpt-5-mini`) |
| `CX_JUDGE_MODEL` | model id | LLM used by the codifiability judge (defaults to `CX_COMPILE_MODEL`) |

- **refine** (`sembaker.core.refine`) — one LLM call that turns a fuzzy predicate
  into a concrete recipe (e.g. discovers a structured `scoreSentiment` column
  to compare instead of guessing sentiment from text).
- **validate** (`sembaker.core.validate`) — labels a few sample items once (LLM,
  in parallel) and scores each compile draw against them, keeping the best.
- **cache** (`sembaker.core.cache`) — persistent, keyed on `(method, op,
  canonicalized predicate, columns, model)`. The key **excludes the backend**:
  an artifact compiled for one engine is reused by another.

## IR-decoupled compilation (`CX_COMPILE=ir1|ir2`)

Free-form code generation is high-variance. The IR path decouples *finding the
logic* from *writing the code*: one LLM call emits a small **structured IR**,
then a **deterministic transpiler (no LLM)** renders it to Python — so
temperature-1 variance is confined to a small structured space, and the code
step is byte-reproducible.

- **`ir1`** — a flat "feature + decision" DSL (`sembaker.core.ir`).
- **`ir2`** — a nestable **expression tree (AST)** produced via grammar
  prompting; richer (nesting, comparisons, `to_int`, `first_nonempty`,
  `ifelse`) yet still one LLM call (`sembaker.core.ir2`). Design follows TRANX
  (Yin & Neubig, EMNLP 2018) with grammar prompting (Wang et al., 2023).

## Compile-vs-native decision

`sembaker.optimizer.decide()` routes each operator: a codifiability judge
(heuristic or LLM) asks whether the predicate is expressible as deterministic
logic over the visible columns, and a cost model amortizes the one-shot
compile cost against the per-item native cost at the operator's estimated
cardinality. Force the compiled path with each adapter's force flag / the
`FORCE_CX=1` convention where supported.

## Tested versions

Developed and tested against: `palimpzest` 1.5.x · `lotus-ai` 1.1.x ·
`nirvana-ai` 1.3.x · `docetl` 0.3.x · Python 3.10+.

## Citation

If you use SemBaker in your research, please cite:

```bibtex
@misc{dong2026compilation,
  title         = {From Interpretation to Compilation: Compilation-Based
                   Execution of Semantic Operators},
  author        = {Dong, Wenkai and Wang, Yifan},
  year          = {2026},
  eprint        = {2607.13407},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DB},
  url           = {https://arxiv.org/abs/2607.13407},
}
```

## License

SemBaker is **dual-licensed**:

- **AGPL-3.0** ([LICENSE](https://github.com/iDB-Research-Group/SemBaker/blob/main/LICENSE)) — free for research, evaluation, and any
  use that complies with the AGPL's copyleft, including its network-service
  provision (running a modified version as a service requires releasing your
  corresponding source under the AGPL).
- **Commercial license** — for closed-source or proprietary use that cannot
  meet AGPL obligations. Contact <dongw@hawaii.edu> or <yifanw@hawaii.edu>.

Patent applications covering techniques in this software have been filed;
see [NOTICE](https://github.com/iDB-Research-Group/SemBaker/blob/main/NOTICE).
