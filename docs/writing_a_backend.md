# Writing your own backend adapter

sembaker ships adapters for Palimpzest, LOTUS, Nirvana, and DocETL — but the
compile core is engine-agnostic and public API. If you have your own
semantic-operator engine (or any system with a "run this predicate per item"
loop), an adapter is typically ~100 lines. The four built-in adapters are the
reference implementations:

| Adapter | Injection strategy | Read it for |
|---|---|---|
| `sembaker/backends/lotus.py` | replace plan nodes with CX nodes | engines with a rewritable logical plan |
| `sembaker/backends/nirvana.py` | fill the engine's native UDF slot | engines that already accept per-op functions |
| `sembaker/backends/docetl.py` | wrap the user-facing API, reroute to the engine's code operators | engines with native code ops but no plan hooks |
| `sembaker/backends/pz.py` + `pz_ops/` | register custom physical operators + rules | engines with an extensible cost-based optimizer |

## The contract

An adapter does four things:

### 1. Find the semantic operators

Walk whatever your engine exposes — a node list, a lineage DAG, or just the
user's call site — and collect, per operator: its **kind** (`"filter"` /
`"map"` / `"join"`), its **natural-language intent**, the **columns/fields**
it sees, and (optionally) a few **sample rows**.

### 2. Decide compile-vs-native (optional)

```python
from sembaker.optimizer import decide

d = decide("filter", predicate, est_n=len(df), fields=["reviewText"])
if not d.use_cx:
    return  # leave the operator native
```

`decide()` combines a codifiability judge (is this predicate expressible as
deterministic logic over the visible columns?) with a cost model (does the
one-shot compile amortize at this cardinality?). You can skip this step and
always compile.

### 3. Compile

Use the unified entry point — it gives you the `CX_COMPILE` / `CX_CACHE` /
`CX_REFINE` / `CX_VALIDATE` knobs, the persistent cross-backend cache, and
single-flight deduplication for free:

```python
from sembaker.optimizer.compile_op import compile_operator

art = compile_operator(
    "filter",                      # "filter" | "map" | "join"
    "the review is clearly positive",
    model_id="gpt-5-mini",
    cols=["reviewText"],           # columns the function may look at
    samples=sample_rows,           # optional: list[dict], for refine/validate
)
# art: CompiledArtifact — .fn (the callable), .code, .cost_usd,
#      .input_tokens/.output_tokens, .duration_secs, .from_cache
```

The compiled function's calling convention (set by the prompt builders in
`sembaker.core.compiler`):

- **filter**: `fn(row: dict) -> bool`
- **map**: `fn(row: dict) -> value`
- **join**: `fn(left_row: dict, right_row: dict) -> bool`

Compile lazily (at first execution) or eagerly in a warm phase —
`sembaker.core.scheduler.run` / `run_map` give you parallel compile scheduling;
see `sembaker/backends/pz_ops/warm.py`.

### 4. Inject

Put `art.fn` wherever your engine executes the operator: swap the plan node,
fill the UDF slot, or wrap the execution callback. Strongly recommended:
**wrap the call in try/except and fall back to the engine's native LLM path
on error** — Nirvana does this natively; the other adapters do it in the
injected node.

## Notes

- The artifact cache key is `(method, op-kind, canonicalized predicate,
  columns, model)` — **no backend in the key**, so an artifact compiled
  through your adapter is reused by every other backend (and vice versa).
- Compile calls go directly to OpenAI (`OPENAI_API_KEY`), independent of
  whatever LM the user configured in the engine. Override the model with
  `CX_COMPILE_MODEL`.
- Keep the user's native API untouched. The whole point of the design is that
  the user writes their engine's own `sem_filter(...)`; your adapter is one
  `optimize(...)`-style call (or a one-time `apply()` patch) on top.
