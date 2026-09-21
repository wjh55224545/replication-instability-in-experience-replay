# Replication package — *Replication Instability in Experience Replay for Multi-Agent LLM Pipelines*

Code, raw results, and figures behind the paper. The intent is that a reader can re-derive
**every number reported in the manuscript** from the shipped result files without any API access,
and can re-run the experiments end to end if they have credentials for the two model endpoints.

The manuscript itself is not included here; it is under review.

---

## What the study found, in one paragraph

The paper studies a five-agent LLM pipeline whose report-generation step retrieves highly rated
past reports and injects them into the prompt. Across fifteen controlled comparisons on two
language models, **no replay effect reproduces**: single runs produce large, significant effects in
opposite directions on the two models, and ten runs of one identical configuration disagree about
the effect's magnitude and, on one model, about its sign. The mechanism claim is that what
relevance-matched retrieval can return is bounded by the **resolution of its retrieval key** — the
number of distinct values that key takes in the store — rather than by the number of entries.

---

## Quick start

```bash
git clone https://github.com/wjh55224545/replication-instability-in-experience-replay.git
cd replication-instability-in-experience-replay
pip install -r requirements.txt
```

**Checking the numbers needs no API access** — the analysis scripts run entirely off the shipped
JSON:

```bash
python scripts/paper/verify_degeneration.py     # the key-resolution measurement (Sec. 5.3)
python scripts/paper/pool_compare.py            # the 5-vs-5 Mann–Whitney comparison (Sec. 5.3)
python scripts/paper/heterogeneity.py           # Cochran's Q, I², τ² (Sec. 5.4)
python scripts/paper/power_numbers.py           # detectable-effect figures (Sec. 5.7)
python scripts/paper/recompute_two_tailed.py    # every p-value, one- vs two-tailed, plus the FDR check
python scripts/paper/analyze_expert_eval.py     # the five expert raters: level, agreement, and vs. the judge
python scripts/paper/recheck_mdes.py            # recomputes them, and checks the formula
python scripts/paper/check_profile_distribution.py   # profile construction vs. the record store (Sec. 4.1)
python scripts/paper/traj_task.py               # the §5.10 task, and a check that its trap works
python scripts/paper/traj_analyze.py             # the §5.10 run-to-run stability analysis
python scripts/paper/probe_psychdb.py           # every aggregation variant of the source database
```

**Regenerating the figures:**

```bash
python scripts/paper/viz_instability.py    # Figure 3 — the fifteen comparisons
python scripts/paper/viz_replicates.py     # Figure 4 — ten runs of one configuration
python scripts/paper/viz_jaamas.py         # Figures 1 and 2 — the pipeline diagram
```

Outputs land in `figures/`.

**Re-running experiments** additionally needs credentials:

```bash
cp .env.example .env      # then fill in your API keys
python scripts/paper/run_ab_repeated.py
```

---

## Layout

```
├── scripts/paper/         47 analysis, experiment, and figure scripts
├── data/                  raw output of every experiment
│   ├── human_eval/        the 12 reports used for the evaluator analysis, plus judge scores
│   └── expert_eval/       five expert score sheets (12 reports x 8 dimensions) and the LLM scores they are compared against
├── figures/               the paper's figures (PDF + PNG)
├── logs/llm_calls.jsonl   every LLM call: model, timestamp, tokens, latency
└── backend/               the three platform modules the scripts import
    ├── config.py          (trimmed) model-endpoint settings
    ├── aigc/llm_client.py the LLM client, with its concurrency guard
    └── evolution/memory.py the experience-replay mechanism itself
```

---

## Reproducing each result

| Manuscript | Script | Output |
|---|---|---|
| §5.1 fifteen comparisons | `run_ab_experiment.py`, `run_ab_semantic.py`, `run_ab_repeated.py`, `deepseek_repeated.py`, `bigger_pool_validation.py`, `diversity_validation.py` | `data/ab_*.json`, `data/deepseek_*.json`, `data/lingshu_bigpool_results.json` |
| §5.1 table | `analyze_replicates_v2.py` | prints |
| §5.3 pool size (5 vs 5) | `replicate_runs_parallel.py <model> <n> <pool_mult>`, then `pool_compare.py` | `data/*_pool15_run*.json` |
| §5.3 key resolution | **`verify_degeneration.py`** — no model calls | prints the 7-of-24 cell measurement |
| §5.4 ten identical runs | `replicate_runs_parallel.py <model> 5 3` | `data/*_replicate_run*.json` |
| §5.4 heterogeneity | `heterogeneity.py` | `data/heterogeneity_results.json` |
| §5.5 evaluator analysis | `cross_eval_recheck.py` (needs API), `cross_eval_reverse.py` | `data/cross_eval_recheck.json` |
| §5.5 expert raters | **`analyze_expert_eval.py`** — no model calls | `data/expert_eval_analysis.json` |
| §5.7 ablation | `run_ablation.py` | `data/ablation_results.json` |
| §5.7 power figures | `power_numbers.py`, `recheck_mdes.py` | prints |
| §5.4 multiplicity (FDR) | `recompute_two_tailed.py` | `data/two_tailed_recompute.json` |
| §5.8 safety checks | `safety_check.py` | prints |
| §5.9 cross-task validation | `setselect_task.py`, `setselect_experiment.py`, `setselect_analyze.py` | `data/setselect_*.json` |
| §5.10 trajectory task | **`traj_task.py`** (defines the task and the trap; no API), then `traj_experiment.py` (needs API), then `traj_analyze.py` | `data/traj_experiment_*.json` |
| Figures 1–4 | `viz_jaamas.py` (Fig 1 pipeline, Fig 2 replay loop), `viz_instability.py` (Fig 3), `viz_replicates.py` (Fig 4) | `figures/` |

The **triage** and **setselect pilot** scripts record designs that did not work: an argmax-style
extraction task whose accuracy sat at the majority-class rate. They are kept because the pilot
results are what motivated the final task design.

> ⚠️ **A caution worth carrying forward.** While preparing the bibliography we queried Crossref by
> *title* — a fuzzy match — and it returned a **wrong paper for 10 of 18 entries**: `instructgpt`
> matched a patent-law article; an ICLR paper matched a bioRxiv preprint. Every candidate DOI had
> to be re-checked by looking its title back up before being written. Do not trust title-based
> metadata lookup, in either direction.

---

## The data

`data/` holds the raw output of every experiment as emitted. The paper's numbers are computed from
these files rather than transcribed by hand.

Some files are large because they store per-item scores rather than aggregates. That is deliberate:
the paper's whole point is that aggregates conceal the instability.

`logs/llm_calls.jsonl` records every model call made during the study — model name, timestamp,
token counts, latency, and errors. It is the closest thing to a run log we have. See the
"traceability" paragraph in §4.2 of the manuscript for what it does and does not establish.

### A note on the constructed profiles

The student profiles are **constructed experimental units**, not a sample of a population. The
relevant facts, stated plainly:

- Aggregating the record store (`data/psych.db`) per student-day yields **19** profiles:
  16 medium-risk, 3 low-risk, **0 high-risk**.
- The pool used in the paper has 44: 14 low (3 real + 11 synthetic), 16 medium (all real), and
  14 high (**all synthetic**).
- The synthetic profiles are drawn **uniformly** over the range observed for records in that band,
  not sampled from the empirical distribution. A high-risk *record* exists in the store; a
  high-risk *student-day* does not.

`check_profile_distribution.py` reproduces these numbers, and `probe_psychdb.py` enumerates the
aggregation variants that rule out the alternatives.

The record store contains placeholder names only (张三 / 李四 / 王五) and no contact details.

---

## What is not included, and why

| Not included | Reason |
|---|---|
| The manuscript (`.tex`, `.docx`) | Under review |
| `.env` | Contains API keys. See `.env.example` |
| The platform application (`backend/agents`, `api`, `frontend`, `vibraimage_engine`, …) | Separate codebase, not needed to reproduce any result. Only the three modules the scripts import are vendored under `backend/` |
| `data/uploads/`, `data/demo/`, `data/yolov8n.pt` | Unrelated application assets (~87 MB) |
| Manuscript-preparation tooling (LaTeX→Word conversion, cross-reference and spelling checkers) | Concerned with producing the submission file, not with the results |
| Scripts superseded during development | 11 earlier document-generation scripts written against manuscript paths that no longer exist |
| — | The five expert score sheets **are** included, under `data/expert_eval/`. They contain scores only: no names, no contact details, and no information about the raters beyond an arbitrary number |

---

## Environment

Developed and run on Windows with Python 3.13. Requires `numpy`, `scipy`, `matplotlib`,
`python-docx`, `openai`, and `pydantic-settings`.

Two behaviours matter if you re-run experiments:

- **`LLM_MAX_CONCURRENCY`** (default 3) caps simultaneous model calls. Raising it to 8–12 shortens
  a replication sweep roughly fivefold; `probe_concurrency.py` measures the trade-off.
- **`LLM_ACQUIRE_TIMEOUT`** (default 30 s) is the semaphore wait. **Set it to `0` for batch runs.**
  A timeout makes the call return `None`, which the caller treats as a failed generation and
  **drops silently**. With a non-zero timeout, a saturated run discards samples without saying so.
  Our own runs use `0`.

---

## Citation

If you use this code or data, please cite the manuscript (details on acceptance). Until then:

> Wang, J., Wu, K., & Liu, S. *Replication Instability in Experience Replay for Multi-Agent LLM Pipelines.*
> Manuscript under review.

Correspondence: Shouqiang Liu (corresponding author), School of Artificial Intelligence,
South China Normal University.

## License

Code released under the MIT License (see `LICENSE`). Result files and figures are released under
CC BY 4.0.
