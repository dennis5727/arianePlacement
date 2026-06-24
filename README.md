# LLM-Guided Chip Placement — a cost/quality study vs RL

**Research question.** For a *legal* (zero-overlap, in-canvas) placement of all 932 macros of
the `ariane` RISC-V design, **how close to MaskPlace (RL) can a training-free, LLM-guided,
seconds-not-hours method get — and at what cost?**

This reframes the project away from "beat RL". A fair, legal comparison shows MaskPlace's RL
is meaningfully better on wirelength while we are far cheaper and faster; the contribution is
the **trade-off**, quantified honestly, not a win.

## Why "we beat RL" did not survive a fair comparison

The earlier story ("training-free beats RL, free, in minutes") came from two artifacts:

1. **The placement was illegal.** The two-stage soft fill (`two_stage.soft_fill`) places the
   799 std-cell clusters at their wirelength-optimal positions with **no overlap checking** —
   ~150k overlapping pairs at ~94% grid density. HPWL has no idea macros overlap, so piling
   connected clusters on the same spot buys artificially low wirelength. MaskPlace's number is
   for a **legal** layout (0.00% overlap, paper Table 3).
2. **It was measured against its own baseline, not MaskPlace.**

Once legalized (`two_stage.two_stage_legal_hpwl`, grid 448, **0 overlaps, all 932 placed**):

| Method | legal HPWL (bbox) | wirelength (MST) | legal? | grid | cost |
|---|---|---|---|---|---|
| Training-free (this repo) | ~4.7e6 | (see harness) | yes | 448 | seconds, no GPU |
| **MaskPlace RL (paper)** | — | **1.46e6** (Table 2) | yes (0.00%) | 224 | ~hours, 1 GPU |

MST ≥ bbox always, so MaskPlace beats the training-free legal placement by **>3×** any way the
metric is aligned — and it does so at the native 224 grid.

## Key facts (verified against the paper, arXiv 2211.13382)

- ariane = **932 macros** (134 hard + 798 soft std-cell clusters), 0 separate std cells,
  **Area Util 78.39%** (Table 14) — exactly the continuous density measured here. The 93.9% at
  grid 224 is `ceil()` rounding inflation.
- MaskPlace places **all 932 by RL** at **N=224** with **0.00% overlap** (App. A.6; Tables 3, 10).
- Our integrated greedy fits only **814/932 at grid 224** but **932/932 at 448** — so the 448
  grid is a *packer* workaround (greedy myopia + rounding), not a real shortage of room.

## Repo map

- `maskplace/two_stage.py` — two-stage placement (hard wiremask greedy + soft median fill),
  **`legalize_placement` / `two_stage_legal_hpwl` / `legal_scores`** (zero-overlap legalizer),
  and the order searches (`random_order_control`, `llm_full_order_search`, `llm_prefix_search`).
- `maskplace/trade_off_eval.py` — the honest harness: re-scores each method on the **legal**
  metric, tracks wall-clock and $ token cost, and prints a table with the MaskPlace reference
  row. Also `proxy_vs_legal()` — does minimizing the overlapping proxy HPWL even track the
  legal HPWL? (If not, the searches optimize the wrong objective.)
- `maskplace/strong_search.py` — strong (topology) baseline order, greedy driver, LLM ordering
  advisor, no-LLM random-restart control.
- `maskplace/{greedy_place,place_db,comp_res,parse_netlist,region_constraint}.py` — the
  training-free wiremask engine and PlaceDB, reused unchanged.

## Run the trade-off study

```python
from trade_off_eval import evaluate_trade_off, proxy_vs_legal, print_trade_off, save_csv
from place_db import PlaceDB
from strong_search import hard_macro_names, topology_order

placedb = PlaceDB("ariane")
strong = topology_order(placedb, hard_macro_names(placedb))

# Does the fast overlapping proxy track the legal objective? (no API key needed)
proxy_vs_legal(placedb, strong, n=20)

# FREE rows need no API key; LLM rows need ANTHROPIC_API_KEY.
rows = evaluate_trade_off(model="claude-sonnet-4-6", run_llm=True)
print_trade_off(rows); save_csv(rows, "trade_off.csv")
```
