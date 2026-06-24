"""Honest cost/quality trade-off harness: training-free LEGAL placement vs MaskPlace RL.

Reframes the project from "beat RL" to the real question: how close to MaskPlace (RL) can a
training-free, LLM-guided, seconds-not-hours method get on a LEGAL (zero-overlap, in-canvas)
full-932 ariane placement, and at what cost?

Why this harness exists
-----------------------
The order searches in two_stage.py optimize an OPTIMISTIC overlapping HPWL (two_stage_hpwl):
the soft clusters are stacked on top of each other, which buys artificially low wirelength.
That number is not comparable to anything. Here every training-free method is re-scored on
the SAME legal metric (two_stage.legal_scores -> zero overlaps, all macros in-canvas), and we
put MaskPlace's published numbers next to it.

MaskPlace reference (arXiv 2211.13382), ariane, grid N=224:
  * places ALL 932 macros by RL with 0.00% overlap          (Table 3; App. A.6: "all by RL")
  * wirelength (MST-estimated) = 14.63 x10^5 = 1.463e6        (Table 2)
  * Area Util 78.39% == our measured continuous density       (Table 14)
  * trained ~hours on 1 GPU (RTX 3090)                         (Fig 10 ~250 min on adaptec1)

Grid caveat (disclosed, not a metric bias): we report our legal numbers at grid 448 because
the greedy/analytical packer needs the extra resolution -- the integrated greedy fits only
814/932 at the native 224, while RL fits all 932. Wirelength is physical (cell*ratio on the
same 357 canvas), so 448-vs-224 HPWL is a valid comparison; the 224-packing ability is part
of what RL buys.

Usage:
    from trade_off_eval import evaluate_trade_off, proxy_vs_legal, print_trade_off, save_csv
    rows = evaluate_trade_off(model="claude-sonnet-4-6", run_llm=True)
    print_trade_off(rows); save_csv(rows, "trade_off.csv")
"""

import csv
import time


# Per-1M-token prices (USD), from the claude-api skill model table (cached 2026-06-04).
# cache_read ~= 0.1x input. Update here if pricing changes.
PRICING = {
    "claude-sonnet-4-6": {"in": 3.0, "out": 15.0, "cache_read": 0.30},
    "claude-opus-4-8":   {"in": 5.0, "out": 25.0, "cache_read": 0.50},
}

# Column order shared by the table, CSV, and the reference row.
COLS = ["method", "legal_bbox", "legal_mst", "overlaps", "n_unplaced", "grid",
        "wall_s", "gpu_hours", "calls", "in_tokens", "out_tokens", "usd", "note"]

# MaskPlace paper reference (arXiv 2211.13382), ariane, grid 224 -- a constant, clearly
# labelled. legal_bbox is left None: the paper reports MST wirelength (Table 2), not bbox.
MASKPLACE_ARIANE = {
    "method": "MaskPlace RL (paper)", "legal_bbox": None, "legal_mst": 1.463e6,
    "overlaps": 0, "n_unplaced": 0, "grid": 224, "wall_s": None,
    "gpu_hours": "~hours x1GPU", "calls": 0, "in_tokens": 0, "out_tokens": 0,
    "usd": None, "note": "paper T2/T3",
}


def usd_cost(model, in_tokens, out_tokens, cache_read_tokens=0):
    """Token cost in USD for a model run; None if the model price is unknown."""
    p = PRICING.get(model)
    if p is None:
        return None
    return (in_tokens * p["in"] + out_tokens * p["out"]
            + cache_read_tokens * p["cache_read"]) / 1e6


# --------------------------------------------------------------------------- #
# proxy-fidelity: does the fast overlapping HPWL track the LEGAL HPWL?
# --------------------------------------------------------------------------- #
def proxy_vs_legal(placedb, hard_names, grid=224, refine=2, soft_iters=8, n=20, seed=0,
                   verbose=True):
    """Correlate the OVERLAPPING proxy HPWL with the LEGAL HPWL across hard orders.

    Samples the strong order + n block-move perturbations, scores each on both the
    optimistic overlapping metric (two_stage_hpwl, what the searches actually minimize) and
    the legal metric (legal_scores), and returns Pearson + Spearman correlation. A weak
    correlation is a headline finding: the LLM/random search would be optimizing the wrong
    objective, so its "improvement" need not carry over to a real (legal) layout.
    """
    import random
    import numpy as np
    from two_stage import two_stage_hpwl, legal_scores
    from strong_search import perturb

    rng = random.Random(seed)
    n_h = len(hard_names)
    perms = [list(range(n_h))]
    while len(perms) < n + 1:
        perms.append(perturb(perms[0], rng))

    proxy, legal = [], []
    for k, perm in enumerate(perms):
        order = [hard_names[i] for i in perm]
        ph, _env = two_stage_hpwl(placedb, order, grid=grid, soft_iters=soft_iters)
        ls = legal_scores(placedb, order, grid=grid, soft_iters=soft_iters, refine=refine)
        proxy.append(ph)
        legal.append(ls["bbox"])
        if verbose:
            print(f"  sample {k:2d}: proxy(overlap)={ph:.4e}  legal={ls['bbox']:.4e}")

    proxy = np.asarray(proxy)
    legal = np.asarray(legal)
    pearson = float(np.corrcoef(proxy, legal)[0, 1])
    # Spearman = Pearson of ranks (avoids a scipy dependency)
    spearman = float(np.corrcoef(proxy.argsort().argsort(),
                                 legal.argsort().argsort())[0, 1])
    if verbose:
        print(f"proxy-vs-legal over {len(perms)} orders: "
              f"Pearson={pearson:+.3f}  Spearman={spearman:+.3f}")
        print("  -> WEAK: minimizing the overlapping proxy does NOT reliably lower legal HPWL."
              if spearman < 0.5 else
              "  -> the overlapping proxy tracks the legal objective.")
    return {"pearson": pearson, "spearman": spearman,
            "proxy": proxy.tolist(), "legal": legal.tolist()}


# --------------------------------------------------------------------------- #
# trade-off matrix
# --------------------------------------------------------------------------- #
def _legal_row(method, placedb, hard_names, perm, grid, refine, soft_iters, wall_s,
               model=None, calls=0, in_tok=0, out_tok=0, cache_tok=0):
    """Re-score a hard order (best_perm) on the LEGAL metric and build a result row."""
    from two_stage import legal_scores
    order = [hard_names[i] for i in perm]
    ls = legal_scores(placedb, order, grid=grid, soft_iters=soft_iters, refine=refine)
    return {
        "method": method, "legal_bbox": ls["bbox"], "legal_mst": ls["mst"],
        "overlaps": 0, "n_unplaced": ls["n_unplaced"], "grid": ls["grid_fine"],
        "wall_s": wall_s, "gpu_hours": 0, "calls": calls,
        "in_tokens": in_tok, "out_tokens": out_tok,
        "usd": usd_cost(model, in_tok, out_tok, cache_tok) if model else 0.0,
        "note": "",
    }


def evaluate_trade_off(benchmark="ariane", model="claude-sonnet-4-6", grid=224, refine=2,
                       max_iters=8, patience=3, soft_iters=8, n_random=12, prefix_k=50,
                       run_llm=True, verbose=True):
    """Run every training-free method, score each on the LEGAL metric, append the MaskPlace
    reference row. FREE rows (strong baseline, random control) need no API key; the LLM rows
    are skipped unless run_llm and an ANTHROPIC_API_KEY are available.
    """
    from place_db import PlaceDB
    from strong_search import hard_macro_names, topology_order
    import two_stage as ts

    placedb = PlaceDB(benchmark)
    hard = hard_macro_names(placedb)
    strong = topology_order(placedb, hard)
    n_h = len(strong)
    rows = []

    # strong baseline (free): identity order, scored legally
    t = time.time()
    rows.append(_legal_row("strong baseline", placedb, strong, list(range(n_h)),
                           grid, refine, soft_iters, time.time() - t))
    if verbose:
        print(f"strong baseline   legal bbox={rows[-1]['legal_bbox']:.4e} "
              f"MST={rows[-1]['legal_mst']:.4e}")

    # no-LLM random-order control (free)
    t = time.time()
    rc = ts.random_order_control(placedb, strong, grid=grid, n_evals=n_random,
                                 soft_iters=soft_iters, verbose=verbose)
    rows.append(_legal_row("random control", placedb, strong, rc["best_perm"],
                           grid, refine, soft_iters, time.time() - t))

    if run_llm:
        t = time.time()
        rf = ts.llm_full_order_search(placedb, strong, grid=grid, model=model,
                                      max_iters=max_iters, patience=patience,
                                      soft_iters=soft_iters, verbose=verbose)
        rows.append(_legal_row("LLM full-order", placedb, strong, rf["best_perm"],
                               grid, refine, soft_iters, time.time() - t, model=model,
                               calls=rf["calls"], in_tok=rf["in_tokens"],
                               out_tok=rf["out_tokens"], cache_tok=rf["cache_read_tokens"]))

        t = time.time()
        rp = ts.llm_prefix_search(placedb, strong, grid=grid, model=model,
                                  max_iters=max_iters, patience=patience, k=prefix_k,
                                  soft_iters=soft_iters, verbose=verbose)
        rows.append(_legal_row("LLM prefix", placedb, strong, rp["best_perm"],
                               grid, refine, soft_iters, time.time() - t, model=model,
                               calls=rp["calls"], in_tok=rp["in_tokens"],
                               out_tok=rp["out_tokens"], cache_tok=rp["cache_read_tokens"]))

    rows.append(dict(MASKPLACE_ARIANE))   # reference row, clearly labelled
    return rows


def print_trade_off(rows):
    """Pretty-print the trade-off table."""
    print("\n=== COST / QUALITY TRADE-OFF (legal full-932 ariane) ===")
    hdr = (f"{'method':<22}{'legal bbox':>12}{'legal MST':>12}{'grid':>6}"
           f"{'overlaps':>9}{'wall(s)':>9}{'$':>8}{'calls':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        bbox = f"{r['legal_bbox']:.3e}" if r.get("legal_bbox") is not None else "-"
        mst = f"{r['legal_mst']:.3e}" if r.get("legal_mst") is not None else "-"
        wall = f"{r['wall_s']:.1f}" if r.get("wall_s") is not None else "-"
        usd = f"{r['usd']:.3f}" if r.get("usd") is not None else "-"
        print(f"{r['method']:<22}{bbox:>12}{mst:>12}{str(r['grid']):>6}"
              f"{str(r['overlaps']):>9}{wall:>9}{usd:>8}{str(r.get('calls', 0)):>7}")
    print("\nMaskPlace row is from the paper (grid 224, ~hours on 1 GPU). Training-free rows"
          "\nare legal at grid 448; lower wirelength is better.")


def save_csv(rows, path="trade_off.csv"):
    """Write the trade-off rows to CSV."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in COLS})
    return path
