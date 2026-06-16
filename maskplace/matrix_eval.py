"""Full comparison matrix: 2 LLM methods x 2 baselines x {text, image}.

Methods
  * region   -- LLM assigns a few macros to 3x3 regions (constrains greedy cells).
                Implemented by llm_guided_placement.run_llm_loop.
  * ordering -- LLM proposes the macro placement ORDER.
                Implemented by strong_search.llm_order_search.

Baselines (same greedy placer, different macro order)
  * weak   -- area order (largest first)        ~1.25e5
  * strong -- topology/connectivity order        ~8.4e4   (the real target)

For each baseline we also report a no-LLM random-order-search control, so the
LLM is always measured against (a) its own baseline and (b) dumb search.

Each LLM run reports improvement vs ITS OWN baseline (iter-0 greedy in that
order), so cells are comparable within a baseline. Run:

    from matrix_eval import evaluate_matrix, print_matrix, save_csv
    rows = evaluate_matrix(model="claude-sonnet-4-6", max_iters=8)
    print_matrix(rows); save_csv(rows, "matrix.csv")
"""

import csv
import os


def _run_region(benchmark, base_names, grid, model, max_iters, patience, use_image, advisor, fig):
    from llm_guided_placement import run_llm_loop
    r = run_llm_loop(benchmark=benchmark, base_names=base_names, grid=grid, advisor=advisor,
                     use_image=use_image, model=model, max_iters=max_iters, patience=patience,
                     save_best_fig=fig, verbose=False)
    return dict(hpwl=r["best_hpwl"], improvement=r["improvement_pct"],
                calls=r["llm_calls"], out_tokens=r["out_tokens"], base_hpwl=r["baseline_hpwl"])


def _run_ordering(placedb, base_names, grid, model, max_iters, patience, use_image, fig):
    from strong_search import llm_order_search
    r = llm_order_search(placedb, base_names, grid=grid, model=model, max_iters=max_iters,
                         patience=patience, use_image=use_image, save_best_fig=fig, verbose=False)
    base = r["curve"][0]
    return dict(hpwl=r["best_hpwl"], improvement=100.0 * (base - r["best_hpwl"]) / base,
                calls=r["calls"], out_tokens=r["out_tokens"], base_hpwl=base)


def evaluate_matrix(benchmark="ariane", grid=224, model="claude-sonnet-4-6",
                    max_iters=8, patience=3, advisor="claude", n_random=12,
                    outdir=".", verbose=True):
    from place_db import PlaceDB
    from strong_search import (hard_macro_names, area_order, connectivity_order,
                               greedy_hpwl, random_restart_search)

    os.makedirs(outdir, exist_ok=True)
    placedb = PlaceDB(benchmark)
    hard = hard_macro_names(placedb)
    # both orders computed up front, before any greedy run mutates node_id_to_name
    baselines = {
        "weak (area)":     area_order(placedb, hard),
        "strong (topo)":   connectivity_order(placedb, hard),
    }

    rows = []
    for bname, order in baselines.items():
        if verbose:
            print(f"\n===== baseline: {bname} ({len(order)} macros) =====")
        _, base_hpwl, _ = greedy_hpwl(placedb, order, grid)
        rows.append(dict(baseline=bname, method="greedy baseline", mode="-",
                         hpwl=base_hpwl, improvement=0.0, calls=0, out_tokens=0))
        if verbose:
            print(f"  greedy baseline HPWL = {base_hpwl:.4e}")

        # no-LLM control: random ordering search
        rnd = random_restart_search(placedb, order, grid=grid, n_evals=n_random, verbose=False)
        rows.append(dict(baseline=bname, method="random order search", mode="-",
                         hpwl=rnd["best_hpwl"],
                         improvement=100.0 * (base_hpwl - rnd["best_hpwl"]) / base_hpwl,
                         calls=0, out_tokens=0))
        if verbose:
            print(f"  random search   HPWL = {rnd['best_hpwl']:.4e}")

        tag = os.path.join(outdir, bname.split()[0])  # 'weak' / 'strong'
        for mode, use_image in [("text", False), ("image", True)]:
            r = _run_region(benchmark, order, grid, model, max_iters, patience, use_image,
                            advisor, f"{tag}_region_{mode}.png")
            rows.append(dict(baseline=bname, method="LLM region", mode=mode, **r))
            if verbose:
                print(f"  LLM region {mode:<5} HPWL = {r['hpwl']:.4e} ({r['improvement']:+.2f}%)")
        for mode, use_image in [("text", False), ("image", True)]:
            r = _run_ordering(placedb, order, grid, model, max_iters, patience, use_image,
                              f"{tag}_order_{mode}.png")
            rows.append(dict(baseline=bname, method="LLM ordering", mode=mode, **r))
            if verbose:
                print(f"  LLM order  {mode:<5} HPWL = {r['hpwl']:.4e} ({r['improvement']:+.2f}%)")

    return rows


def print_matrix(rows):
    print("\n" + "=" * 74)
    print(f"{'baseline':<15}{'method':<22}{'mode':<7}{'HPWL':>13}{'vs base':>11}{'calls':>6}")
    print("-" * 74)
    last = None
    for r in rows:
        if r["baseline"] != last:
            print("-" * 74)
            last = r["baseline"]
        print(f"{r['baseline']:<15}{r['method']:<22}{r['mode']:<7}"
              f"{r['hpwl']:>13.4e}{r['improvement']:>+10.2f}%{r['calls']:>6}")
    print("=" * 74)


def save_csv(rows, path="matrix.csv"):
    cols = ["baseline", "method", "mode", "hpwl", "improvement", "calls", "out_tokens"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("saved", path)
