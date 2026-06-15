"""Phase 6: run the baselines, emit the comparison table + figures.

Runs (all on the 133 hard SRAM macros, deterministic order):
  1. pure greedy (wiremask)            -- baseline #1, 0 LLM calls
  2. LLM-guided greedy, text-only      -- baseline #2
  3. LLM-guided greedy + image (ours)  -- main method   (run_image=True)

Produces:
  * a printed + CSV comparison table (HPWL, % vs greedy, LLM calls, tokens)
  * eval_curve.png  : HPWL-vs-iteration for text vs image, greedy as a baseline
  * eval_{greedy,text,image}.png : best layouts

The text-vs-image gap is the project's headline ablation.
"""

import argparse
import csv
import os


def evaluate(benchmark="ariane", grid=224, hard_order="area", model="claude-sonnet-4-6",
             max_iters=15, patience=3, outdir=".", run_image=True, verbose=True):
    from llm_guided_placement import run_llm_loop

    os.makedirs(outdir, exist_ok=True)
    p = lambda name: os.path.join(outdir, name)

    runs = {}
    # text-only loop also gives us the pure-greedy baseline as its iter 0.
    runs["text"] = run_llm_loop(benchmark=benchmark, grid=grid, hard_order=hard_order,
                                advisor="claude", use_image=False, model=model,
                                max_iters=max_iters, patience=patience,
                                save_best_fig=p("eval_text.png"), verbose=verbose)
    if run_image:
        runs["image"] = run_llm_loop(benchmark=benchmark, grid=grid, hard_order=hard_order,
                                     advisor="claude", use_image=True, model=model,
                                     max_iters=max_iters, patience=patience,
                                     save_best_fig=p("eval_image.png"), verbose=verbose)

    greedy_hpwl = runs["text"]["baseline_hpwl"]
    rows = [{"method": "greedy (wiremask)", "hpwl": greedy_hpwl, "vs_greedy_%": 0.0,
             "llm_calls": 0, "out_tokens": 0}]
    rows.append({"method": "LLM text-only", "hpwl": runs["text"]["best_hpwl"],
                 "vs_greedy_%": runs["text"]["improvement_pct"],
                 "llm_calls": runs["text"]["llm_calls"],
                 "out_tokens": runs["text"]["out_tokens"]})
    if run_image:
        rows.append({"method": "LLM + image (ours)", "hpwl": runs["image"]["best_hpwl"],
                     "vs_greedy_%": runs["image"]["improvement_pct"],
                     "llm_calls": runs["image"]["llm_calls"],
                     "out_tokens": runs["image"]["out_tokens"]})

    # ---- table ----
    print("\n================ COMPARISON (ariane, 133 hard macros) ================")
    print(f"{'method':<22}{'HPWL':>14}{'vs greedy':>12}{'LLM calls':>11}{'out tok':>10}")
    for r in rows:
        print(f"{r['method']:<22}{r['hpwl']:>14.4e}{r['vs_greedy_%']:>+11.2f}%"
              f"{r['llm_calls']:>11}{r['out_tokens']:>10}")
    with open(p("eval_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "hpwl", "vs_greedy_%", "llm_calls", "out_tokens"])
        w.writeheader()
        w.writerows(rows)
    print("saved", p("eval_table.csv"))

    # ---- HPWL-vs-iteration plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axhline(greedy_hpwl, color="k", ls="--", lw=1, label="pure greedy")
        ax.plot(runs["text"]["hpwl_curve"], marker="o", label="LLM text-only")
        if run_image:
            ax.plot(runs["image"]["hpwl_curve"], marker="s", label="LLM + image")
        ax.set_xlabel("iteration")
        ax.set_ylabel("HPWL")
        ax.set_title("LLM-guided placement on ariane (133 hard macros)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(p("eval_curve.png"), dpi=120)
        plt.close(fig)
        print("saved", p("eval_curve.png"))
    except Exception as e:
        print("plot skipped:", e)

    return {"rows": rows, "runs": runs}


def _parse_args():
    a = argparse.ArgumentParser(description="Phase 6 evaluation: greedy vs LLM text vs image.")
    a.add_argument("--benchmark", default="ariane")
    a.add_argument("--grid", type=int, default=224)
    a.add_argument("--hard-order", default="area")
    a.add_argument("--model", default="claude-sonnet-4-6")
    a.add_argument("--max-iters", type=int, default=15)
    a.add_argument("--patience", type=int, default=3)
    a.add_argument("--outdir", default=".")
    a.add_argument("--no-image", action="store_true", help="skip the image method")
    return a.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate(benchmark=args.benchmark, grid=args.grid, hard_order=args.hard_order,
             model=args.model, max_iters=args.max_iters, patience=args.patience,
             outdir=args.outdir, run_image=not args.no_image)
