"""Phase 4: Main LLM-guided placement loop (text-only).

Each iteration:  history -> advisor proposes {macro: region} -> greedy places
under those regions -> HPWL measured -> recorded -> early-stop check.

Iteration 0 is an unconstrained greedy baseline so the advisor has a starting
HPWL to improve on. Use advisor="dummy" to exercise the whole loop for free
before switching to advisor="claude".

Run:
    from llm_guided_placement import run_llm_loop
    res = run_llm_loop(pnm=30, advisor="dummy")        # free plumbing test
    res = run_llm_loop(pnm=30, advisor="claude")       # real LLM (needs API key)
"""

import argparse


def run_llm_loop(benchmark="ariane", pnm=30, grid=224, hard_only=True, hard_order="keep",
                 max_iters=20, patience=3, advisor="dummy",
                 model="claude-sonnet-4-6", save_best_fig="llm_best.png", verbose=True):
    from place_db import PlaceDB
    from comp_res import comp_res
    from greedy_place import run_greedy, count_overlaps, select_hard_macros
    from region_constraint import REGION_LABELS, check_region_compliance
    from parse_netlist import build_summary
    from history_tracker import HistoryTracker
    from llm_interface import DummyAdvisor, ClaudeAdvisor

    placedb = PlaceDB(benchmark)
    if hard_only:
        hard = select_hard_macros(placedb, order=hard_order)
        placedb.node_id_to_name = hard
        pnm = min(pnm, len(hard))
    else:
        pnm = min(pnm, len(placedb.node_id_to_name))

    macro_ids = list(range(pnm))
    summary = build_summary(placedb, pnm, REGION_LABELS)
    if verbose:
        print(f"placing {pnm} macros | advisor={advisor} | max_iters={max_iters} "
              f"| patience={patience}")

    if advisor == "dummy":
        adv = DummyAdvisor(macro_ids, REGION_LABELS)
    elif advisor == "claude":
        adv = ClaudeAdvisor(summary, macro_ids, REGION_LABELS, model=model)
    else:
        raise ValueError("advisor must be 'dummy' or 'claude'")

    history = HistoryTracker(patience=patience)

    def place(regions):
        env, n, fb = run_greedy(placedb, pnm=pnm, grid=grid, regions=regions, verbose=False)
        hpwl, _cost = comp_res(placedb, env.node_pos, env.ratio)
        return env, hpwl, n, fb

    # ----- iter 0: unconstrained baseline -----
    env, hpwl, n, fb = place(None)
    history.record(0, hpwl, {})
    best_env = env
    if verbose:
        print(f"iter  0 (no regions): HPWL={hpwl:.4e} placed={n} "
              f"overlaps={count_overlaps(env)}")

    # ----- iters 1..max: advisor-guided -----
    regions = None
    for it in range(1, max_iters + 1):
        proposed = adv.suggest(history.get_history_text(),
                               history.records[-1]["hpwl"], history.best_hpwl,
                               prev_regions=regions)
        if proposed is None:                  # advisor gave up -> reuse previous regions
            proposed = regions
        regions = proposed

        env, hpwl, n, fb = place(regions)
        ov = count_overlaps(env)
        nviol, _ = check_region_compliance(env, regions or {}, grid)
        history.record(it, hpwl, regions or {})
        if hpwl <= history.best_hpwl:
            best_env = env
        if verbose:
            print(f"iter {it:2d}: HPWL={hpwl:.4e} placed={n} overlaps={ov} "
                  f"fallback={fb} out-of-region={nviol}")
        if history.should_stop():
            if verbose:
                print(f"early stop: no improvement for {patience} iterations")
            break

    if save_best_fig and best_env is not None:
        best_env.save_fig(save_best_fig)

    if verbose:
        print("\n=== LLM LOOP RESULT ===")
        print(f"best HPWL   : {history.best_hpwl:.6e} at iter {history.best_iter}")
        print(f"iter 0 base : {history.records[0]['hpwl']:.6e} (unconstrained greedy)")
        print(f"iterations  : {len(history.records)}")
        print(f"LLM calls   : {adv.calls} | input_tokens {adv.in_tokens} "
              f"| output_tokens {adv.out_tokens}")
        if save_best_fig:
            print(f"best layout : {save_best_fig}")

    return {
        "best_hpwl": history.best_hpwl,
        "best_iter": history.best_iter,
        "best_regions": history.best_regions,
        "baseline_hpwl": history.records[0]["hpwl"],
        "iters": len(history.records),
        "llm_calls": adv.calls,
        "in_tokens": adv.in_tokens,
        "out_tokens": adv.out_tokens,
        "history": history,
        "best_env": best_env,
        "placedb": placedb,
    }


def _parse_args():
    p = argparse.ArgumentParser(description="LLM-guided greedy placement loop (text-only).")
    p.add_argument("--benchmark", default="ariane")
    p.add_argument("--pnm", type=int, default=30)
    p.add_argument("--grid", type=int, default=224)
    p.add_argument("--max-iters", type=int, default=20)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--advisor", default="dummy", choices=["dummy", "claude"])
    p.add_argument("--model", default="claude-sonnet-4-6")
    p.add_argument("--fig", default="llm_best.png")
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    run_llm_loop(benchmark=a.benchmark, pnm=a.pnm, grid=a.grid, max_iters=a.max_iters,
                 patience=a.patience, advisor=a.advisor, model=a.model, save_best_fig=a.fig)
