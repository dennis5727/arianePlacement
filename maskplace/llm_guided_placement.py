"""Phase 4: Main LLM-guided placement loop (text-only).

Each iteration:
  diagnostics (which strong links are long right now) -> advisor proposes regions
  for ONLY the macros worth moving -> greedy places (those constrained, the rest
  free) -> HPWL measured -> recorded -> early-stop check.

Iteration 0 is an unconstrained greedy baseline, which is also the floor: because
unspecified macros stay free, a placement can never be forced worse than this
baseline by the framework -- only the macros the LLM chooses to move are
constrained.

Run:
    from llm_guided_placement import run_llm_loop
    res = run_llm_loop(advisor="dummy")    # free plumbing test, all 133 macros
    res = run_llm_loop(advisor="claude")   # real LLM (needs ANTHROPIC_API_KEY)
"""

import argparse


def _long_links(env, names, conns, top=10):
    """Strongly-connected macro pairs that are far apart in the current layout.

    Returns a formatted text block listing the top ``top`` pairs by centroid
    (Manhattan) distance, so the LLM knows which macros to pull together.
    """
    pos = env.node_pos
    scored = []
    for i, j, c in conns:
        ni, nj = names[i], names[j]
        if ni not in pos or nj not in pos:
            continue
        xi, yi, sxi, syi = pos[ni]
        xj, yj, sxj, syj = pos[nj]
        ci = (xi + sxi / 2.0, yi + syi / 2.0)
        cj = (xj + sxj / 2.0, yj + syj / 2.0)
        d = abs(ci[0] - cj[0]) + abs(ci[1] - cj[1])
        scored.append((d, i, j, c))
    scored.sort(reverse=True)
    if not scored:
        return "(no macro-to-macro links)"
    return "; ".join(f"M{i}<->M{j} ~{d:.0f} cells apart ({c} nets)"
                     for d, i, j, c in scored[:top])


def run_llm_loop(benchmark="ariane", pnm=None, grid=224, hard_only=True, hard_order="keep",
                 max_iters=15, patience=3, advisor="dummy",
                 model="claude-sonnet-4-6", save_best_fig="llm_best.png", verbose=True):
    from place_db import PlaceDB
    from comp_res import comp_res
    from greedy_place import run_greedy, count_overlaps, select_hard_macros
    from region_constraint import REGION_LABELS
    from parse_netlist import build_summary, macro_connections
    from history_tracker import HistoryTracker
    from llm_interface import DummyAdvisor, ClaudeAdvisor

    placedb = PlaceDB(benchmark)
    if hard_only:
        placedb.node_id_to_name = select_hard_macros(placedb, order=hard_order)
    full = len(placedb.node_id_to_name)
    pnm = full if pnm is None else min(pnm, full)

    names = placedb.node_id_to_name[:pnm]
    macro_ids = list(range(pnm))
    summary = build_summary(placedb, pnm, REGION_LABELS)
    conns = macro_connections(placedb, names, top_k=60)   # for diagnostics
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

    # ----- iter 0: unconstrained baseline (also the floor) -----
    env, hpwl, n, fb = place(None)
    history.record(0, hpwl, {})
    best_env = last_env = env
    if verbose:
        print(f"iter  0 (no regions): HPWL={hpwl:.4e} placed={n} "
              f"overlaps={count_overlaps(env)}")

    # ----- iters 1..max: advisor-guided partial constraints -----
    regions = {}
    for it in range(1, max_iters + 1):
        diag = _long_links(last_env, names, conns)
        proposed = adv.suggest(history.get_history_text(),
                               history.records[-1]["hpwl"], history.best_hpwl,
                               prev_regions=regions, diagnostics=diag)
        if proposed is None:                  # advisor failed -> reuse previous regions
            proposed = regions
        regions = proposed

        env, hpwl, n, fb = place(regions)
        last_env = env
        ov = count_overlaps(env)
        history.record(it, hpwl, regions)
        if hpwl <= history.best_hpwl:
            best_env = env
        if verbose:
            print(f"iter {it:2d}: HPWL={hpwl:.4e} placed={n} overlaps={ov} "
                  f"constrained={len(regions)} fallback={fb}")
        if history.should_stop():
            if verbose:
                print(f"early stop: no improvement for {patience} iterations")
            break

    if save_best_fig and best_env is not None:
        best_env.save_fig(save_best_fig)

    base = history.records[0]["hpwl"]
    improved = 100.0 * (base - history.best_hpwl) / base
    if verbose:
        print("\n=== LLM LOOP RESULT ===")
        print(f"baseline (iter 0) : {base:.6e}")
        print(f"best HPWL         : {history.best_hpwl:.6e} at iter {history.best_iter}")
        print(f"improvement       : {improved:+.2f}% vs unconstrained greedy")
        print(f"iterations        : {len(history.records)}")
        print(f"LLM calls         : {adv.calls} | input {adv.in_tokens} "
              f"| output {adv.out_tokens} | cache_read {getattr(adv, 'cache_read_tokens', 0)}")
        if save_best_fig:
            print(f"best layout       : {save_best_fig}")

    return {
        "baseline_hpwl": base,
        "best_hpwl": history.best_hpwl,
        "best_iter": history.best_iter,
        "best_regions": history.best_regions,
        "improvement_pct": improved,
        "iters": len(history.records),
        "llm_calls": adv.calls,
        "in_tokens": adv.in_tokens,
        "out_tokens": adv.out_tokens,
        "cache_read_tokens": getattr(adv, "cache_read_tokens", 0),
        "history": history,
        "best_env": best_env,
        "placedb": placedb,
    }


def _parse_args():
    p = argparse.ArgumentParser(description="LLM-guided greedy placement loop (text-only).")
    p.add_argument("--benchmark", default="ariane")
    p.add_argument("--pnm", type=int, default=None, help="macros to place (default: all hard)")
    p.add_argument("--grid", type=int, default=224)
    p.add_argument("--max-iters", type=int, default=15)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--advisor", default="dummy", choices=["dummy", "claude"])
    p.add_argument("--model", default="claude-sonnet-4-6")
    p.add_argument("--fig", default="llm_best.png")
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    run_llm_loop(benchmark=a.benchmark, pnm=a.pnm, grid=a.grid, max_iters=a.max_iters,
                 patience=a.patience, advisor=a.advisor, model=a.model, save_best_fig=a.fig)
