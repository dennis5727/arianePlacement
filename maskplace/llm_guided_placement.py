"""Phase 4/5: Main LLM-guided placement loop (text-only or +image).

Best-anchored hill-climb. Each iteration:
  diagnostics (long links + which macros fell back + last move outcome)
  -> advisor proposes regions for ONLY the macros worth moving, building on the
     CURRENT BEST regions
  -> greedy places (constrained subset, rest free)
  -> if HPWL improved, ACCEPT (new anchor); else REJECT and keep the best so far.

Iteration 0 is the unconstrained greedy baseline, which is also the floor.
Because we only ever accept improvements and unspecified macros stay free, the
result can never end up worse than pure greedy.

Run:
    from llm_guided_placement import run_llm_loop
    run_llm_loop(advisor="dummy")                 # free plumbing test (all 133)
    run_llm_loop(advisor="claude")                # text-only
    run_llm_loop(advisor="claude", use_image=True)  # +image (Phase 5)
"""

import argparse


def _long_links(env, names, conns, top=10):
    """Strongly-connected macro pairs that are far apart in the current layout."""
    pos = env.node_pos
    scored = []
    for i, j, c in conns:
        ni, nj = names[i], names[j]
        if ni not in pos or nj not in pos:
            continue
        xi, yi, sxi, syi = pos[ni]
        xj, yj, sxj, syj = pos[nj]
        d = abs((xi + sxi / 2) - (xj + sxj / 2)) + abs((yi + syi / 2) - (yj + syj / 2))
        scored.append((d, i, j, c))
    scored.sort(reverse=True)
    if not scored:
        return "(no macro-to-macro links)"
    return "; ".join(f"M{i}<->M{j} ~{d:.0f} apart ({c} nets)" for d, i, j, c in scored[:top])


def _build_feedback(env, names, conns, last_outcome):
    parts = ["LONG CONNECTIONS RIGHT NOW: " + _long_links(env, names, conns)]
    fb = getattr(env, "fallback_ids", [])
    if fb:
        parts.append("DID NOT FIT their region last time (placed elsewhere) -> "
                     "use a different region or move fewer macros into one region: "
                     + ", ".join(f"M{i}" for i in fb))
    if last_outcome == "accepted":
        parts.append("Your last change IMPROVED HPWL and was kept. Refine further.")
    elif last_outcome == "rejected":
        parts.append("Your last change did NOT improve HPWL and was reverted. "
                     "Try a different, smaller adjustment.")
    return "\n".join(parts)


def run_llm_loop(benchmark="ariane", pnm=None, grid=224, hard_only=True, hard_order="area",
                 max_iters=15, patience=3, advisor="dummy", use_image=False,
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
    conns = macro_connections(placedb, names, top_k=60)
    if verbose:
        print(f"placing {pnm} macros | advisor={advisor} | image={use_image} "
              f"| max_iters={max_iters} | patience={patience}")

    if advisor == "dummy":
        adv = DummyAdvisor(macro_ids, REGION_LABELS)
    elif advisor == "claude":
        adv = ClaudeAdvisor(summary, macro_ids, REGION_LABELS, model=model)
    else:
        raise ValueError("advisor must be 'dummy' or 'claude'")

    image_path = None
    if use_image:
        from visualize import render_placement

    history = HistoryTracker(patience=patience)

    def place(regions):
        env, n, fb = run_greedy(placedb, pnm=pnm, grid=grid, regions=regions, verbose=False)
        hpwl, _cost = comp_res(placedb, env.node_pos, env.ratio)
        return env, hpwl, n, fb

    # ----- iter 0: unconstrained baseline = the floor and first anchor -----
    cur_env, cur_hpwl, n, _fb = place(None)
    cur_regions = {}
    history.record(0, cur_hpwl, {})
    best_env = cur_env
    last_outcome = None
    if verbose:
        print(f"iter  0 (no regions): HPWL={cur_hpwl:.4e} placed={n} "
              f"overlaps={count_overlaps(cur_env)}")

    # ----- iters 1..max: hill-climb from the current best -----
    for it in range(1, max_iters + 1):
        feedback = _build_feedback(cur_env, names, conns, last_outcome)
        if use_image:
            image_path = render_placement(cur_env, grid, names=names, regions=cur_regions,
                                          path="llm_iter_view.png")
        proposed = adv.suggest(history.get_history_text(), cur_hpwl, history.best_hpwl,
                               prev_regions=cur_regions, feedback=feedback,
                               image_path=image_path)
        if proposed is None:
            proposed = cur_regions

        env, hpwl, n, fb = place(proposed)
        history.record(it, hpwl, proposed)
        accepted = hpwl < cur_hpwl
        if accepted:
            cur_env, cur_hpwl, cur_regions = env, hpwl, proposed
            best_env = env
            last_outcome = "accepted"
        else:
            last_outcome = "rejected"
        if verbose:
            print(f"iter {it:2d}: HPWL={hpwl:.4e} ({'ACCEPT' if accepted else 'reject'}) "
                  f"constrained={len(proposed)} fallback={fb} overlaps={count_overlaps(env)}")
        if history.should_stop():
            if verbose:
                print(f"early stop: no improvement for {patience} iterations")
            break

    if save_best_fig and best_env is not None:
        try:
            from visualize import render_placement
            render_placement(best_env, grid, names=names, regions=cur_regions,
                             path=save_best_fig)
        except Exception:
            best_env.save_fig(save_best_fig)

    base = history.records[0]["hpwl"]
    improved = 100.0 * (base - history.best_hpwl) / base
    if verbose:
        print("\n=== LLM LOOP RESULT ===")
        print(f"baseline (iter 0) : {base:.6e}")
        print(f"best HPWL         : {history.best_hpwl:.6e} at iter {history.best_iter}")
        print(f"improvement       : {improved:+.2f}% vs unconstrained greedy")
        print(f"iterations        : {len(history.records)} | LLM calls: {adv.calls}")
        print(f"tokens            : input {adv.in_tokens} | output {adv.out_tokens} "
              f"| cache_read {getattr(adv, 'cache_read_tokens', 0)}")
        if save_best_fig:
            print(f"best layout       : {save_best_fig}")

    return {
        "baseline_hpwl": base,
        "best_hpwl": history.best_hpwl,
        "best_iter": history.best_iter,
        "best_regions": cur_regions,
        "improvement_pct": improved,
        "iters": len(history.records),
        "llm_calls": adv.calls,
        "in_tokens": adv.in_tokens,
        "out_tokens": adv.out_tokens,
        "cache_read_tokens": getattr(adv, "cache_read_tokens", 0),
        "hpwl_curve": [r["hpwl"] for r in history.records],
        "history": history,
        "best_env": best_env,
        "placedb": placedb,
    }


def _parse_args():
    p = argparse.ArgumentParser(description="LLM-guided greedy placement loop.")
    p.add_argument("--benchmark", default="ariane")
    p.add_argument("--pnm", type=int, default=None)
    p.add_argument("--grid", type=int, default=224)
    p.add_argument("--max-iters", type=int, default=15)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--advisor", default="dummy", choices=["dummy", "claude"])
    p.add_argument("--image", action="store_true", help="send the placement image (Phase 5)")
    p.add_argument("--model", default="claude-sonnet-4-6")
    p.add_argument("--fig", default="llm_best.png")
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    run_llm_loop(benchmark=a.benchmark, pnm=a.pnm, grid=a.grid, max_iters=a.max_iters,
                 patience=a.patience, advisor=a.advisor, use_image=a.image,
                 model=a.model, save_best_fig=a.fig)
