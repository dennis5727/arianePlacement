"""LLM-guided LEGAL placement of ALL 932 macros via the integrated position-mask greedy.

Motivation
----------
two_stage.py places the 134 hard macros with the wiremask greedy, then fills the 799 soft
clusters with an overlapping median sweep and legalizes afterward. That makes the LLM's lever
(the HARD order) weak for the full-932 objective: the hard greedy is blind to hard<->soft and
soft<->soft nets, the soft stage re-optimizes soft every time, and legalization scatters them.

Here ALL 932 macros are placed by the SAME wiremask greedy, one at a time, with the position
mask forbidding overlap -- so the layout is LEGAL BY CONSTRUCTION (zero overlap, in-canvas):
no soft_fill, no separate legalizer. At grid 448 there is enough slack to place all 932 (only
~814/932 fit at the native 224). Now the placement ORDER controls EVERY macro and every
placement sees all connections, so the order is a real lever on the full-932 HPWL -- the same
property that makes MaskPlace's RL work, but training-free.

To keep the LLM decision tractable, it reorders only the top-N most-connected HUB macros
(ranked from the netlist); the remaining macros keep the topology order. Compared three ways:
  * heuristic baseline -- topology order of all 932, no search                (free)
  * random hub control -- perturb the hub order, keep improvements            (free)
  * llm hub search      -- the model reorders the hubs, accept-if-improves     (paid)

Legality ('all 932 placed') is order-dependent at high density, so every score reports
n_placed and an order that strands macros is rejected (scored +inf).

This module is additive: it only READS the existing engine (greedy_place, comp_res, place_db,
parse_netlist, strong_search) -- it changes none of them.
"""

import json
import random
import re
import time

INF = float("inf")


# --------------------------------------------------------------------------- #
# ranking + placement
# --------------------------------------------------------------------------- #
def hub_names(placedb, top_n, by="degree"):
    """Top-N most-connected macro names (hubs), straight from the netlist.

    by="degree" ranks by number of nets touching the macro (len of its net set);
    ties broken by name for determinism.
    """
    names = list(placedb.node_info.keys())
    deg = {n: len(placedb.node_to_net_dict.get(n, ())) for n in names}
    names.sort(key=lambda n: (-deg[n], n))
    return names[:top_n]


def topology_all(placedb):
    """Frontier-growth topology order of ALL 932 macros (the heuristic baseline order)."""
    return list(placedb.node_id_to_name)


def integrated_greedy_hpwl(placedb, order_names, grid=448):
    """Place ALL macros in ``order_names`` with the position-mask greedy.

    Overlap-free by construction. Returns (hpwl, env, n_placed, n_total); n_placed < n_total
    means the order stranded the free space (still overlap-free, just incomplete).
    """
    from greedy_place import run_greedy
    from comp_res import comp_res

    saved = placedb.node_id_to_name
    try:
        placedb.node_id_to_name = list(order_names)
        env, n_placed, _fb = run_greedy(placedb, pnm=len(order_names), grid=grid, verbose=False)
        hpwl, _cost = comp_res(placedb, env.node_pos, env.ratio)
    finally:
        placedb.node_id_to_name = saved
    return hpwl, env, n_placed, len(order_names)


def _score(placedb, order_names, grid):
    """Score an order on the integrated legal layout. Incomplete placements -> +inf so the
    search rejects any order that cannot fit all 932 macros. Returns (hpwl, env, n_placed)."""
    hpwl, env, n_placed, n_total = integrated_greedy_hpwl(placedb, order_names, grid)
    return (hpwl if n_placed >= n_total else INF), env, n_placed


def _full_order(hub_perm, hubs, rest):
    """Full 932-order = hubs in the given (LLM/random) order, then the rest in topology order."""
    return [hubs[i] for i in hub_perm] + rest


def overlaps_and_oob(env, grid):
    """(overlapping macro pairs, out-of-canvas macros) for a placed env -- a legality check."""
    rects = list(env.node_pos.values())
    ov = 0
    for i in range(len(rects)):
        xi, yi, sxi, syi = rects[i]
        for j in range(i + 1, len(rects)):
            xj, yj, sxj, syj = rects[j]
            if xi < xj + sxj and xj < xi + sxi and yi < yj + syj and yj < yi + syi:
                ov += 1
    oob = sum(1 for (x, y, sx, sy) in env.node_pos.values()
              if x < 0 or y < 0 or x + sx > grid or y + sy > grid)
    return ov, oob


def mst_of(placedb, env):
    """Minimum-spanning-tree wirelength of a placed env -- comp_res's SECOND return value.

    The searches optimize/report bbox HPWL (comp_res's first value); this reports the MST
    estimate (its second value), which is the metric MaskPlace's paper Table 2 uses, so the
    two can be compared apples-to-apples. MST >= bbox HPWL for the same layout.
    """
    from comp_res import comp_res
    return comp_res(placedb, env.node_pos, env.ratio)[1]


# --------------------------------------------------------------------------- #
# searches over the hub order (all scored on the legal full-932 layout)
# --------------------------------------------------------------------------- #
def heuristic_baseline(placedb, grid=448, verbose=True):
    """Topology order of all 932 placed by the integrated greedy -- the no-search baseline."""
    order = topology_all(placedb)
    t = time.time()
    hpwl, env, n_placed, n_total = integrated_greedy_hpwl(placedb, order, grid)
    if verbose:
        print(f"heuristic baseline (topology): HPWL={hpwl:.4e} placed={n_placed}/{n_total} "
              f"({time.time() - t:.0f}s)")
    return {"hpwl": hpwl, "best_env": env, "n_placed": n_placed, "best_order": order,
            "wall_s": time.time() - t}


def random_hub_control(placedb, top_n=300, grid=448, n_evals=12, seed=0, verbose=True):
    """No-LLM control: hill-climb the HUB order by random block moves; integrated legal score.

    Starts from the hub set in degree order (rest = topology) and keeps any perturbation of
    the hub order that lowers the legal full-932 HPWL. Fair 'is the LLM beating dumb search?'.
    """
    from strong_search import perturb

    hubs = hub_names(placedb, top_n)
    hubset = set(hubs)
    rest = [n for n in topology_all(placedb) if n not in hubset]
    N = len(hubs)
    rng = random.Random(seed)

    # FLOOR = the plain topology order (already strong); the search only keeps a hub
    # arrangement that BEATS it, so the result never regresses below topology.
    topo = topology_all(placedb)
    best = list(range(N))               # hub-perm anchor (degree order) we perturb
    best_order = topo
    best_h, best_env, npl = _score(placedb, topo, grid)
    curve = [best_h]
    if verbose:
        print(f"random-hub start: topology floor HPWL={best_h:.4e} placed={npl} "
              f"| reordering top {N} hubs")
    for it in range(1, n_evals + 1):
        cand = perturb(best, rng)
        order = _full_order(cand, hubs, rest)
        h, env, npl = _score(placedb, order, grid)
        keep = h < best_h
        if keep:
            best, best_h, best_env, best_order = cand, h, env, order
        curve.append(best_h)
        if verbose:
            tag = "keep" if keep else ("drop" if h < INF else "drop(incomplete)")
            print(f"  rand iter {it:2d}: HPWL={h:.4e} ({tag}) placed={npl} best={best_h:.4e}")
    return {"hpwl": best_h, "best_env": best_env, "best_hub_perm": best,
            "best_order": best_order, "curve": curve}


def llm_hub_search(placedb, top_n=300, grid=448, model="claude-sonnet-4-6",
                   max_iters=8, patience=3, verbose=True):
    """LLM reorders the top-N hub macros; rest keep topology order. Scored on the integrated
    LEGAL full-932 layout, accept-if-improves, anchored on the best so far (never regresses
    below the topology baseline). Returns the legal placement env + token/cost counters."""
    from strong_search import OrderingAdvisor, repair_perm, long_links
    from parse_netlist import build_summary, macro_connections
    from region_constraint import REGION_LABELS

    hubs = hub_names(placedb, top_n)
    hubset = set(hubs)
    rest = [n for n in topology_all(placedb) if n not in hubset]
    N = len(hubs)
    id_of = {name: i for i, name in enumerate(hubs)}

    # static prompt context over the HUB subset (cached by the advisor)
    saved = placedb.node_id_to_name
    placedb.node_id_to_name = list(hubs)
    summary = build_summary(placedb, N, REGION_LABELS, top_k_conn=30)
    link_pairs = macro_connections(placedb, hubs, top_k=40)
    placedb.node_id_to_name = saved
    links = [(hubs[i], hubs[j], c) for i, j, c in link_pairs]

    adv = OrderingAdvisor(summary, N, model=model)

    # FLOOR = the plain topology order; the LLM only ever improves on it (never regresses).
    topo = topology_all(placedb)
    best = list(range(N))               # hub-perm anchor shown to the model (degree order)
    best_order = topo
    best_h, best_env, npl = _score(placedb, topo, grid)
    curve = [best_h]
    if verbose:
        print(f"iter 0 (topology floor): HPWL={best_h:.4e} placed={npl} | reordering top {N} hubs")

    attempts = [(best_h, True, "baseline")]
    tried = {tuple(best)}

    def history_text():
        lines = [f"attempt {k}: HPWL={h:.4e} "
                 f"({note or ('accepted, NEW BEST' if acc else 'rejected')})"
                 for k, (h, acc, note) in enumerate(attempts)]
        lines.append(f"Best so far: HPWL={best_h:.4e}")
        return "\n".join(lines)

    no_improve = 0
    for it in range(1, max_iters + 1):
        fb = "LONG CONNECTIONS RIGHT NOW: " + long_links(best_env, id_of, links)
        proposed = adv.suggest(best_h, best, history_text(), feedback=fb, image_path=None)
        if proposed is None:
            if verbose:
                print(f"iter {it:2d}: no proposal -> stop")
            break
        hub_perm = repair_perm(proposed, N, list(range(N)))
        key = tuple(hub_perm)
        if key in tried:
            no_improve += 1
            attempts.append((best_h, False, "duplicate hub order (skipped)"))
            if verbose:
                print(f"iter {it:2d}: already-tried hub order -> skipped (no_improve={no_improve})")
            if no_improve >= patience:
                break
            continue
        tried.add(key)
        order = _full_order(hub_perm, hubs, rest)
        h, env, npl = _score(placedb, order, grid)
        accepted = h < best_h
        if accepted:
            best, best_h, best_env, best_order, no_improve = hub_perm, h, env, order, 0
        else:
            no_improve += 1
        attempts.append((h, accepted, ""))
        curve.append(best_h)
        if verbose:
            tag = "ACCEPT" if accepted else ("reject" if h < INF else "reject(incomplete)")
            print(f"iter {it:2d}: HPWL={h:.4e} ({tag}) placed={npl} best={best_h:.4e}")
        if no_improve >= patience:
            if verbose:
                print(f"early stop: no improvement for {patience} iterations")
            break

    return {"hpwl": best_h, "best_env": best_env, "best_hub_perm": best,
            "best_order": best_order, "curve": curve,
            "calls": adv.calls, "in_tokens": adv.in_tokens, "out_tokens": adv.out_tokens,
            "cache_read_tokens": adv.cache_read_tokens}


# --------------------------------------------------------------------------- #
# (C) targeted edits to the topology order: "place macro A immediately before B"
# --------------------------------------------------------------------------- #
def apply_moves(order, moves, hubs):
    """Apply 'place hubs[a] immediately before hubs[b]' edits to a placement order.

    Each move (a, b) relocates one macro next to another WITHOUT otherwise disturbing
    the order -- so the strong topology structure is preserved and only a few targeted
    macros shift. Invalid/no-op moves (out of range, a==b, same macro) are skipped.
    """
    order = list(order)
    N = len(hubs)
    for a, b in moves:
        if not (0 <= a < N and 0 <= b < N) or a == b:
            continue
        na, nb = hubs[a], hubs[b]
        if na == nb or na not in order or nb not in order:
            continue
        order.remove(na)
        order.insert(order.index(nb), na)        # na goes immediately before nb
    return order


class PromoteAdvisor:
    """Anthropic-backed advisor for option (C): proposes a FEW targeted moves that pull
    strongly-connected macros together in an already-good topology order, instead of
    reshuffling everything. Output is a JSON list of [a, b] pairs = 'place macro a
    immediately before macro b'."""

    SYSTEM = (
        "You are an expert chip floorplanning assistant. A training-free greedy placer puts "
        "macros down ONE AT A TIME in a given order; placing two strongly-connected macros "
        "consecutively in the order tends to place them physically close. You are given an "
        "already-good topology order and a list of strongly-connected macro pairs that ended "
        "up FAR APART in the current layout. You improve it with a FEW targeted moves -- each "
        "move places one macro immediately before another in the order, pulling a connected "
        "pair together -- without reshuffling the rest. You always answer with a single JSON "
        "list of [a, b] integer pairs."
    )

    def __init__(self, summary, n, model="claude-sonnet-4-6", max_tokens=1500,
                 max_retries=2, api_key=None):
        import anthropic
        import os
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.summary = summary
        self.n = n
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.calls = self.in_tokens = self.out_tokens = self.cache_read_tokens = 0

    def _user_msg(self, best_hpwl, history_text, feedback):
        return (
            f"There are {self.n} hub macros with ids M0..M{self.n - 1} (see summary).\n\n"
            "=== MOVES TRIED SO FAR ===\n"
            f"{history_text}\n\n"
            "=== STRONGLY-CONNECTED MACROS THAT ARE FAR APART RIGHT NOW ===\n"
            f"{feedback}\n\n"
            "=== YOUR TASK ===\n"
            f"Current best LEGAL full-932 HPWL: {best_hpwl:.4e} (lower is better).\n"
            "Propose 1-5 targeted moves to pull far-apart connected macros together. Each move "
            "is a pair [a, b] meaning 'place macro a immediately before macro b in the "
            "placement order'. Prefer moving the less-connected macro of a far-apart pair next "
            "to the more-connected one. Do NOT repeat moves that did not help above. Think "
            "briefly, then output ONLY a JSON list of pairs, e.g. [[12,5],[30,5]]."
        )

    @staticmethod
    def _parse_pairs(text, n):
        """Extract valid [a, b] id pairs from the reply (robust to surrounding prose)."""
        out = []
        for a, b in re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", text):
            a, b = int(a), int(b)
            if 0 <= a < n and 0 <= b < n and a != b:
                out.append((a, b))
        return out

    def suggest(self, best_hpwl, history_text, feedback):
        msg = self._user_msg(best_hpwl, history_text, feedback)
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.messages.create(
                    model=self.model, max_tokens=self.max_tokens,
                    system=[{"type": "text", "text": self.SYSTEM},
                            {"type": "text", "text": self.summary,
                             "cache_control": {"type": "ephemeral"}}],
                    messages=[{"role": "user", "content": [{"type": "text", "text": msg}]}],
                )
                self.calls += 1
                u = resp.usage
                self.in_tokens += getattr(u, "input_tokens", 0)
                self.out_tokens += getattr(u, "output_tokens", 0)
                self.cache_read_tokens += getattr(u, "cache_read_input_tokens", 0) or 0
                text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
                moves = self._parse_pairs(text, self.n)
                if moves:
                    return moves
                print(f"  [llm] attempt {attempt + 1}: no usable [a,b] pairs; retrying")
            except Exception as e:
                print(f"  [llm] attempt {attempt + 1} error: {e}")
        return None


def random_promote_control(placedb, top_n=300, grid=448, n_evals=12, seed=0, n_moves=3,
                           verbose=True):
    """No-LLM control for option (C): random targeted moves on the topology order, kept only
    if they beat topology. The fair 'is the LLM beating dumb targeted search?' baseline."""
    hubs = hub_names(placedb, top_n)
    N = len(hubs)
    rng = random.Random(seed)
    topo = topology_all(placedb)
    best_order = topo
    best_h, best_env, npl = _score(placedb, topo, grid)
    curve = [best_h]
    if verbose:
        print(f"random-promote start: topology floor HPWL={best_h:.4e} placed={npl}")
    for it in range(1, n_evals + 1):
        moves = [(rng.randrange(N), rng.randrange(N)) for _ in range(n_moves)]
        order = apply_moves(best_order, moves, hubs)
        h, env, npl = _score(placedb, order, grid)
        keep = h < best_h
        if keep:
            best_order, best_h, best_env = order, h, env
        curve.append(best_h)
        if verbose:
            tag = "keep" if keep else ("drop" if h < INF else "drop(incomplete)")
            print(f"  rand iter {it:2d}: HPWL={h:.4e} ({tag}) placed={npl} best={best_h:.4e}")
    return {"hpwl": best_h, "best_env": best_env, "best_order": best_order, "curve": curve}


def llm_promote_search(placedb, top_n=300, grid=448, model="claude-sonnet-4-6",
                       max_iters=8, patience=3, verbose=True):
    """Option (C): the LLM proposes a few targeted 'place A before B' moves on the topology
    order, scored on the integrated LEGAL full-932 layout, accept-if-improves, floored at
    topology. The structure-preserving lever most likely to beat the strong baseline."""
    from strong_search import long_links
    from parse_netlist import build_summary, macro_connections
    from region_constraint import REGION_LABELS

    hubs = hub_names(placedb, top_n)
    id_of = {name: i for i, name in enumerate(hubs)}

    saved = placedb.node_id_to_name
    placedb.node_id_to_name = list(hubs)
    summary = build_summary(placedb, len(hubs), REGION_LABELS, top_k_conn=30)
    link_pairs = macro_connections(placedb, hubs, top_k=40)
    placedb.node_id_to_name = saved
    links = [(hubs[i], hubs[j], c) for i, j, c in link_pairs]

    adv = PromoteAdvisor(summary, len(hubs), model=model)

    topo = topology_all(placedb)
    best_order = topo
    best_h, best_env, npl = _score(placedb, topo, grid)
    curve = [best_h]
    if verbose:
        print(f"iter 0 (topology floor): HPWL={best_h:.4e} placed={npl} | targeted moves over "
              f"top {len(hubs)} hubs")

    attempts = [(best_h, True, "baseline")]
    tried = {tuple(topo)}

    def history_text():
        lines = [f"attempt {k}: HPWL={h:.4e} "
                 f"({note or ('accepted, NEW BEST' if acc else 'rejected')})"
                 for k, (h, acc, note) in enumerate(attempts)]
        lines.append(f"Best so far: HPWL={best_h:.4e}")
        return "\n".join(lines)

    no_improve = 0
    for it in range(1, max_iters + 1):
        fb = "LONG CONNECTIONS RIGHT NOW: " + long_links(best_env, id_of, links)
        proposed = adv.suggest(best_h, history_text(), fb)
        if not proposed:
            if verbose:
                print(f"iter {it:2d}: no proposal -> stop")
            break
        order = apply_moves(best_order, proposed, hubs)
        key = tuple(order)
        if key in tried:
            no_improve += 1
            attempts.append((best_h, False, "no-op / already-tried moves (skipped)"))
            if verbose:
                print(f"iter {it:2d}: no-op or already-tried -> skipped (no_improve={no_improve})")
            if no_improve >= patience:
                break
            continue
        tried.add(key)
        h, env, npl = _score(placedb, order, grid)
        accepted = h < best_h
        if accepted:
            best_order, best_h, best_env, no_improve = order, h, env, 0
        else:
            no_improve += 1
        attempts.append((h, accepted, ""))
        curve.append(best_h)
        if verbose:
            tag = "ACCEPT" if accepted else ("reject" if h < INF else "reject(incomplete)")
            print(f"iter {it:2d}: HPWL={h:.4e} ({tag}) placed={npl} best={best_h:.4e} "
                  f"moves={proposed}")
        if no_improve >= patience:
            if verbose:
                print(f"early stop: no improvement for {patience} iterations")
            break

    return {"hpwl": best_h, "best_env": best_env, "best_order": best_order, "curve": curve,
            "calls": adv.calls, "in_tokens": adv.in_tokens, "out_tokens": adv.out_tokens,
            "cache_read_tokens": adv.cache_read_tokens}


# --------------------------------------------------------------------------- #
# end-to-end comparison
# --------------------------------------------------------------------------- #
# MaskPlace paper reference (arXiv 2211.13382), ariane, grid 224, legal (0.00% overlap).
MASKPLACE_ARIANE_MST = 1.463e6


def evaluate_integrated(placedb, top_n=300, grid=448, model="claude-sonnet-4-6",
                        max_iters=8, patience=3, n_random=12, run_llm=True,
                        llm_action="promote", verbose=True):
    """Run heuristic baseline + matched random control (+ LLM search), all on the integrated
    LEGAL full-932 layout. Returns rows; each carries a legality check.

    llm_action="promote" (recommended, option C): the LLM makes targeted 'place A before B'
    moves on the topology order. llm_action="hub": the LLM reorders the top-N hubs to the
    front (weaker -- hubs-first tends not to beat topology). The random control matches the
    chosen action so the comparison is fair.
    """
    rows = []

    def row(method, res, calls=0, in_tok=0, out_tok=0, cache_tok=0):
        ov, oob = overlaps_and_oob(res["best_env"], grid)
        from trade_off_eval import usd_cost
        return {"method": method, "hpwl": res["hpwl"], "mst": mst_of(placedb, res["best_env"]),
                "overlaps": ov, "out_of_canvas": oob,
                "macros": len(res["best_env"].node_pos), "grid": grid, "calls": calls,
                "in_tokens": in_tok, "out_tokens": out_tok,
                "usd": usd_cost(model, in_tok, out_tok, cache_tok) if calls else 0.0}

    rows.append(row("heuristic baseline", heuristic_baseline(placedb, grid, verbose)))

    if llm_action == "promote":
        rc = random_promote_control(placedb, top_n=top_n, grid=grid, n_evals=n_random,
                                    verbose=verbose)
        rows.append(row("random promote ctrl", rc))
    else:
        rc = random_hub_control(placedb, top_n=top_n, grid=grid, n_evals=n_random,
                                verbose=verbose)
        rows.append(row("random hub control", rc))

    if run_llm:
        search = llm_promote_search if llm_action == "promote" else llm_hub_search
        r = search(placedb, top_n=top_n, grid=grid, model=model,
                   max_iters=max_iters, patience=patience, verbose=verbose)
        rows.append(row(f"LLM {llm_action}", r, calls=r["calls"], in_tok=r["in_tokens"],
                        out_tok=r["out_tokens"], cache_tok=r["cache_read_tokens"]))

    # MaskPlace reports MST (Table 2); it does not report bbox HPWL, so hpwl is None and its
    # number goes in the MST column -> compare MST-to-MST with our rows.
    rows.append({"method": "MaskPlace RL (paper)", "hpwl": None, "mst": MASKPLACE_ARIANE_MST,
                 "overlaps": 0, "out_of_canvas": 0, "macros": 932, "grid": 224, "calls": 0,
                 "in_tokens": 0, "out_tokens": 0, "usd": None})
    return rows


def print_integrated(rows):
    """Pretty-print the integrated comparison (all legal full-932). HPWL = bounding box;
    MST = spanning-tree (the paper's metric). Compare MST-to-MST with the MaskPlace row."""
    def fmt(v):
        return f"{v:.3e}" if v not in (None, INF) else "-"
    print("\n=== INTEGRATED LEGAL full-932 placement ===")
    hdr = (f"{'method':<22}{'HPWL(bbox)':>12}{'MST':>12}{'macros':>8}{'overlaps':>9}"
           f"{'oob':>5}{'grid':>6}{'$':>8}{'calls':>7}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        usd = f"{r['usd']:.3f}" if r.get("usd") is not None else "-"
        print(f"{r['method']:<22}{fmt(r.get('hpwl')):>12}{fmt(r.get('mst')):>12}"
              f"{str(r['macros']):>8}{str(r['overlaps']):>9}{str(r['out_of_canvas']):>5}"
              f"{str(r['grid']):>6}{usd:>8}{str(r['calls']):>7}")
    print("\nCompare the MST column to MaskPlace (its number is MST, grid 224). Our rows are"
          "\nlegal-by-construction at grid 448; judge the LLM by whether it beats BOTH the"
          "\nheuristic baseline and the random control.")
