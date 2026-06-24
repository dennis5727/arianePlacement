"""Two-stage placement of ALL 932 ariane macros + LLM-guided hard ordering.

Why two stages
--------------
The wiremask greedy (greedy_place.run_greedy) places macros by WIRELENGTH, not by
space-packing. The full ariane netlist has 932 macros whose total area is ~78% of
the canvas (~94% of the 224 grid after cell rounding), so a literal "greedy all
932" runs out of legal cells and stops early. Instead we:

  1. HARD stage  -- place the 133 hard SRAM macros with the proven discrete
                    wiremask greedy (this is the order we search/LLM-guide), and
  2. SOFT stage  -- fill the 799 soft `Grp_*` std-cell clusters analytically by a
                    coordinate-median sweep with the hard macros held fixed.

`comp_res` over all 932 then gives the full-932 HPWL. The soft stage is
deterministic given the hard positions, so every method here (strong baseline,
random control, LLM full-order, LLM prefix) differs ONLY in the hard order.

Number caveat: full-932 HPWL counts soft pins on every net, so it is a NEW
baseline (~6.6e5) and is NOT comparable to the old hard-only ~8.4e4.

Everything is training-free. We reuse strong_search (greedy + LLM ordering),
greedy_place.run_greedy, comp_res, and parse_netlist unchanged.
"""

import json
import math
import random

import numpy as np

from strong_search import (
    greedy_hpwl,
    long_links,
    repair_perm,
    perturb,
    OrderingAdvisor,
)


# --------------------------------------------------------------------------- #
# soft stage: analytical median fill of the 799 soft clusters
# --------------------------------------------------------------------------- #
def _soft_names(placedb):
    return [n for n in placedb.node_info if not placedb.node_info[n].get("is_hard")]


def _node_size(placedb, name, ratio):
    """Macro footprint in grid cells (matches place_env / greedy_place)."""
    sx = math.ceil(max(1, placedb.node_info[name]["x"] / ratio))
    sy = math.ceil(max(1, placedb.node_info[name]["y"] / ratio))
    return sx, sy


def soft_fill(placedb, env, grid, iters=8, init="median"):
    """Place the soft clusters by an HPWL-reducing coordinate-median sweep.

    The 133 hard macros are already in ``env.node_pos`` (integer grid cells); they
    and the ports stay FIXED. Each soft node is moved to the coordinate-wise median
    of its net-neighbor pin centers, which minimizes the per-net 1-D bbox span, so
    HPWL drops and plateaus in a few sweeps. Soft positions are written back as
    INTEGER grid cells (cx-sx/2 rounded+clipped) -- fractional corners make
    comp_res trip its internal `assert hpwl_tmp <= prim_cost` on a few 2-node nets.

    Mutates env.node_pos in place (adds the soft nodes) and returns env.
    """
    ratio = env.ratio
    soft = _soft_names(placedb)
    soft_set = set(soft)

    sizes = {n: _node_size(placedb, n, ratio) for n in soft}

    # FIXED centers (grid cells): placed hard macros + ports.
    fixed = {}
    for name, (x, y, sx, sy) in env.node_pos.items():
        fixed[name] = (x + sx / 2.0, y + sy / 2.0)
    for p in placedb.port_info:
        fixed[p] = (placedb.port_info[p]["x"] / ratio,
                    placedb.port_info[p]["y"] / ratio)

    # adjacency: soft node -> list of (neighbor names) per net it belongs to.
    # We pre-flatten each net's members so a sweep is a few list walks.
    net_members = {}
    for nm, info in placedb.net_info.items():
        members = list(info["nodes"].keys()) + list(info["ports"].keys())
        net_members[nm] = members
    n2net = placedb.node_to_net_dict

    # neighbor name lists per soft node (excluding itself), built once.
    neighbors = {}
    for n in soft:
        nbrs = []
        for nm in n2net.get(n, ()):
            for m in net_members.get(nm, ()):
                if m != n:
                    nbrs.append(m)
        neighbors[n] = nbrs

    center = (grid / 2.0, grid / 2.0)
    soft_ctr = {}
    for n in soft:
        if init == "center":
            soft_ctr[n] = center
        else:  # median of already-fixed (hard/port) neighbors
            xs = [fixed[m][0] for m in neighbors[n] if m in fixed]
            ys = [fixed[m][1] for m in neighbors[n] if m in fixed]
            soft_ctr[n] = (float(np.median(xs)), float(np.median(ys))) if xs else center

    def cur_center(m):
        c = fixed.get(m)
        if c is not None:
            return c
        return soft_ctr.get(m)

    for _ in range(iters):
        for n in soft:
            xs = []
            ys = []
            for m in neighbors[n]:
                c = cur_center(m)
                if c is not None:
                    xs.append(c[0])
                    ys.append(c[1])
            if xs:
                soft_ctr[n] = (float(np.median(xs)), float(np.median(ys)))

    # write back as integer grid cells (lower-left corner = center - size/2)
    for n in soft:
        cx, cy = soft_ctr[n]
        sx, sy = sizes[n]
        x = min(max(int(round(cx - sx / 2.0)), 0), grid - 1)
        y = min(max(int(round(cy - sy / 2.0)), 0), grid - 1)
        env.node_pos[n] = (x, y, sx, sy)
    return env


def two_stage_hpwl(placedb, hard_order, grid=224, soft_iters=8):
    """Place 133 hard (given order) + fill 799 soft; return (full932_hpwl, env).

    NOTE: the soft stage minimizes wirelength only -- it does NOT avoid overlaps,
    so this env has the 799 soft clusters stacked on top of each other and the hard
    macros (~94% grid density). The HPWL is therefore an OPTIMISTIC, NON-legal
    number: connected nodes are free to pile up. For a real, overlap-free 932 HPWL
    use two_stage_legal_hpwl below.
    """
    from comp_res import comp_res

    env, _hard_hpwl, _fb = greedy_hpwl(placedb, hard_order, grid)
    soft_fill(placedb, env, grid, iters=soft_iters)
    hpwl, _cost = comp_res(placedb, env.node_pos, env.ratio)
    return hpwl, env


# --------------------------------------------------------------------------- #
# legalization: make ALL 932 macros overlap-free and inside the canvas
# --------------------------------------------------------------------------- #
def _rect_free(occ, x, y, sx, sy, g):
    """True if the sx-by-sy footprint at corner (x, y) is in-grid and unoccupied."""
    if x < 0 or y < 0 or x + sx > g or y + sy > g:
        return False
    return not occ[x:x + sx, y:y + sy].any()


def _nearest_free(occ, tx, ty, sx, sy, g):
    """Nearest (Chebyshev-ring) free corner to target (tx, ty); None if grid is full."""
    if _rect_free(occ, tx, ty, sx, sy, g):
        return tx, ty
    for r in range(1, g):
        x0, x1, y0, y1 = tx - r, tx + r, ty - r, ty + r
        for x in range(x0, x1 + 1):                       # top & bottom edges of ring
            if _rect_free(occ, x, y0, sx, sy, g):
                return x, y0
            if _rect_free(occ, x, y1, sx, sy, g):
                return x, y1
        for y in range(y0 + 1, y1):                       # left & right edges of ring
            if _rect_free(occ, x0, y, sx, sy, g):
                return x0, y
            if _rect_free(occ, x1, y, sx, sy, g):
                return x1, y
    return None


def legalize_placement(placedb, env, grid, refine=2):
    """Remove ALL overlaps and keep every macro inside the canvas, on a refined grid.

    soft_fill leaves the 799 soft clusters massively overlapping. This pass makes
    the whole 932-macro layout legal. It re-expresses the layout on a ``refine`` x
    finer grid -- which undoes the coarse-grid ceil() area inflation that made the
    224 grid look ~94% full when the true canvas is only ~78% full -- then:

      * the 133 hard macros keep their legal greedy positions (scaled by ``refine``,
        an exact remap because the footprints just scale), and
      * each soft cluster is snapped to the NEAREST free spot to its soft_fill
        target (largest-first Tetris fill), so it stays near its wirelength-optimal
        location while overlaps are removed.

    Mutates env.node_pos to the refined integer cells (and env.ratio) and returns
    (env, ratio_fine, unplaced_names). Score with comp_res(placedb, env.node_pos,
    ratio_fine). With refine=2 all 932 ariane macros fit with zero overlaps.
    """
    g = grid * refine
    ratio_f = placedb.max_height / g
    occ = np.zeros((g, g), dtype=bool)
    new_pos = {}
    hard = {n for n in placedb.node_info if placedb.node_info[n].get("is_hard")}

    # hard macros: exact scale of the already-legal greedy positions
    for name, (x, y, sx, sy) in env.node_pos.items():
        if name in hard:
            X, Y, SX, SY = x * refine, y * refine, sx * refine, sy * refine
            occ[X:X + SX, Y:Y + SY] = True
            new_pos[name] = (X, Y, SX, SY)

    def fsize(name):
        return (math.ceil(max(1, placedb.node_info[name]["x"] / ratio_f)),
                math.ceil(max(1, placedb.node_info[name]["y"] / ratio_f)))

    # largest-first packs tighter and leaves small holes for the small clusters
    soft = sorted((n for n in env.node_pos if n not in hard),
                  key=lambda n: -(fsize(n)[0] * fsize(n)[1]))
    unplaced = []
    for name in soft:
        sx, sy = fsize(name)
        tx = min(max(int(round(env.node_pos[name][0] * refine)), 0), g - sx)
        ty = min(max(int(round(env.node_pos[name][1] * refine)), 0), g - sy)
        spot = _nearest_free(occ, tx, ty, sx, sy, g)
        if spot is None:                                  # grid genuinely full
            unplaced.append(name)
            continue
        X, Y = spot
        occ[X:X + sx, Y:Y + sy] = True
        new_pos[name] = (X, Y, sx, sy)

    env.node_pos = new_pos
    env.ratio = ratio_f               # keep env self-consistent for comp_res
    return env, ratio_f, unplaced


def two_stage_legal_hpwl(placedb, hard_order, grid=224, soft_iters=8, refine=2):
    """Full LEGAL 932 placement: hard greedy + soft fill + legalization.

    Returns (legal_bbox_hpwl, legal_mst_cost, env, ratio_fine, n_unplaced). Unlike
    two_stage_hpwl this layout has ZERO overlaps and every macro inside the canvas,
    so the numbers are real, comparable -- and necessarily HIGHER than the
    overlapping one, because connected clusters can no longer pile on the same spot.

    Two wirelength numbers are returned because they match different baselines:
      * bbox_hpwl -- half-perimeter bounding box (what comp_res calls hpwl), and
      * mst_cost  -- the minimum-spanning-tree estimate (comp_res's prim cost),
        which is the metric MaskPlace's paper Table 2 reports.
    """
    from comp_res import comp_res

    env, _hard_hpwl, _fb = greedy_hpwl(placedb, hard_order, grid)
    soft_fill(placedb, env, grid, iters=soft_iters)
    env, ratio_f, unplaced = legalize_placement(placedb, env, grid, refine=refine)
    bbox_hpwl, mst_cost = comp_res(placedb, env.node_pos, ratio_f)
    return bbox_hpwl, mst_cost, env, ratio_f, len(unplaced)


def legal_scores(placedb, hard_order, grid=224, soft_iters=8, refine=2):
    """Thin wrapper over two_stage_legal_hpwl for the trade-off harness.

    Returns dict(bbox, mst, n_unplaced, grid_fine) -- the legal (zero-overlap)
    wirelength of a hard order, scored apples-to-apples with MaskPlace.
    """
    bbox, mst, _env, _ratio_f, n_unplaced = two_stage_legal_hpwl(
        placedb, hard_order, grid=grid, soft_iters=soft_iters, refine=refine)
    return {"bbox": bbox, "mst": mst, "n_unplaced": n_unplaced, "grid_fine": grid * refine}


# --------------------------------------------------------------------------- #
# no-LLM control: random hill-climb over the HARD order, scored on full 932
# --------------------------------------------------------------------------- #
def random_order_control(placedb, hard_names, grid=224, n_evals=12, soft_iters=8,
                         seed=0, verbose=True, legal=False, refine=2):
    """Hill-climb the hard order by random block moves; score = full-932 HPWL.

    Starts from the strong (connectivity/topology) order = identity and keeps any
    block perturbation that lowers full-932 HPWL. This is the fair 'is the LLM
    beating dumb search?' control, mirroring strong_search.random_restart_search.

    legal=True scores on the zero-overlap LEGAL layout, so this is the no-LLM control
    for the LLM-guided legal placement.
    """
    n = len(hard_names)
    rng = random.Random(seed)
    best = list(range(n))
    best_hpwl, best_env = _score_order(placedb, hard_names, grid, soft_iters, legal, refine)
    curve = [best_hpwl]
    if verbose:
        print(f"random-control start (strong order): full932 HPWL={best_hpwl:.4e}")
    for it in range(1, n_evals + 1):
        cand = perturb(best, rng)
        h, env = _score_order(placedb, [hard_names[i] for i in cand], grid, soft_iters,
                              legal, refine)
        keep = h < best_hpwl
        if keep:
            best, best_hpwl, best_env = cand, h, env
        curve.append(best_hpwl)
        if verbose:
            print(f"  rand iter {it:2d}: full932 HPWL={h:.4e} "
                  f"({'keep' if keep else 'drop'})  best={best_hpwl:.4e}")
    return {"best_perm": best, "best_hpwl": best_hpwl, "curve": curve, "best_env": best_env}


# --------------------------------------------------------------------------- #
# LLM advisors / searches (hard order; scored on full 932)
# --------------------------------------------------------------------------- #
class PrefixAdvisor(OrderingAdvisor):
    """Asks for a short PRIORITY PREFIX of hub macros to place first, not a full
    permutation. The unlisted macros keep the strong baseline order (via
    repair_perm), so the model only has to identify the few macros that matter
    most -- far fewer output tokens than a full 133-id order."""

    SYSTEM = (
        "You are an expert chip floorplanning assistant. A training-free greedy placer "
        "places macros ONE AT A TIME, each at the cell that adds the least wirelength given "
        "the macros already placed. Placing a highly-connected hub early (so later macros "
        "cluster around it) shortens total wirelength. Your job is to pick the SHORT LIST of "
        "the most important macros to place FIRST; every macro you do not list keeps its "
        "default position. You always answer with a single JSON array of macro ids."
    )

    def __init__(self, *args, k=50, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k

    def _user_msg(self, best_hpwl, best_order, history_text, feedback, has_image):
        return (
            f"There are {self.n} hard macros with ids M0..M{self.n - 1} (see summary).\n\n"
            "=== PREFIXES TRIED SO FAR ===\n"
            f"{history_text}\n\n"
            "=== CURRENT BEST PRIORITY PREFIX (your anchor) ===\n"
            f"{json.dumps(list(best_order))}\n"
            "These macros are placed FIRST, in this order; all others keep the strong "
            "baseline order afterwards. Start from THIS prefix and make a TARGETED change "
            "-- add or promote a hub macro, drop one that is not helping, or reorder a "
            "tightly-connected group so its members are early and consecutive.\n\n"
            "=== CURRENT BEST LAYOUT FEEDBACK ===\n"
            f"{feedback}\n\n"
            "=== YOUR TASK ===\n"
            f"Current best full-932 HPWL: {best_hpwl:.4e} (lower is better).\n"
            f"Propose a NEW priority prefix: a SHORT JSON array of about {self.k} ids (the "
            "most-connected hub macros to place first), NOT all ids. Do NOT repeat a prefix "
            "already tried above. Think briefly, then output ONLY a JSON array of ids, e.g. "
            "[3,17,2,...]."
        )


def _build_prompt_context(placedb, hard_names, top_k_conn=30, top_k_links=40):
    """Build the static chip summary + macro-link pairs in canonical (hard) id order."""
    from parse_netlist import build_summary, macro_connections
    from region_constraint import REGION_LABELS

    n = len(hard_names)
    saved = placedb.node_id_to_name
    placedb.node_id_to_name = list(hard_names)
    try:
        summary = build_summary(placedb, n, REGION_LABELS, top_k_conn=top_k_conn)
        link_pairs = macro_connections(placedb, hard_names, top_k=top_k_links)
    finally:
        placedb.node_id_to_name = saved
    links = [(hard_names[i], hard_names[j], c) for i, j, c in link_pairs]
    return summary, links


def _score_order(placedb, order_names, grid, soft_iters, legal, refine):
    """Score one hard order. legal=True -> zero-overlap LEGAL bbox HPWL (the real,
    MaskPlace-comparable number); legal=False -> the fast OVERLAPPING proxy. Returns
    (hpwl, env), where env is the (legalized, if legal) layout."""
    if legal:
        bbox, _mst, env, _ratio_f, _n = two_stage_legal_hpwl(
            placedb, order_names, grid=grid, soft_iters=soft_iters, refine=refine)
        return bbox, env
    return two_stage_hpwl(placedb, order_names, grid, soft_iters)


def _llm_search(placedb, hard_names, advisor, complete_fn, grid=224, soft_iters=8,
                max_iters=8, patience=3, links=None, init_anchor=None, verbose=True,
                what="order", legal=False, refine=2):
    """Shared best-anchored, accept-if-improves loop scored on full-932 HPWL.

    complete_fn(proposed_ids, n) -> a full permutation (0..n-1) used for scoring.
    init_anchor is the starting "best" value shown to the model (full perm for the
    full-order search, short prefix [] for the prefix search). The strong baseline
    (iter 0) is always the identity hard order, so the result never regresses.

    legal=True scores each candidate on the LEGAL (zero-overlap, in-canvas) layout
    via two_stage_legal_hpwl, so best_perm/best_env are a real legal placement of all
    932 macros -- the LLM-guided result that is directly comparable to MaskPlace.
    """
    n = len(hard_names)
    id_of = {name: i for i, name in enumerate(hard_names)}

    best_perm = list(range(n))                       # identity = strong order
    best_hpwl, best_env = _score_order(placedb, hard_names, grid, soft_iters, legal, refine)
    anchor = list(range(n)) if init_anchor is None else list(init_anchor)
    curve = [best_hpwl]
    if verbose:
        print(f"iter 0 (strong baseline): full932 HPWL={best_hpwl:.4e}")

    attempts = [(best_hpwl, True, "baseline")]
    tried = {tuple(best_perm)}

    def history_text():
        lines = []
        for k, (h, acc, note) in enumerate(attempts):
            tag = note if note else ("accepted, NEW BEST" if acc else "rejected")
            lines.append(f"attempt {k}: HPWL={h:.4e} ({tag})")
        lines.append(f"Best so far: HPWL={best_hpwl:.4e}")
        return "\n".join(lines)

    no_improve = 0
    for it in range(1, max_iters + 1):
        fb = "(none)"
        if links is not None:
            fb = "LONG CONNECTIONS RIGHT NOW: " + long_links(best_env, id_of, links)
        proposed = advisor.suggest(best_hpwl, anchor, history_text(), feedback=fb,
                                   image_path=None)
        if proposed is None:
            if verbose:
                print(f"iter {it:2d}: no proposal -> stop")
            break
        perm = complete_fn(proposed, n)
        key = tuple(perm)
        if key in tried:
            no_improve += 1
            attempts.append((best_hpwl, False, "duplicate of an earlier order (skipped)"))
            if verbose:
                print(f"iter {it:2d}: already-tried order -> skipped (no_improve={no_improve})")
            if no_improve >= patience:
                if verbose:
                    print(f"early stop: no improvement for {patience} iterations")
                break
            continue
        tried.add(key)
        h, env = _score_order(placedb, [hard_names[i] for i in perm], grid, soft_iters,
                              legal, refine)
        accepted = h < best_hpwl
        if accepted:
            best_perm, best_hpwl, best_env, no_improve = perm, h, env, 0
            # anchor follows the accepted proposal so targeted edits build on it
            anchor = [int(x) for x in proposed] if what == "prefix" else perm
        else:
            no_improve += 1
        attempts.append((h, accepted, ""))
        curve.append(best_hpwl)
        if verbose:
            print(f"iter {it:2d}: full932 HPWL={h:.4e} "
                  f"({'ACCEPT' if accepted else 'reject'})  best={best_hpwl:.4e}")
        if no_improve >= patience:
            if verbose:
                print(f"early stop: no improvement for {patience} iterations")
            break

    return {"best_perm": best_perm, "best_hpwl": best_hpwl, "curve": curve,
            "best_env": best_env, "calls": advisor.calls, "in_tokens": advisor.in_tokens,
            "out_tokens": advisor.out_tokens, "cache_read_tokens": advisor.cache_read_tokens}


def llm_full_order_search(placedb, hard_names, grid=224, model="claude-sonnet-4-6",
                          max_iters=8, patience=3, soft_iters=8, verbose=True,
                          legal=False, refine=2):
    """LLM proposes a FULL order of all 133 hard macros; scored on full-932 HPWL.

    legal=True scores on the zero-overlap LEGAL layout, so the returned best_perm/
    best_env are a real legal 932-macro placement comparable to MaskPlace."""
    n = len(hard_names)
    summary, links = _build_prompt_context(placedb, hard_names)
    advisor = OrderingAdvisor(summary, n, model=model)

    def complete_fn(proposed, n):
        # missing/junk ids fall back to the current best (anchor handled in loop);
        # here we complete against the strong baseline identity order.
        return repair_perm(proposed, n, list(range(n)))

    return _llm_search(placedb, hard_names, advisor, complete_fn, grid=grid,
                       soft_iters=soft_iters, max_iters=max_iters, patience=patience,
                       links=links, init_anchor=list(range(n)), verbose=verbose,
                       what="order", legal=legal, refine=refine)


def llm_prefix_search(placedb, hard_names, grid=224, model="claude-sonnet-4-6",
                      max_iters=8, patience=3, k=50, soft_iters=8, verbose=True,
                      legal=False, refine=2):
    """LLM proposes a short PRIORITY PREFIX of hub macros; the rest keep the strong
    baseline order. Scored on full-932 HPWL. Far fewer output tokens than full-order.

    legal=True scores on the zero-overlap LEGAL layout (see llm_full_order_search)."""
    n = len(hard_names)
    summary, links = _build_prompt_context(placedb, hard_names)
    advisor = PrefixAdvisor(summary, n, model=model, k=k)
    strong = list(range(n))

    def complete_fn(proposed, n):
        # keep only the prefix ids, then append the rest in strong-baseline order
        return repair_perm(proposed, n, strong)

    return _llm_search(placedb, hard_names, advisor, complete_fn, grid=grid,
                       soft_iters=soft_iters, max_iters=max_iters, patience=patience,
                       links=links, init_anchor=[], verbose=verbose, what="prefix",
                       legal=legal, refine=refine)
