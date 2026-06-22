"""Strong-baseline experiment: LLM-guided PLACEMENT-ORDER search.

Motivation
----------
The greedy placer is order-dependent: the macro placement order swings ariane
HPWL from ~8.6e4 (MaskPlace's connectivity order) to ~1.25e5 (area order) -- a
~45% spread. Region constraints, by contrast, tend to HURT an already-good
greedy. So the lever with real headroom is the ORDER, not the regions.

This module searches the ordering space and asks whether an Anthropic model can
find an order that beats the strong (connectivity) baseline -- and whether it
beats a no-LLM random-restart search of the same space (the fair control).

Everything here is training-free. We reuse our Phase-2 greedy driver
(`run_greedy`) and MaskPlace's wiremask/env unchanged; we only change the order
in which macros are fed to the placer.

Vocabulary
----------
* canonical ids  : a FIXED id<->macro mapping (connectivity order). "M5" always
                   means the same macro, regardless of the order being tried.
* placement order: a permutation of canonical ids; this is what we search over.
"""

import json
import os
import random
import re

import numpy as np


# --------------------------------------------------------------------------- #
# placement helpers (reuse the Phase-2 greedy + MaskPlace engine unchanged)
# --------------------------------------------------------------------------- #
def hard_macro_names(placedb):
    """All hard (is_hard==1) macro names, order-independent."""
    return [n for n, info in placedb.node_info.items() if info.get("is_hard")]


def _net_count(placedb, name):
    return len(placedb.node_to_net_dict.get(name, []))


def connectivity_order(placedb, names):
    """STRONG baseline order = MaskPlace's topology (frontier-growth) order,
    filtered to ``names``.

    This is the order that yields the ~8.6e4 ariane baseline. It is NOT a
    degree sort -- a plain "most-connected-first" sort scatters connected
    macros and scores like the weak order (~1.24e5). The real strength comes
    from frontier growth (each macro added is the one most-connected to those
    already placed), so connected macros end up consecutive. That traversal is
    already computed at PlaceDB init and stored in placedb.node_id_to_name, so
    we just filter it here.

    Call this BEFORE mutating placedb.node_id_to_name (i.e. right after
    PlaceDB(...)). Run-to-run it can drift ~3% because the underlying traversal
    uses a hash() tiebreak; for a fixed order use topology_order() below.
    """
    keep = set(names)
    return [n for n in placedb.node_id_to_name if n in keep]


def topology_order(placedb, names=None):
    """Deterministic reproduction of MaskPlace's frontier-growth order.

    Same selection rule as place_db.get_node_id_to_name_topology for ariane
    (candidates*30000 + degree*1000 + area + tiebreak) but with a STABLE
    md5 tiebreak instead of hash(), so it is identical every run. Traverses the
    full node graph, then optionally filters to ``names``. Use this for a
    reproducible strong baseline; sanity-check it matches connectivity_order().
    """
    import hashlib
    from itertools import combinations

    ni = placedb.node_info
    n2net = placedb.node_to_net_dict

    adjacency = {}
    for net in placedb.net_info.values():
        members = list(net["nodes"])
        for a, b in combinations(members, 2):
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)

    degree = {n: len(n2net.get(n, [])) for n in ni}
    area = {n: ni[n]["x"] * ni[n]["y"] for n in ni}
    tie = {n: int(hashlib.md5(n.encode()).hexdigest(), 16) % 10000 * 1e-6 for n in ni}

    remaining = set(ni.keys())
    order, cand = [], {}

    def visit(node):
        order.append(node)
        remaining.discard(node)
        cand.pop(node, None)
        for nb in adjacency.get(node, ()):
            if nb in remaining:
                cand[nb] = cand.get(nb, 0) + 1

    if "V" in remaining:
        visit("V")
    visit(max(remaining, key=lambda v: (degree[v], tie[v])))
    while remaining:
        visit(max(remaining, key=lambda v: cand.get(v, 0) * 30000 + degree[v] * 1000
                  + area[v] + tie[v]))

    if names is not None:
        keep = set(names)
        order = [n for n in order if n in keep]
    return order


def area_order(placedb, names):
    """Largest-first (the weak but deterministic order)."""
    return sorted(names, key=lambda n: (-(placedb.node_info[n]["x"] * placedb.node_info[n]["y"]), n))


def greedy_hpwl(placedb, order_names, grid=224):
    """Run the Phase-2 greedy with a given full macro order. Returns (env, hpwl, n_fallback).

    Mutates placedb.node_id_to_name for the run, then restores it -- the env
    reads that attribute to decide what/which-order to place.
    """
    from greedy_place import run_greedy
    from comp_res import comp_res

    saved = placedb.node_id_to_name
    try:
        placedb.node_id_to_name = list(order_names)
        env, _n, fb = run_greedy(placedb, pnm=len(order_names), grid=grid, verbose=False)
        hpwl, _cost = comp_res(placedb, env.node_pos, env.ratio)
    finally:
        placedb.node_id_to_name = saved
    return env, hpwl, fb


def count_overlaps(env):
    from greedy_place import count_overlaps as _co
    return _co(env)


# --------------------------------------------------------------------------- #
# permutation utilities
# --------------------------------------------------------------------------- #
def repair_perm(proposed_ids, n, fallback_ids):
    """Turn a possibly-invalid id list into a valid permutation of 0..n-1.

    Keeps the first occurrence of each valid id, drops junk/dupes, then appends
    any missing ids in `fallback_ids` order (so unmentioned macros default to
    the strong-baseline position rather than the end-of-list).
    """
    seen, out = set(), []
    for x in proposed_ids:
        try:
            xi = int(x)
        except (TypeError, ValueError):
            continue
        if 0 <= xi < n and xi not in seen:
            out.append(xi)
            seen.add(xi)
    for x in fallback_ids:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def perturb(perm, rng, max_block=None):
    """Move a random contiguous block of the order to a new position."""
    p = list(perm)
    m = len(p)
    k = rng.randint(2, max(2, max_block or m // 6))
    i = rng.randint(0, m - k)
    block = p[i:i + k]
    del p[i:i + k]
    j = rng.randint(0, len(p))
    p[j:j] = block
    return p


def long_links(env, id_of, links, top=12):
    """Strongest macro<->macro links that are far apart in this layout (by canonical id)."""
    pos = env.node_pos
    scored = []
    for ni, nj, c in links:
        if ni not in pos or nj not in pos:
            continue
        xi, yi, sxi, syi = pos[ni]
        xj, yj, sxj, syj = pos[nj]
        d = abs((xi + sxi / 2) - (xj + sxj / 2)) + abs((yi + syi / 2) - (yj + syj / 2))
        scored.append((d, id_of[ni], id_of[nj], c))
    scored.sort(reverse=True)
    if not scored:
        return "(no placed macro-to-macro links)"
    return "; ".join(f"M{i}<->M{j} ~{d:.0f} apart ({c} nets)" for d, i, j, c in scored[:top])


# --------------------------------------------------------------------------- #
# searches
# --------------------------------------------------------------------------- #
def random_restart_search(placedb, canonical_names, grid=224, n_evals=12, seed=0, verbose=True):
    """No-LLM control: hill-climb over orderings by random block moves.

    Starts from the connectivity (strong) order and keeps any perturbation that
    lowers HPWL. This is the fair 'is the LLM beating dumb search?' baseline.
    """
    n = len(canonical_names)
    rng = random.Random(seed)
    best = list(range(n))                       # identity = canonical = connectivity order
    env, best_hpwl, _ = greedy_hpwl(placedb, canonical_names, grid)
    curve = [best_hpwl]
    if verbose:
        print(f"random-search start (connectivity): HPWL={best_hpwl:.4e}")
    for it in range(1, n_evals + 1):
        cand = perturb(best, rng)
        _e, h, _fb = greedy_hpwl(placedb, [canonical_names[i] for i in cand], grid)
        keep = h < best_hpwl
        if keep:
            best, best_hpwl, env = cand, h, _e
        curve.append(best_hpwl)
        if verbose:
            print(f"  rand iter {it:2d}: HPWL={h:.4e} ({'keep' if keep else 'drop'})  best={best_hpwl:.4e}")
    return {"best_perm": best, "best_hpwl": best_hpwl, "curve": curve, "best_env": env}


class OrderingAdvisor:
    """Anthropic-backed advisor that proposes a full placement ORDER."""

    SYSTEM = (
        "You are an expert chip floorplanning assistant. A training-free greedy placer "
        "places macros ONE AT A TIME, each at the cell that adds the least wirelength given "
        "the macros already placed. Because of this, the ORDER matters a lot: placing a "
        "highly-connected hub early (so later macros cluster around it) and keeping "
        "strongly-connected macros consecutive in the order produces shorter total "
        "wirelength. You propose a placement order to minimize HPWL. You always answer "
        "with a single JSON array of macro ids."
    )

    def __init__(self, summary, n, model="claude-sonnet-4-6", max_tokens=4000,
                 max_retries=2, api_key=None, temperature=None):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.summary = summary
        self.n = n
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.temperature = temperature
        self.calls = self.in_tokens = self.out_tokens = self.cache_read_tokens = 0

    def _user_msg(self, best_hpwl, best_order, history_text, feedback, has_image):
        img = ""
        if has_image:
            img = ("An image of the CURRENT best layout is attached; numbers are macro ids. "
                   "Use it to see which connected macros ended up far apart.\n")
        return (
            f"There are {self.n} macros with ids M0..M{self.n - 1} (see summary).\n\n"
            "=== ORDERS TRIED SO FAR ===\n"
            f"{history_text}\n\n"
            "=== CURRENT BEST ORDER (your anchor) ===\n"
            f"{json.dumps(best_order)}\n"
            "This exact order produced the best HPWL so far. Start from THIS order and make a "
            "TARGETED change -- pull one hub macro earlier, or move a tightly-connected group so "
            "its members become early and consecutive. Keep most of the order intact; do not "
            "reshuffle everything.\n\n"
            "=== CURRENT BEST LAYOUT FEEDBACK ===\n"
            f"{feedback}\n\n"
            "=== YOUR TASK ===\n"
            f"Current best HPWL: {best_hpwl:.4e} (lower is better).\n"
            f"{img}"
            "Propose a MODIFIED placement order (a permutation of ALL ids 0..%d) that you expect "
            "to lower HPWL. Do NOT repeat any order already tried above -- if a change was "
            "rejected, try a DIFFERENT one. Think briefly, then output ONLY a JSON array of "
            "all ids in placement order, e.g. [3,17,2,...]. The array MUST contain every id "
            "exactly once." % (self.n - 1)
        )

    def suggest(self, best_hpwl, best_order, history_text="(no attempts yet)",
                feedback="", image_path=None):
        content = []
        if image_path:
            import base64
            with open(image_path, "rb") as f:
                b64 = base64.standard_b64encode(f.read()).decode()
            content.append({"type": "image", "source": {"type": "base64",
                            "media_type": "image/png", "data": b64}})
        content.append({"type": "text", "text": self._user_msg(
            best_hpwl, list(best_order), history_text, feedback or "(none)", bool(image_path))})
        # temperature is deprecated on some models (e.g. claude-opus-4-8), so only
        # send it when explicitly set; omitting it uses the model default.
        kwargs = {} if self.temperature is None else {"temperature": self.temperature}
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.messages.create(
                    model=self.model, max_tokens=self.max_tokens,
                    system=[{"type": "text", "text": self.SYSTEM},
                            {"type": "text", "text": self.summary,
                             "cache_control": {"type": "ephemeral"}}],
                    messages=[{"role": "user", "content": content}],
                    **kwargs,
                )
                self.calls += 1
                u = resp.usage
                self.in_tokens += getattr(u, "input_tokens", 0)
                self.out_tokens += getattr(u, "output_tokens", 0)
                self.cache_read_tokens += getattr(u, "cache_read_input_tokens", 0) or 0
                text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
                ids = self._parse_array(text)
                if ids is not None:
                    return ids
                print(f"  [llm] attempt {attempt + 1}: no usable JSON array; retrying")
            except Exception as e:
                print(f"  [llm] attempt {attempt + 1} error: {e}")
        return None

    @staticmethod
    def _parse_array(text):
        for blob in reversed(re.findall(r"\[[^\[\]]*\]", text, flags=re.DOTALL)):
            try:
                arr = json.loads(blob)
            except json.JSONDecodeError:
                continue
            if isinstance(arr, list) and len(arr) >= 2:
                return arr
        return None


def llm_order_search(placedb, canonical_names, grid=224, model="claude-sonnet-4-6",
                     max_iters=8, patience=3, use_image=False, save_best_fig=None, verbose=True):
    """LLM-guided, best-anchored ordering search vs the strong (connectivity) baseline.

    iter 0 = connectivity order (the strong baseline = the floor). Each iter the
    model proposes a full order; we accept it only if HPWL improves. The result
    can never end up worse than the strong baseline.
    """
    from parse_netlist import build_summary, macro_connections
    from region_constraint import REGION_LABELS

    n = len(canonical_names)
    id_of = {name: i for i, name in enumerate(canonical_names)}
    # build summary/links with canonical ids
    saved = placedb.node_id_to_name
    placedb.node_id_to_name = list(canonical_names)
    summary = build_summary(placedb, n, REGION_LABELS, top_k_conn=30)
    link_pairs = macro_connections(placedb, canonical_names, top_k=40)
    placedb.node_id_to_name = saved
    links = [(canonical_names[i], canonical_names[j], c) for i, j, c in link_pairs]

    adv = OrderingAdvisor(summary, n, model=model)

    best = list(range(n))
    best_env, best_hpwl, _ = greedy_hpwl(placedb, canonical_names, grid)
    curve = [best_hpwl]
    if verbose:
        print(f"iter 0 (connectivity strong baseline): HPWL={best_hpwl:.4e} "
              f"overlaps={count_overlaps(best_env)}")

    # history of every distinct order evaluated, plus a set of all proposed orders
    # (incl. duplicates) for dedup so we never re-run the greedy on the same perm.
    attempts = [(best_hpwl, True, "baseline")]   # (hpwl, accepted, note)
    tried = {tuple(best)}

    def history_text():
        lines = []
        for k, (h, acc, note) in enumerate(attempts):
            tag = note if note else ("accepted, NEW BEST" if acc else "rejected")
            lines.append(f"attempt {k}: HPWL={h:.4e} ({tag})")
        lines.append(f"Best so far: HPWL={best_hpwl:.4e}")
        return "\n".join(lines)

    no_improve = 0
    for it in range(1, max_iters + 1):
        fb = "LONG CONNECTIONS RIGHT NOW: " + long_links(best_env, id_of, links)
        image_path = None
        if use_image:
            from visualize import render_placement
            image_path = render_placement(best_env, grid, names=canonical_names, regions={},
                                           path="strong_view.png")
        proposed = adv.suggest(best_hpwl, best, history_text(), feedback=fb, image_path=image_path)
        if proposed is None:
            if verbose:
                print(f"iter {it:2d}: no proposal -> stop")
            break
        # missing/junk ids fall back to the CURRENT BEST order, so an omitted tail
        # keeps the anchor instead of reverting to the canonical baseline.
        perm = repair_perm(proposed, n, best)
        key = tuple(perm)
        if key in tried:                       # dedup: don't re-evaluate a known order
            no_improve += 1
            attempts.append((best_hpwl, False, "duplicate of an earlier order (skipped)"))
            if verbose:
                print(f"iter {it:2d}: proposed an already-tried order -> skipped "
                      f"(no_improve={no_improve})")
            if no_improve >= patience:
                if verbose:
                    print(f"early stop: no improvement for {patience} iterations")
                break
            continue
        tried.add(key)
        env, h, fbk = greedy_hpwl(placedb, [canonical_names[i] for i in perm], grid)
        accepted = h < best_hpwl
        if accepted:
            best, best_hpwl, best_env, no_improve = perm, h, env, 0
        else:
            no_improve += 1
        attempts.append((h, accepted, ""))
        curve.append(best_hpwl)
        if verbose:
            print(f"iter {it:2d}: HPWL={h:.4e} ({'ACCEPT' if accepted else 'reject'})  "
                  f"best={best_hpwl:.4e}  overlaps={count_overlaps(env)}  fallback={fbk}")
        if no_improve >= patience:
            if verbose:
                print(f"early stop: no improvement for {patience} iterations")
            break

    if save_best_fig:
        try:
            from visualize import render_placement
            render_placement(best_env, grid, names=canonical_names, regions={}, path=save_best_fig)
        except Exception:
            best_env.save_fig(save_best_fig)

    return {"best_perm": best, "best_hpwl": best_hpwl, "curve": curve, "best_env": best_env,
            "calls": adv.calls, "in_tokens": adv.in_tokens, "out_tokens": adv.out_tokens,
            "cache_read_tokens": adv.cache_read_tokens}
