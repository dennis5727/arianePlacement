"""Phase 2: Training-free greedy placement using wiremask argmin.

Loads PlaceDB, resets the MaskPlace env, and places each macro (in
``node_id_to_name`` order) at the legal cell with the lowest wiremask
value. No neural network, no training -- this is Mode A and produces
baseline row #1 (pure greedy) for the project.

Engine facts this relies on (see place_env/place_env.py):
  * state = [count, canvas(G^2), wiremask(G^2), posmask(G^2),
             wiremask2(G^2), mask2(G^2), next_x/G, next_y/G]
  * position mask slice = state[1+2*G*G : 1+3*G*G]; 1 == illegal cell
  * action = x*G + y  (step decodes x=action//G, y=action%G)
  * wiremask for the current macro = env.get_net_img() (raw, un-normalized)
  * HPWL = comp_res(placedb, env.node_pos, env.ratio)[0]

Usage (from the maskplace/ code dir):
    python greedy_place.py --benchmark ariane --pnm 128 --grid 224 \
        --fig greedy_ariane.png
or import:
    from greedy_place import greedy_place
    res = greedy_place(benchmark="ariane", pnm=128)
    print(res["hpwl"])
"""

import argparse
import numpy as np


def run_greedy(placedb, pnm=128, grid=224, verbose=True):
    """Place ``pnm`` macros greedily on the wiremask. Returns (env, n_placed)."""
    from place_env.place_env import PlaceEnv

    env = PlaceEnv(placedb, placed_num_macro=pnm, grid=grid)
    G = env.grid
    mask_lo, mask_hi = 1 + 2 * G * G, 1 + 3 * G * G

    state = env.reset()
    n_placed = 0
    for t in range(pnm):
        # Cost map for the macro about to be placed (node_id_to_name[t]).
        wiremask = env.get_net_img()                      # (G, G) raw cost
        posmask = state[mask_lo:mask_hi].reshape(G, G)    # 1 == illegal

        cost = wiremask.copy()
        cost[posmask == 1] = np.inf
        if not np.isfinite(cost).any():
            print(f"[warn] macro {t} ({env.node_name_list[t]}): "
                  f"no legal cell remaining -- stopping early.")
            break

        action = int(np.argmin(cost))                     # == x*G + y
        state, reward, done, _info = env.step(action)
        n_placed += 1

        if verbose and (t % 10 == 0 or t == pnm - 1):
            x, y = action // G, action % G
            print(f"  [{t:3d}] {env.node_name_list[t]:>22s} -> cell ({x:3d},{y:3d})"
                  f"  min_wiremask={wiremask[x, y]:.3g}  reward={reward:.1f}")

        if done:
            break

    return env, n_placed


def count_overlaps(env):
    """Count overlapping macro pairs (edge-touching is allowed, not an overlap)."""
    rects = list(env.node_pos.values())  # each: (x, y, size_x, size_y)
    n = 0
    for i in range(len(rects)):
        xi, yi, sxi, syi = rects[i]
        for j in range(i + 1, len(rects)):
            xj, yj, sxj, syj = rects[j]
            if xi < xj + sxj and xj < xi + sxi and yi < yj + syj and yj < yi + syi:
                n += 1
    return n


def select_hard_macros(placedb, order="keep"):
    """Return the list of hard (is_hard==1) macro names.

    order="keep" preserves their relative connectivity order from
    node_id_to_name; order="area" sorts largest-first.
    """
    hard = [n for n in placedb.node_id_to_name if placedb.node_info[n].get("is_hard")]
    if order == "area":
        hard.sort(key=lambda n: placedb.node_info[n]["x"] * placedb.node_info[n]["y"],
                  reverse=True)
    return hard


def greedy_place(benchmark="ariane", pnm=128, grid=224, save_fig=None, verbose=True,
                 hard_only=False, hard_order="keep"):
    """End-to-end greedy placement. Returns a result dict with HPWL etc.

    hard_only=True restricts placement to the hard SRAM macros only (Mode B
    focus from the plan). It overwrites placedb.node_id_to_name with the hard
    list -- the env reads that attribute to decide what to place, so no env
    change is needed -- and sets pnm to the number of hard macros.
    """
    from place_db import PlaceDB
    from comp_res import comp_res

    placedb = PlaceDB(benchmark)

    if hard_only:
        hard = select_hard_macros(placedb, order=hard_order)
        placedb.node_id_to_name = hard       # env places exactly these, in this order
        pnm = len(hard)
        if verbose:
            print(f"[hard_only] placing {pnm} hard MACROs only (order={hard_order})")

    env, n_placed = run_greedy(placedb, pnm=pnm, grid=grid, verbose=verbose)

    hpwl, cost = comp_res(placedb, env.node_pos, env.ratio)
    overlaps = count_overlaps(env)

    if save_fig:
        env.save_fig(save_fig)

    if verbose:
        print("\n=== GREEDY PLACEMENT RESULT ===")
        print(f"benchmark      : {benchmark}")
        print(f"grid           : {grid} x {grid}  (ratio 357/{grid} = {env.ratio:.4f})")
        print(f"macros placed  : {n_placed} / {pnm}")
        print(f"overlaps       : {overlaps}  ({'OK' if overlaps == 0 else 'BAD'})")
        print(f"HPWL           : {hpwl:.6e}")
        print(f"prim cost      : {cost:.6e}")
        if save_fig:
            print(f"layout figure  : {save_fig}")

    return {
        "benchmark": benchmark,
        "hpwl": hpwl,
        "cost": cost,
        "placed": n_placed,
        "overlaps": overlaps,
        "grid": grid,
        "ratio": env.ratio,
        "env": env,
        "placedb": placedb,
    }


def _parse_args():
    p = argparse.ArgumentParser(description="Training-free greedy wiremask placement (Mode A).")
    p.add_argument("--benchmark", default="ariane", help="benchmark name (default: ariane)")
    p.add_argument("--pnm", type=int, default=128, help="number of macros to place (default: 128)")
    p.add_argument("--grid", type=int, default=224, help="placement grid resolution (default: 224)")
    p.add_argument("--fig", default=None, help="optional path to save the layout PNG")
    p.add_argument("--quiet", action="store_true", help="suppress per-macro logging")
    p.add_argument("--hard-only", action="store_true",
                   help="place only the hard SRAM macros (overrides --pnm)")
    p.add_argument("--hard-order", default="keep", choices=["keep", "area"],
                   help="ordering of hard macros: keep connectivity order or sort by area")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    greedy_place(
        benchmark=args.benchmark,
        pnm=args.pnm,
        grid=args.grid,
        save_fig=args.fig,
        verbose=not args.quiet,
        hard_only=args.hard_only,
        hard_order=args.hard_order,
    )
