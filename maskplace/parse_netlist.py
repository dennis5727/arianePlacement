"""Phase 4: PlaceDB -> compact text summary for the LLM.

Produces a fixed (per-run) text block describing the chip: canvas size, the
macros being placed (with short ids M0..M{N-1} in placement order), their sizes,
the strongest macro-to-macro connections, and the region labels. This is the
static part of the prompt -- it does not change across iterations, so the LLM
loop sends it once and caches it.

The short ids M0..M{N-1} map to placedb.node_id_to_name[0..N-1], i.e. the exact
order the env places macros. Call this AFTER any hard_only filtering so that
node_id_to_name already holds the macros you intend to place.
"""

import re
from itertools import combinations


def short_name(full):
    """Human-readable short tag for a long hierarchical macro name.

    e.g. '.../i_nbdcache/sram_block_5__data_sram/mem/..._0x6' -> 'dca_blk5D_0x6'
    Falls back to the last path segment if the pattern is not found.
    """
    mod = "dca" if "i_nbdcache" in full else ("ica" if "i_icache" in full else "oth")
    m = re.search(r"sram_block_(\d+)__(\w+?)_sram", full)
    suffix = full.rsplit("_", 1)[-1]            # e.g. '0x6'
    if m:
        return f"{mod}_blk{m.group(1)}{m.group(2)[0].upper()}_{suffix}"
    return full.split("/")[-1][:24]


def macro_connections(placedb, names, top_k=15):
    """Return the top_k strongest macro<->macro connections among ``names``.

    Each entry is (i, j, n_shared_nets) with i < j as placement indices into
    ``names``. Counts how many nets connect each pair of macros.
    """
    index_of = {name: i for i, name in enumerate(names)}
    pair_count = {}
    for net in placedb.net_info.values():
        members = [index_of[n] for n in net["nodes"] if n in index_of]
        if len(members) < 2:
            continue
        for i, j in combinations(sorted(members), 2):
            pair_count[(i, j)] = pair_count.get((i, j), 0) + 1
    ranked = sorted(pair_count.items(), key=lambda kv: kv[1], reverse=True)
    return [(i, j, c) for (i, j), c in ranked[:top_k]]


def build_summary(placedb, n_macros, region_labels, top_k_conn=15):
    """Build the static chip-summary text block for the prompt."""
    names = placedb.node_id_to_name[:n_macros]
    lines = []
    lines.append("=== CHIP SUMMARY ===")
    lines.append(f"Canvas: {placedb.max_height} x {placedb.max_width} (square). "
                 f"Placing {n_macros} hard macros in fixed order M0..M{n_macros - 1}.")
    lines.append("")
    lines.append("MACROS (id | short name | width x height, canvas units):")
    for i, name in enumerate(names):
        w = placedb.node_info[name]["x"]
        h = placedb.node_info[name]["y"]
        lines.append(f"  M{i} | {short_name(name)} | {w:.0f}x{h:.0f}")
    lines.append("")
    lines.append("TOP CONNECTIONS (more shared nets = keep closer together):")
    conns = macro_connections(placedb, names, top_k=top_k_conn)
    if conns:
        lines.append("  " + "; ".join(f"M{i}<->M{j} ({c} nets)" for i, j, c in conns))
    else:
        lines.append("  (no macro-to-macro nets among this set)")
    lines.append("")
    lines.append("REGIONS (a 3x3 grid; higher row = top of canvas):")
    lines.append("  " + ", ".join(region_labels))
    return "\n".join(lines)
