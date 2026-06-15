"""Phase 5: render the placement + region-grid overlay as a PNG.

Used both for the report figures and as the image sent to the LLM. Constrained
macros are drawn red, free macros blue; the 3x3 region grid is dashed with region
names, and each macro is annotated with its id (M-index) so the model can map the
text feedback (e.g. "M5<->M12 far apart") to the picture.

Coordinate convention matches place_env: x = horizontal, y = vertical, higher y
is the top of the canvas.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from region_constraint import REGIONS


def render_placement(env, grid, names=None, regions=None, path="layout.png",
                     label=True, title=None):
    """Render env.node_pos to ``path``. Returns the path.

    names   : list mapping placement index -> macro name (for ids/highlighting).
    regions : dict {index -> label}; those macros are drawn red (constrained).
    """
    regions = regions or {}
    id_of = {n: i for i, n in enumerate(names)} if names else {}

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)

    # region grid (thirds) + labels
    for f in (1 / 3.0, 2 / 3.0):
        ax.axhline(f, color="0.6", lw=0.6, ls="--")
        ax.axvline(f, color="0.6", lw=0.6, ls="--")
    for lab, (xmn, xmx, ymn, ymx) in REGIONS.items():
        ax.text((xmn + xmx) / 2, (ymn + ymx) / 2, lab, ha="center", va="center",
                color="0.8", fontsize=7, zorder=0)

    # macros
    for name, (x, y, sx, sy) in env.node_pos.items():
        idx = id_of.get(name)
        constrained = idx in regions
        ax.add_patch(patches.Rectangle(
            (x / grid, y / grid), sx / grid, sy / grid,
            facecolor=("tab:red" if constrained else "tab:blue"),
            edgecolor="k", lw=0.4, alpha=0.55, zorder=1))
        if label and idx is not None:
            ax.text((x + sx / 2) / grid, (y + sy / 2) / grid, str(idx),
                    ha="center", va="center", fontsize=4.5, zorder=2)

    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path
