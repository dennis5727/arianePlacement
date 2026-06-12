"""Phase 3: Region label -> grid bounds -> boolean restriction mask.

Defines a 3x3 grid of named regions as fractions of the canvas, converts a
region label into a box of grid cells, and builds a boolean mask of the cells
where a macro's lower-left placement corner is allowed.

Coordinate convention (must match place_env.py):
  * action = x*G + y ; the macro's lower-left corner is placed at (x, y) and it
    occupies canvas[x : x+size_x, y : y+size_y].
  * x is the FIRST array axis, y is the SECOND. In env.save_fig, x maps to the
    horizontal axis and y to the vertical axis, so higher y == "top". The region
    fractions below follow that orientation (top = y in [.67, 1]).

A macro is constrained so its WHOLE footprint fits inside the region box
(fit_inside=True). If a region cannot hold the macro (too small, or fully
occupied), the caller falls back to an unconstrained legal placement -- this is
the ~20% "region full" fallback VeoPlace also reported.
"""

import numpy as np

# (x_min, x_max, y_min, y_max) as fractions of the canvas.
REGIONS = {
    "top-left":   (0.00, 0.33, 0.67, 1.00),
    "top-center": (0.33, 0.67, 0.67, 1.00),
    "top-right":  (0.67, 1.00, 0.67, 1.00),
    "mid-left":   (0.00, 0.33, 0.33, 0.67),
    "center":     (0.33, 0.67, 0.33, 0.67),
    "mid-right":  (0.67, 1.00, 0.33, 0.67),
    "bot-left":   (0.00, 0.33, 0.00, 0.33),
    "bot-center": (0.33, 0.67, 0.00, 0.33),
    "bot-right":  (0.67, 1.00, 0.00, 0.33),
}
REGION_LABELS = list(REGIONS.keys())


def region_bounds(label, grid):
    """Return (x_lo, x_hi, y_lo, y_hi) half-open grid-cell bounds for a region."""
    if label not in REGIONS:
        raise KeyError(f"unknown region {label!r}; valid labels: {REGION_LABELS}")
    xmn, xmx, ymn, ymx = REGIONS[label]
    x_lo, x_hi = int(np.floor(xmn * grid)), int(np.ceil(xmx * grid))
    y_lo, y_hi = int(np.floor(ymn * grid)), int(np.ceil(ymx * grid))
    return x_lo, x_hi, y_lo, y_hi


def region_mask(label, grid, size_x=1, size_y=1, fit_inside=True):
    """Boolean (grid, grid) mask: True where the macro's lower-left corner is allowed.

    With fit_inside, the entire size_x-by-size_y footprint must lie inside the
    region box, so valid corners stop ``size-1`` cells short of the far edge.
    Returns an all-False mask if the macro cannot fit in the region.
    """
    allowed = np.zeros((grid, grid), dtype=bool)
    x_lo, x_hi, y_lo, y_hi = region_bounds(label, grid)
    if fit_inside:
        x_hi = x_hi - size_x + 1
        y_hi = y_hi - size_y + 1
    if x_hi > x_lo and y_hi > y_lo:
        allowed[x_lo:x_hi, y_lo:y_hi] = True
    return allowed


def check_region_compliance(env, regions, grid, fit_inside=True):
    """Verify each placed macro sits inside its assigned region.

    regions: dict {placement_index -> region label}. Returns (n_violations,
    violation_list) where each violation is (index, name, (x,y,sx,sy), label).
    Macros with no assigned region are ignored.
    """
    violations = []
    name_list = env.node_name_list
    for idx, label in regions.items():
        if label is None:
            continue
        name = name_list[idx]
        if name not in env.node_pos:
            continue
        x, y, sx, sy = env.node_pos[name]
        x_lo, x_hi, y_lo, y_hi = region_bounds(label, grid)
        inside = (x >= x_lo and y >= y_lo and
                  (x + sx <= x_hi if fit_inside else x < x_hi) and
                  (y + sy <= y_hi if fit_inside else y < y_hi))
        if not inside:
            violations.append((idx, name, (x, y, sx, sy), label))
    return len(violations), violations


def uniform_regions(n_macros, label):
    """Convenience: assign every macro index 0..n_macros-1 to the same region."""
    if label not in REGIONS:
        raise KeyError(f"unknown region {label!r}; valid labels: {REGION_LABELS}")
    return {t: label for t in range(n_macros)}
