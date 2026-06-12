"""Phase 4: Per-iteration log of HPWL + region choices.

Tracks each iteration's HPWL and the region assignment that produced it, exposes
a compact text history for the LLM prompt, remembers the best-so-far, and decides
when to early-stop (no improvement for ``patience`` iterations).
"""


class HistoryTracker:
    def __init__(self, patience=3):
        self.patience = patience
        self.records = []          # list of dict(iter, hpwl, regions, delta)
        self.best_hpwl = float("inf")
        self.best_iter = -1
        self.best_regions = None

    def record(self, it, hpwl, regions):
        prev = self.records[-1]["hpwl"] if self.records else None
        delta = None if prev is None else (hpwl - prev)
        self.records.append({"iter": it, "hpwl": hpwl,
                             "regions": dict(regions) if regions else {},
                             "delta": delta})
        if hpwl < self.best_hpwl:
            self.best_hpwl = hpwl
            self.best_iter = it
            self.best_regions = dict(regions) if regions else {}

    def get_history_text(self, max_rows=8):
        """Compact history for the prompt (most recent ``max_rows`` iterations)."""
        if not self.records:
            return "(no history yet -- this is the first placement)"
        rows = self.records[-max_rows:]
        lines = []
        for r in rows:
            if r["delta"] is None:
                change = ""
            else:
                pct = 100.0 * r["delta"] / max(1e-9, r["hpwl"] - r["delta"])
                arrow = "improved" if r["delta"] < 0 else "worse"
                change = f" ({arrow} {abs(pct):.1f}%)"
            lines.append(f"Iter {r['iter']}: HPWL={r['hpwl']:.4e}{change}")
        lines.append(f"Best so far: HPWL={self.best_hpwl:.4e} at iter {self.best_iter}")
        return "\n".join(lines)

    def should_stop(self):
        """Stop when there has been no new best for ``patience`` iterations."""
        if len(self.records) <= self.patience:
            return False
        last_iter = self.records[-1]["iter"]
        return (last_iter - self.best_iter) >= self.patience
