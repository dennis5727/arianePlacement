"""Phase 4 (text-only) / Phase 5 (+ image): the region advisor.

Two interchangeable advisors with the same ``suggest`` signature:

  * DummyAdvisor  -- no API calls; random PARTIAL regions. Use it to test the
    whole loop (ordering, history, early-stop) for FREE before spending tokens.
  * ClaudeAdvisor -- one Anthropic API call per iteration.

Key design (after the pnm=30 experiment showed hard full-partition hurts a
already-good greedy): the advisor proposes regions for ONLY the macros it wants
to relocate; every other macro is left UNCONSTRAINED so the greedy places it
optimally. The floor is therefore the unconstrained baseline -- a good nudge can
only help, a bad/empty one is neutral. The prompt also receives the current
"long connections" (strongly-connected macros that are far apart right now) so
the model has real placement feedback, not just static connectivity.

suggest(history_text, cur_hpwl, best_hpwl, prev_regions, diagnostics) -> dict|None
Returns {index: label} (possibly empty = free everything) or None on parse/API
failure (caller then reuses previous regions). Keyed by integer index (M5 -> 5).
"""

import json
import os
import random
import re
import time


SYSTEM_INSTRUCTIONS = (
    "You are an expert chip floorplanning assistant. A training-free greedy "
    "placer already puts every UNCONSTRAINED macro at its lowest-wirelength cell. "
    "You improve the layout by assigning a region ONLY to the few macros that are "
    "currently placed far from their strongly-connected partners, so they get "
    "pulled together. Constraining a macro that is already well placed only hurts, "
    "so be selective. You always answer with a single JSON object."
)


def _parse_region_json(text, valid_indices, valid_labels):
    """Extract {index: label} from an LLM reply.

    Returns a dict (possibly empty, meaning 'free everything') when a JSON object
    was found, or None when no JSON object could be parsed at all (a real error
    the caller should treat as a failed iteration).
    """
    matches = re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)
    parsed_any = False
    for blob in reversed(matches):                 # models reason before the JSON
        try:
            raw = json.loads(blob)
        except json.JSONDecodeError:
            continue
        parsed_any = True
        out = {}
        for k, v in raw.items():
            m = re.fullmatch(r"\s*M?(\d+)\s*", str(k))
            if not m:
                continue
            idx = int(m.group(1))
            label = str(v).strip()
            if idx in valid_indices and label in valid_labels:
                out[idx] = label
        if out:
            return out
    return {} if parsed_any else None


def _fmt_regions(regions):
    if not regions:
        return "(none -- all macros are currently free/unconstrained)"
    return ", ".join(f"M{i}:{lab}" for i, lab in sorted(regions.items()))


class DummyAdvisor:
    """Free, offline advisor: random PARTIAL regions. For plumbing tests only."""

    def __init__(self, macro_ids, region_labels, seed=0, fraction=0.3):
        self.macro_ids = list(macro_ids)
        self.region_labels = list(region_labels)
        self.rng = random.Random(seed)
        self.fraction = fraction
        self.calls = 0
        self.in_tokens = 0
        self.out_tokens = 0

    def suggest(self, history_text, cur_hpwl, best_hpwl, prev_regions=None, diagnostics=""):
        self.calls += 1
        k = max(1, int(len(self.macro_ids) * self.fraction))
        chosen = self.rng.sample(self.macro_ids, k)
        return {i: self.rng.choice(self.region_labels) for i in chosen}


class ClaudeAdvisor:
    """Real advisor backed by the Anthropic Messages API (text-only)."""

    def __init__(self, summary_text, macro_ids, region_labels,
                 model="claude-sonnet-4-6", max_tokens=2500, max_retries=2,
                 api_key=None):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.summary = summary_text
        self.macro_ids = set(macro_ids)
        self.n = len(macro_ids)
        self.region_labels = list(region_labels)
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.calls = 0
        self.in_tokens = 0
        self.out_tokens = 0
        self.cache_read_tokens = 0

    def _build_user_msg(self, history_text, cur_hpwl, best_hpwl, prev_regions, diagnostics):
        return (
            "=== PLACEMENT HISTORY ===\n"
            f"{history_text}\n\n"
            "=== CURRENT REGION ASSIGNMENT ===\n"
            f"{_fmt_regions(prev_regions)}\n\n"
            "=== LONG CONNECTIONS RIGHT NOW (strongly-linked macros far apart) ===\n"
            f"{diagnostics or '(no diagnostics)'}\n\n"
            "=== YOUR TASK ===\n"
            f"Current HPWL: {cur_hpwl:.4e} (lower is better). Best so far: {best_hpwl:.4e}.\n"
            "Pick the macros above that are far from their strong partners and put "
            "each such pair/group into the SAME region so they are pulled together. "
            "Assign a region ONLY to macros you want to move; OMIT every other macro "
            "(it stays free and optimally placed). It is fine to move just a handful. "
            "Your JSON REPLACES the current assignment, so re-list any currently-"
            "assigned macro you still want held in place. "
            "If nothing looks worth changing, return an empty object {}.\n"
            "Output ONLY a JSON object mapping the chosen macros to a region, e.g. "
            "{\"M5\":\"center\",\"M12\":\"center\"}.\n"
            f"Valid regions: {', '.join(self.region_labels)}."
        )

    def suggest(self, history_text, cur_hpwl, best_hpwl, prev_regions=None, diagnostics=""):
        user_msg = self._build_user_msg(history_text, cur_hpwl, best_hpwl,
                                        prev_regions or {}, diagnostics)
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=[
                        {"type": "text", "text": SYSTEM_INSTRUCTIONS},
                        {"type": "text", "text": self.summary,
                         "cache_control": {"type": "ephemeral"}},
                    ],
                    messages=[{"role": "user", "content": user_msg}],
                )
                self.calls += 1
                u = resp.usage
                self.in_tokens += getattr(u, "input_tokens", 0)
                self.out_tokens += getattr(u, "output_tokens", 0)
                self.cache_read_tokens += getattr(u, "cache_read_input_tokens", 0) or 0
                text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
                parsed = _parse_region_json(text, self.macro_ids, set(self.region_labels))
                if parsed is not None:                 # dict (maybe empty) = valid answer
                    return parsed
                print(f"  [llm] attempt {attempt + 1}: no JSON object found; retrying")
            except Exception as e:
                print(f"  [llm] attempt {attempt + 1} error: {e}")
                time.sleep(1.5 * (attempt + 1))
        print("  [llm] giving up this iteration -> caller reuses previous regions")
        return None
