"""Phase 4 (text-only) / Phase 5 (+ image): the region advisor.

Two interchangeable advisors with the same ``suggest`` signature:

  * DummyAdvisor  -- no API calls; returns random valid regions. Use it to test
    the whole loop (ordering, history, early-stop) for FREE before spending
    tokens.
  * ClaudeAdvisor -- one Anthropic API call per iteration. Sends the cached chip
    summary (system) + the dynamic history/HPWL (user), parses the JSON region
    map, retries on bad JSON / API errors, and returns None on give-up so the
    caller can fall back to the previous regions.

suggest(history_text, cur_hpwl, best_hpwl, prev_regions) -> {index: label} | None
Returned dicts are keyed by integer placement index (M5 -> 5).
"""

import json
import os
import random
import re
import time


SYSTEM_INSTRUCTIONS = (
    "You are an expert chip floorplanning assistant. You assign each hard macro "
    "to one region of a 3x3 grid so that strongly connected macros end up in the "
    "same or neighbouring regions, reducing total wirelength (HPWL). You always "
    "reply with a single JSON object mapping every macro id to a region label."
)


def _parse_region_json(text, valid_indices, valid_labels):
    """Extract {index: label} from an LLM reply. Returns {} if nothing usable."""
    # Grab the last {...} block (models sometimes reason before the JSON).
    matches = re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)
    for blob in reversed(matches):
        try:
            raw = json.loads(blob)
        except json.JSONDecodeError:
            continue
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
    return {}


class DummyAdvisor:
    """Free, offline advisor: random valid regions. For plumbing tests only."""

    def __init__(self, macro_ids, region_labels, seed=0):
        self.macro_ids = list(macro_ids)
        self.region_labels = list(region_labels)
        self.rng = random.Random(seed)
        self.calls = 0
        self.in_tokens = 0
        self.out_tokens = 0

    def suggest(self, history_text, cur_hpwl, best_hpwl, prev_regions=None):
        self.calls += 1
        return {i: self.rng.choice(self.region_labels) for i in self.macro_ids}


class ClaudeAdvisor:
    """Real advisor backed by the Anthropic Messages API (text-only)."""

    def __init__(self, summary_text, macro_ids, region_labels,
                 model="claude-sonnet-4-6", max_tokens=2000, max_retries=2,
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
        # cost / usage tracking
        self.calls = 0
        self.in_tokens = 0
        self.out_tokens = 0

    def _build_user_msg(self, history_text, cur_hpwl, best_hpwl):
        return (
            "=== PLACEMENT HISTORY ===\n"
            f"{history_text}\n\n"
            "=== YOUR TASK ===\n"
            f"Current HPWL: {cur_hpwl:.4e} (lower is better). "
            f"Best so far: {best_hpwl:.4e}.\n"
            "Reason briefly about which strongly-connected macros are likely far "
            "apart and how to regroup them. Then output ONLY a JSON object mapping "
            f"every macro id to a region, e.g. {{\"M0\":\"center\",\"M1\":\"top-left\"}}.\n"
            f"Valid regions: {', '.join(self.region_labels)}.\n"
            f"Assign all {self.n} macros M0..M{self.n - 1}."
        )

    def suggest(self, history_text, cur_hpwl, best_hpwl, prev_regions=None):
        user_msg = self._build_user_msg(history_text, cur_hpwl, best_hpwl)
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
                self.in_tokens += getattr(resp.usage, "input_tokens", 0)
                self.out_tokens += getattr(resp.usage, "output_tokens", 0)
                text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
                parsed = _parse_region_json(text, self.macro_ids, set(self.region_labels))
                if parsed:
                    # Fill any macros the model omitted with their previous region.
                    if prev_regions:
                        merged = dict(prev_regions)
                        merged.update(parsed)
                        return merged
                    return parsed
                print(f"  [llm] attempt {attempt + 1}: no valid JSON regions parsed; retrying")
            except Exception as e:                       # API / network / SDK error
                print(f"  [llm] attempt {attempt + 1} error: {e}")
                time.sleep(1.5 * (attempt + 1))
        print("  [llm] giving up this iteration -> caller falls back to previous regions")
        return None
