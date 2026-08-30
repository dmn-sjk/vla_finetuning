"""Render the LIBERO success-rate tables straight from the results JSON.

Hand-copying three success rates and thirty per-task numbers into two model cards is a
reliable way to publish a typo, so the cards get their tables from the measurement files.
`--check` re-derives every rate from the raw per-episode records and fails if a stored
summary disagrees, which catches a truncated or partially-gathered run.

    python scripts/render_results.py results/libero_spatial_*.json
"""

import argparse
import json
from pathlib import Path

LABELS = {
    "base": "base `lerobot/pi0` (zero-shot)",
    "lora_best": "LoRA `adapter_best`",
    "full_best": "full FT `model_best`",
}
ORDER = ["base", "lora_best", "full_best"]


def load(paths):
    out = {}
    for p in paths:
        d = json.loads(Path(p).read_text())
        key = Path(p).stem.replace("libero_spatial_", "")
        eps = d["episodes"]
        n = len(eps)
        rate = sum(e["success"] for e in eps) / n
        if abs(rate - d["success_rate"]) > 1e-9:
            raise SystemExit(f"{p}: stored rate {d['success_rate']} != recomputed {rate}")
        if n != d["n_episodes"]:
            raise SystemExit(f"{p}: stored n {d['n_episodes']} != {n} records")
        d["_rate"], d["_n"], d["_se"] = rate, n, (rate * (1 - rate) / n) ** 0.5
        out[key] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--check", action="store_true", help="validate only, print nothing else")
    args = ap.parse_args()
    R = load(args.results)
    keys = [k for k in ORDER if k in R] + [k for k in R if k not in ORDER]
    if args.check:
        print(f"OK: {len(R)} result files consistent")
        return

    first = R[keys[0]]
    print(f"LIBERO-{first['suite'].replace('libero_', '')}, "
          f"{first['_n']} episodes ({first['_n'] // 10} trials x 10 tasks), "
          f"n_action_steps={first['n_action_steps']}, {first['max_steps']}-step cap.\n")
    print("| checkpoint | success rate | successes |")
    print("| --- | --- | --- |")
    for k in keys:
        d = R[k]
        print(f"| {LABELS.get(k, k)} | **{d['_rate'] * 100:.1f}% ± {d['_se'] * 100:.1f}** "
              f"| {round(d['_rate'] * d['_n'])}/{d['_n']} |")

    print("\n### Per task\n")
    tids = sorted({t for d in R.values() for t in d["per_task"]}, key=int)
    print("| task | " + " | ".join(LABELS.get(k, k) for k in keys) + " |")
    print("| --- |" + " --- |" * len(keys))
    for t in tids:
        cells = " | ".join(f"{R[k]['per_task'].get(t, float('nan')) * 100:.0f}%" for k in keys)
        print(f"| {t} | {cells} |")


if __name__ == "__main__":
    main()
