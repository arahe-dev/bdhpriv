"""Analyze benchmark rows (matrix json or container log files).

Usage:
  python opt/analyze_log.py results/local_matrix.json
  python opt/analyze_log.py /var/log/matrix_bscale.log   (in container)

Prints per-cell medians, replicate spreads, and ref-vs-opt speedups.
"""

import collections
import json
import sys


def load(path):
    rows = []
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "variant" in r:
                rows.append(r)
    return rows


def main():
    rows = load(sys.argv[1])
    ok = [r for r in rows if r.get("status") == "OK"]
    print(f"rows={len(rows)} ok={len(ok)}")
    by = collections.defaultdict(list)
    for r in ok:
        by[(r.get("shape"), r["variant"], r.get("microbatch"),
            bool(r.get("compiled")), r.get("compile_mode", "default"),
            r.get("packed", "single"))].append(r)
    print(f"\n{'cell':58s} {'n':>2s} {'med':>8s} {'spread':>7s} {'tok/s':>9s} {'mem':>7s}")
    cells = {}
    for k in sorted(by):
        v = [r["median_ms"] for r in by[k]]
        t = [r["input_tok_s"] for r in by[k]]
        m = [r.get("peak_alloc_GiB", 0) for r in by[k]]
        a, b = min(v), max(v)
        cells[k] = sum(v) / len(v)
        print(f"{str(k):58s} {len(v):2d} {sum(v)/len(v):8.1f} {(b-a)/a*100:6.1f}% "
              f"{sum(t)/len(t):9.0f} {sum(m)/len(m):7.2f}")
    print("\nspeedups (opt/ref, matched shape+mb+compiled+packed):")
    shapes = sorted({k[0] for k in cells})
    for s in shapes:
        refs = {k: v for k, v in cells.items()
                if k[0] == s and k[1] == "ref_ckpt"}
        for k, v in sorted(cells.items()):
            if k[0] != s or k[1] == "ref_ckpt":
                continue
            rk = (s, "ref_ckpt", k[2], k[3], k[4], k[5])
            if rk in refs:
                print(f"  {s} {k[1]} mb={k[2]} comp={int(k[3])} pack={k[5]}: "
                      f"{refs[rk]/v:.2f}x ({refs[rk]:.0f}/{v:.0f} ms)")


if __name__ == "__main__":
    main()
