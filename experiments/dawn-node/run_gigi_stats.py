#!/usr/bin/env python3
"""Run Gigi's WebGPU unit tests with the patched comparison helpers and summarize
how far each readback is from its gold file. Usage: run_gigi_stats.py <root> <icd> <out.json>
"""
import json
import os
import pathlib
import subprocess
import sys
import time

root = pathlib.Path(sys.argv[1])
icd = sys.argv[2]
# Tolerances wide open: nothing aborts early, so every comparison in a test reports.
env = dict(os.environ, VK_ICD_FILENAMES=icd, VK_DRIVER_FILES=icd, GIGI_TOL="255", GIGI_BIN_FLOAT_TOL="1e30")
skip = set(sys.argv[4].split(",")) if len(sys.argv) > 4 else set()

rows = []
for index in sorted(root.glob("UnitTests/*/*/index.js")):
    name = f"{index.parent.parent.name}/{index.parent.name}"
    if name in skip:
        rows.append({"test": name, "status": "skipped"})
        print(f"SKIP           {name}")
        continue
    t0 = time.time()
    try:
        p = subprocess.run(["node", "."], cwd=index.parent, env=env, capture_output=True, text=True, timeout=120)
        code, out = p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        code, out = "timeout", ""
    secs = round(time.time() - t0, 2)
    stats = [json.loads(l[len("DIFFSTAT "):]) for l in out.splitlines() if l.startswith("DIFFSTAT ")]
    png = [s for s in stats if s["kind"] == "png"]
    binf = [s for s in stats if s["kind"] == "bin"]
    row = {
        "test": name, "exit": code, "seconds": secs, "comparisons": len(stats),
        "png_max_abs": max((s["maxAbs"] for s in png), default=None),
        "png_differing_frac": max((s["differing"] / s["bytes"] for s in png), default=None),
        "bin_differing_bytes": max((s["differing"] for s in binf), default=None),
        "bin_float_max_abs": max((s["asFloat32"]["maxAbs"] for s in binf), default=None),
        "error": next((l.strip()[:200] for l in out.splitlines() if l.startswith("Error:") or "Could not" in l or "error occurred" in l), ""),
    }
    rows.append(row)
    verdict = "exact" if (row["png_max_abs"] in (0, None) and row["bin_differing_bytes"] in (0, None) and code == 0) else "differs"
    if code != 0:
        verdict = "ERROR"
    print(f"{verdict:8s} {secs:5.2f}s {name:52s} png max|d|={row['png_max_abs']} frac={None if row['png_differing_frac'] is None else round(row['png_differing_frac'], 4)} "
          f"bin diff bytes={row['bin_differing_bytes']} float max|d|={row['bin_float_max_abs']}  {row['error']}", flush=True)

pathlib.Path(sys.argv[3]).write_text(json.dumps(rows, indent=1))
