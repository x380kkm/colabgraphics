#!/usr/bin/env python3
"""Run Gigi's generated WebGPU unit tests under Node + Dawn on this Linux machine.

Each test directory has an index.js that Gigi's own harness runs with `node .`;
exit code 0 means pass. Usage: run_gigi_tests.py <WebGPU test root> <vulkan icd json>
"""
import json
import os
import pathlib
import subprocess
import sys
import time

root = pathlib.Path(sys.argv[1])
icd = sys.argv[2]
env = dict(os.environ, VK_ICD_FILENAMES=icd, VK_DRIVER_FILES=icd)

results = []
for index in sorted(root.glob("UnitTests/*/*/index.js")):
    name = f"{index.parent.parent.name}/{index.parent.name}"
    t0 = time.time()
    try:
        p = subprocess.run(["node", "."], cwd=index.parent, env=env, capture_output=True, text=True, timeout=300)
        code, out = p.returncode, (p.stdout + p.stderr)
    except subprocess.TimeoutExpired as e:
        code, out = "timeout", str(e)
    secs = round(time.time() - t0, 2)
    # First line that explains a failure, if any.
    reason = ""
    for line in out.splitlines():
        if line.startswith("Error:") or "error occurred" in line or "Could not" in line:
            reason = line.strip()[:230]
            break
    if code != 0 and not reason:
        reason = out.strip().splitlines()[-1][:230] if out.strip() else ""
    results.append({"test": name, "exit": code, "seconds": secs, "reason": reason, "tail": out[-1500:]})
    print(f"{'PASS' if code == 0 else 'FAIL'}  {secs:6.2f}s  {name}  {reason}", flush=True)

passed = sum(r["exit"] == 0 for r in results)
print(f"\n{passed}/{len(results)} passed, total {sum(r['seconds'] for r in results):.1f}s")
out_path = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else None
if out_path:
    out_path.write_text(json.dumps(results, indent=1))
