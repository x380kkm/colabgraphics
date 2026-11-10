#!/usr/bin/env python3
"""Dry-run the notebook's Python cells outside Colab.

google.colab and IPython are replaced by stubs, so this only checks the Python glue
(names, control flow, report shape). The browser side is rehearsed by test_local.cjs.
The Node + Dawn cell is compiled but not executed here: it downloads and runs an npm
package, which is meant to happen in the Colab VM.
"""
import json
import pathlib
import sys
import types

HERE = pathlib.Path(__file__).parent
nb = json.loads((HERE / "colab_probe.ipynb").read_text())

# ---- stubs -----------------------------------------------------------------
calls = {"eval_js": 0, "display": 0, "targets": [], "callbacks": []}


def eval_js(script, ignore_result=False, timeout_sec=None):
    calls["eval_js"] += 1
    assert timeout_sec, "every eval_js call must carry a timeout"
    if script == "1":
        return 1
    if script.endswith(").length"):          # kernel -> browser payload test
        return len(script) - len('("') - len('").length')
    if ".repeat(" in script:                 # browser -> kernel payload test
        return "x" * int(script.split("(")[1].rstrip(")"))
    return {"stub": True}                    # frame / sibling / http snippets


output = types.SimpleNamespace(
    eval_js=eval_js,
    register_callback=lambda name, fn: calls["callbacks"].append(name),
)
colab = types.ModuleType("google.colab")
colab.output = output
google = types.ModuleType("google")
google.colab = colab
sys.modules.update({"google": google, "google.colab": colab, "google.colab.output": output})

comm_manager = types.SimpleNamespace(register_target=lambda name, fn: calls["targets"].append((name, fn)))
shell = types.SimpleNamespace(kernel=types.SimpleNamespace(comm_manager=comm_manager))
ipython = types.ModuleType("IPython")
ipython.get_ipython = lambda: shell
display_mod = types.ModuleType("IPython.display")
display_mod.HTML = lambda html: html


def display(obj):
    calls["display"] += 1


display_mod.display = display
ipython.display = display_mod
sys.modules.update({"IPython": ipython, "IPython.display": display_mod})

# ---- run -------------------------------------------------------------------
ns = {"__name__": "__probe__"}
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    compiled = compile(src, f"<cell {i}>", "exec")
    if "npm install" in src:
        print(f"cell {i}: compiled only (Node + Dawn cell)")
        continue
    print(f"--- cell {i} ---")
    exec(compiled, ns)

# ---- the comm handler, driven by hand ---------------------------------------
name, on_open = calls["targets"][0]
sent = []
handlers = []
comm = types.SimpleNamespace(on_msg=handlers.append, send=lambda data, meta, buffers: sent.append((data, [len(b) for b in buffers])))
on_open(comm, {"content": {"data": {"hello": 1}}, "buffers": [memoryview(bytes(1000))]})
handlers[0]({"content": {"data": {"echo_bytes": 5000}}, "buffers": [memoryview(bytes(5000))]})
handlers[0]({"content": {"data": {"final": {"comm": "summary"}}}, "buffers": []})

report = ns["REPORT"]
assert name == "gigi_probe_comm" and calls["callbacks"] == ["gigi_probe_callback"]
assert sent == [({"echo": 5000}, [5000])], sent
assert report["comm"] == {"comm": "summary"}
assert ns["COMM_LOG"] == [{"event": "open", "buffers": [1000]}, {"event": "msg", "buffers": [5000]}]
assert report["kernel"]["python"] and "error" not in report["kernel"], report["kernel"]
assert [s["ok"] for s in report["eval_js"]["kernel_to_browser"]] == [True] * 4, report["eval_js"]
assert [s["ok"] for s in report["eval_js"]["browser_to_kernel"]] == [True] * 4
assert "eval_js_in_progress" not in report
assert report["kernel_http"]["port"] > 0 and report["kernel_http"]["browser"] == {"stub": True}
assert calls["display"] == 3, calls["display"]   # frame A, comm probe, persistence
assert json.loads(pathlib.Path(ns["REPORT_PATH"]).read_text())["probe_version"] == 1
print("\nall Python-side checks passed;", calls["eval_js"], "eval_js calls,", calls["display"], "displays")
