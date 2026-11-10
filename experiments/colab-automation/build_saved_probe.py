#!/usr/bin/env python3
"""Build saved_probe.ipynb: a notebook whose cells already carry SAVED outputs.

Opening it in Colab without a runtime (and without signing in) shows what Colab does
with stored outputs: are they rendered, does their JavaScript run again, does WebGPU
work inside the real output iframe, can one cell's output reach another's.

The browser snippets are shared with ../colab-probe/js.
"""
import base64
import json
import pathlib
import struct
import zlib

HERE = pathlib.Path(__file__).parent
JS = HERE.parent / "colab-probe" / "js"


def read(name):
    return (JS / name).read_text()


def png(width, height, rgb):
    """A solid-colour PNG, standard library only."""
    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))
    row = b"\x00" + bytes(rgb) * width
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(row * height)) + chunk(b"IEND", b""))


def lines(text):
    return text.splitlines(keepends=True)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text.strip("\n"))}


def code(source, outputs, n):
    return {"cell_type": "code", "metadata": {}, "execution_count": n, "source": lines(source), "outputs": outputs}


def display(data):
    return {"output_type": "display_data", "metadata": {}, "data": data}


# One output that runs the frame probe, then looks for "frame A" in a sibling cell output.
FACTS_HTML = """<pre id="gigi-probe-facts" style="font: 12px monospace; white-space: pre-wrap">facts: running…</pre>
<script type="module">
const label = document.getElementById('gigi-probe-facts');
const result = { startedAt: Date.now() };
self.gigiProbeResult = result;
try {
  result.frame = await __FRAME_JS__;
} catch (e) { result.frameError = String(e); }
// Frame A may load after this one; retry for a few seconds.
for (let attempt = 0; attempt < 10; attempt++) {
  try {
    result.sibling = await __SIBLING_JS__;
  } catch (e) { result.siblingError = String(e); }
  if (result.sibling && (result.sibling.direct || result.sibling.broadcast)) break;
  await new Promise((r) => setTimeout(r, 700));
}
result.done = true;
label.textContent = 'facts: ' + JSON.stringify(result, null, 1);
</script>"""
FACTS_HTML = FACTS_HTML.replace("__FRAME_JS__", read("frame.js").rstrip()).replace("__SIBLING_JS__", read("sibling.js").rstrip())

JS_OUTPUT = """self.gigiJsOutputRan = true;
const gigiNote = document.createElement('div');
gigiNote.id = 'gigi-js-output';
gigiNote.style.font = '13px monospace';
gigiNote.textContent = 'application/javascript output ran; kernel access: ' +
  (((self.google || {}).colab || {}).kernel ? google.colab.kernel.accessAllowed : 'no google.colab');
(document.querySelector('#output-area') || document.body).appendChild(gigiNote);
"""

cells = [
    md("# Saved-output probe\n\nEvery output below was stored in the notebook file. Nothing here was executed in this session."),
    code("# frame A: a marker for other cells to find\n", [display({"text/html": lines(read("frame_a.html"))})], 1),
    code("# facts: what this output frame can do, and whether it can reach frame A\n", [display({"text/html": lines(FACTS_HTML)})], 2),
    code("# persistence: a self-contained WebGPU animation\n",
         [display({"text/html": lines(read("persist.html").replace("__EMITTED_AT__", "__EMITTED_AT__"))})], 3),
    code("# mime bundle: HTML and PNG alternatives of the same output\n",
         [display({
             "text/html": ['<div id="gigi-bundle-html" style="font: 13px monospace">mime bundle: the text/html branch was rendered</div>'],
             "image/png": base64.b64encode(png(96, 32, (200, 40, 40))).decode(),
             "text/plain": ["<mime bundle: text/plain branch>"],
         })], 4),
    code("# a display(Javascript(...)) output\n", [display({"application/javascript": lines(JS_OUTPUT)})], 5),
    code("print('plain stream output')\n", [{"output_type": "stream", "name": "stdout", "text": ["plain stream output\n"]}], 6),
]

notebook = {
    "nbformat": 4, "nbformat_minor": 0,
    "metadata": {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"},
                 "language_info": {"name": "python"}},
    "cells": cells,
}
out = HERE / "saved_probe.ipynb"
out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(f"wrote {out} ({out.stat().st_size} bytes)")
