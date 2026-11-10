"""Turn a compiled program into something a notebook (or a browser) can show.

The output is one self-contained HTML fragment: the runtime, the viewer and the program
itself, all inline. Stored as a cell output it keeps working when the notebook is opened
again later, with no kernel attached.
"""
from __future__ import annotations

import json
import re
import uuid
from importlib import resources

CSS = """
.cg { display: flex; flex-wrap: wrap; gap: 12px 20px; align-items: flex-start; font: 13px/1.5 system-ui, sans-serif; color: inherit; }
.cg-stage canvas, .cg-stage img { display: block; max-width: 100%; height: auto; image-rendering: pixelated; background: #000; }
.cg-side { display: flex; flex-direction: column; gap: 6px; min-width: 220px; }
.cg-row { display: flex; align-items: center; gap: 8px; }
.cg-name { min-width: 72px; font-family: ui-monospace, monospace; }
.cg-row input[type=range] { flex: 1; min-width: 100px; }
.cg-row input[type=number] { width: 72px; }
.cg-readout { min-width: 48px; font-family: ui-monospace, monospace; text-align: right; }
.cg-status { opacity: .65; font-family: ui-monospace, monospace; margin-top: 4px; }
.cg-graph { flex-basis: 100%; display: flex; flex-wrap: wrap; gap: 6px 18px; font-family: ui-monospace, monospace; font-size: 12px; }
.cg-node { display: flex; align-items: center; gap: 6px; }
.cg-io { display: flex; gap: 4px; }
.cg-arrow { opacity: .5; }
.cg-pass { padding: 2px 8px; border: 1px solid currentColor; border-radius: 4px; }
.cg-chip { font: inherit; padding: 2px 8px; border: 1px solid transparent; border-radius: 999px; cursor: pointer;
           color: inherit; background: color-mix(in srgb, currentColor 12%, transparent); }
.cg-chip.cg-current { border-color: currentColor; }
.cg-probe { font-family: ui-monospace, monospace; font-size: 12px; opacity: .8; min-height: 1.5em; }
.cg-button { font: inherit; padding: 2px 10px; border: 1px solid currentColor; border-radius: 4px; background: transparent; color: inherit; cursor: pointer; }
.cg-button:disabled { opacity: .35; cursor: default; }
.cg-warning { color: #b45309; white-space: pre-wrap; font: 12px/1.5 ui-monospace, monospace; }
.cg-errors { flex-basis: 100%; margin: 0; padding: 8px 10px; white-space: pre-wrap; color: #b91c1c;
             border-left: 3px solid #b91c1c; font: 12px/1.5 ui-monospace, monospace; }
"""


def _js(name: str) -> str:
    source = resources.files("colabgraphics").joinpath("js", name).read_text()
    return re.sub(r"^export ", "", source, flags=re.M)  # inlined into a single module


def snapshot_html(program, view: str | None = None, fallback: str | None = None) -> str:
    """One HTML fragment that runs `program`. `fallback` is an image URL shown where WebGPU is missing."""
    uid = "cg-" + uuid.uuid4().hex[:10]
    # The data sits inside the module script itself: notebook front ends re-create script
    # elements when they render an output, and a separate data element is not reliably found.
    payload = json.dumps({"program": program.json, "options": {"view": view, "fallback": fallback, "warnings": program.warnings}})
    payload = payload.replace("</", "<\\/")
    return (
        f"<style>{CSS}</style>\n"
        f'<div id="{uid}"></div>\n'
        f'<script type="module">\n{_js("runtime.js")}\n{_js("viewer.js")}\n'
        f"const payload = {payload};\n"
        f'mountViewer(document.getElementById("{uid}"), payload.program, payload.options);\n'
        f"</script>\n"
    )


def standalone_html(program, title: str | None = None, **options) -> str:
    """A complete page, for opening directly in a browser."""
    return (f'<!doctype html>\n<html><head><meta charset="utf-8"><title>{title or program.json["name"]}</title>'
            f'<meta name="viewport" content="width=device-width, initial-scale=1"></head>\n'
            f'<body style="margin: 16px">\n{snapshot_html(program, **options)}</body></html>\n')


class Snapshot:
    """What `show()` returns: displays itself in a notebook, and keeps the HTML for other uses."""

    def __init__(self, program, html: str):
        self.program = program
        self.html = html

    def _repr_mimebundle_(self, include=None, exclude=None):
        return {"text/html": self.html, "text/plain": f"<colabgraphics technique {self.program.json['name']!r}>"}

    def save(self, path):
        with open(path, "w") as f:
            f.write(standalone_html(self.program))
        return path


def show(program, view: str | None = None, fallback: str | None = None) -> Snapshot:
    return Snapshot(program, snapshot_html(program, view=view, fallback=fallback))
