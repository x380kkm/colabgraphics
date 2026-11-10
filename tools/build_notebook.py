#!/usr/bin/env python3
"""Build an .ipynb with saved outputs from a "percent" script, without Jupyter.

The script is split at `# %%` lines; `# %% [markdown]` starts a text cell (its lines are
comments). Code cells run in order in one namespace. What a cell prints becomes a stream
output; if its last statement is an expression with a rich representation (for example
`technique.show()`), that becomes a display output. Lines that start with `%` or `!`
are notebook-only commands: they are kept in the notebook but skipped here.

    PYTHONPATH=src python3 tools/build_notebook.py notebooks/src/01_first_steps.py notebooks/01_first_steps.ipynb
"""
import ast
import contextlib
import io
import json
import pathlib
import sys


def split_cells(text):
    cells, kind, lines = [], None, []
    for line in text.splitlines():
        if line.startswith("# %%"):
            if kind is not None:
                cells.append((kind, lines))
            kind, lines = ("markdown" if "[markdown]" in line else "code"), []
        elif kind is not None:
            lines.append(line)
    if kind is not None:
        cells.append((kind, lines))
    return cells


def run_cell(source, namespace):
    """Execute one cell; return its notebook outputs."""
    runnable = "\n".join("" if l.lstrip().startswith(("%", "!")) else l for l in source.splitlines())
    tree = ast.parse(runnable)
    last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    stdout, value = io.StringIO(), None
    with contextlib.redirect_stdout(stdout):
        exec(compile(tree, "<cell>", "exec"), namespace)
        if last is not None:
            value = eval(compile(ast.Expression(last.value), "<cell>", "eval"), namespace)
    outputs = []
    if stdout.getvalue():
        outputs.append({"output_type": "stream", "name": "stdout", "text": stdout.getvalue().splitlines(keepends=True)})
    if value is not None:
        bundle = value._repr_mimebundle_() if hasattr(value, "_repr_mimebundle_") else {"text/plain": repr(value)}
        outputs.append({"output_type": "display_data", "metadata": {},
                        "data": {k: (v.splitlines(keepends=True) if isinstance(v, str) else v) for k, v in bundle.items()}})
    return outputs


def build(src, dst):
    namespace, cells, count = {"__name__": "__notebook__"}, [], 0
    for kind, lines in split_cells(pathlib.Path(src).read_text()):
        while lines and not lines[-1].strip():
            lines.pop()
        while lines and not lines[0].strip():
            lines.pop(0)
        if kind == "markdown":
            text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in lines)
            cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})
        else:
            source = "\n".join(lines)
            count += 1
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": count,
                          "source": source.splitlines(keepends=True), "outputs": run_cell(source, namespace)})
    notebook = {"nbformat": 4, "nbformat_minor": 0, "cells": cells,
                "metadata": {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"},
                             "language_info": {"name": "python"}}}
    pathlib.Path(dst).write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {dst}: {len(cells)} cells, {pathlib.Path(dst).stat().st_size // 1024} KB")


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
