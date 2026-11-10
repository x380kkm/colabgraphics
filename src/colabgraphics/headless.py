"""Run a compiled program without a browser and get textures back as numpy arrays.

The same `js/runtime.js` that draws in the reader's browser is executed here under
Node with Dawn, Google's WebGPU implementation. This is what makes results checkable
from Python: `technique.read(texture)` returns pixels you can assert on.

What it needs, and where it looks:

- Node: `node` on PATH, or the `nodejs-wheel-binaries` pip package.
- Dawn for Node: the `webgpu` npm package, installed on first use into a cache folder.
- A graphics driver. On a machine without a GPU either of these works:
    * a software Vulkan driver (SwiftShader or Mesa lavapipe), found through the usual
      Vulkan loader settings, or
    * Mesa's software OpenGL ES through EGL. This gives WebGPU "compatibility mode",
      which has fewer features.
"""
from __future__ import annotations

import glob
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
from importlib import resources

import numpy as np

CACHE = pathlib.Path(os.environ.get("COLABGRAPHICS_CACHE", pathlib.Path.home() / ".cache" / "colabgraphics"))

DTYPES = {
    "r8unorm": np.uint8, "rg8unorm": np.uint8, "rgba8unorm": np.uint8, "rgba8unorm-srgb": np.uint8,
    "rgba8snorm": np.int8, "rgba8uint": np.uint8, "rgba16float": np.float16,
    "r32float": np.float32, "rg32float": np.float32, "rgba32float": np.float32,
    "r32uint": np.uint32, "depth32float": np.float32,
}


class HeadlessError(Exception):
    """The program could not be run headless. The message says which piece is missing or what failed."""


def _node() -> str:
    node = os.environ.get("COLABGRAPHICS_NODE") or shutil.which("node")
    if node:
        return node
    try:  # the pip-installable Node
        import nodejs_wheel  # type: ignore

        return str(pathlib.Path(nodejs_wheel.__file__).parent / "bin" / "node")
    except ImportError:
        raise HeadlessError("Node.js was not found. Install it, or run: pip install nodejs-wheel-binaries") from None


def _webgpu_module(node: str) -> str:
    module = CACHE / "node_modules" / "webgpu" / "index.js"
    if not module.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        npm = shutil.which("npm") or str(pathlib.Path(node).with_name("npm"))
        print("colabgraphics: installing Dawn for Node (the 'webgpu' npm package), once…")
        (CACHE / "package.json").write_text('{"private": true, "type": "module"}\n')
        done = subprocess.run([npm, "install", "--no-audit", "--no-fund", "webgpu"], cwd=CACHE, capture_output=True, text=True)
        if done.returncode != 0 or not module.exists():
            raise HeadlessError("could not install the 'webgpu' npm package:\n" + (done.stderr or done.stdout)[-800:])
    return str(module)


def _driver() -> dict:
    """Pick a graphics driver: returns extra environment, Dawn flags and whether to use compatibility mode."""
    choice = os.environ.get("COLABGRAPHICS_DRIVER", "auto")
    icds = [os.environ.get("VK_ICD_FILENAMES") or os.environ.get("VK_DRIVER_FILES")]
    icds += glob.glob("/usr/share/vulkan/icd.d/*.json") + glob.glob("/etc/vulkan/icd.d/*.json")
    icds += glob.glob(str(CACHE / "swiftshader" / "*.json"))
    icds += glob.glob("/opt/pw-browsers/chromium-*/chrome-linux*/vk_swiftshader_icd.json")  # bundled with Playwright's Chromium
    icds = [i for i in icds if i]
    if choice in ("auto", "vulkan") and icds:
        return {"name": "vulkan", "env": {"VK_ICD_FILENAMES": icds[0], "VK_DRIVER_FILES": icds[0]}, "flags": [], "compatibility": False}
    if choice in ("auto", "opengles"):
        # Dawn loads "libEGL.so"; systems usually ship only the versioned name.
        found = sorted(glob.glob("/usr/lib/*/libEGL.so.1") + glob.glob("/usr/lib64/libEGL.so.1"))
        gles = sorted(glob.glob("/usr/lib/*/libGLESv2.so.2") + glob.glob("/usr/lib64/libGLESv2.so.2"))
        if found:
            shim = CACHE / "egl"
            shim.mkdir(parents=True, exist_ok=True)
            for target, name in [(found[0], "libEGL.so")] + ([(gles[0], "libGLESv2.so")] if gles else []):
                link = shim / name
                if not link.exists():
                    link.symlink_to(target)
            paths = f"{shim}:{os.environ.get('LD_LIBRARY_PATH', '')}".rstrip(":")
            return {"name": "opengles", "env": {"LD_LIBRARY_PATH": paths, "EGL_PLATFORM": "surfaceless"},
                    "flags": ["backend=opengles"], "compatibility": True}
    raise HeadlessError("no graphics driver found for headless rendering. Install a software Vulkan driver "
                        "(for example: apt-get install mesa-vulkan-drivers) or Mesa EGL (libegl1).")


def run(program, read: list[str] | None = None, frames: int = 1, vars: dict | None = None) -> dict[str, np.ndarray]:
    """Execute `program` for `frames` frames and return the named textures as arrays (height, width, channels)."""
    node = _node()
    module = _webgpu_module(node)
    driver = _driver()
    script = resources.files("colabgraphics").joinpath("js", "headless.mjs")
    with tempfile.TemporaryDirectory() as tmp:
        request = pathlib.Path(tmp, "request.json")
        request.write_text(json.dumps({
            "program": program.json, "frames": frames, "vars": vars or {}, "read": read or [], "outDir": tmp,
            "webgpuModule": module, "flags": driver["flags"], "compatibility": driver["compatibility"],
        }))
        done = subprocess.run([node, str(script), str(request)], capture_output=True, text=True, timeout=600,
                              env={**os.environ, **driver["env"]})
        line = next((l for l in done.stdout.splitlines() if l.startswith("RESULT ")), None)
        if line is None:
            raise HeadlessError("the headless runner produced no result:\n" + (done.stderr or done.stdout)[-1200:])
        result = json.loads(line[len("RESULT "):])
        if not result["ok"]:
            details = []
            for m in result.get("shaderErrors", []):
                p = next(p for p in program.json["passes"] if p["name"] == m["pass"])
                line_no = m["line"] - p["preambleLines"]
                where = f"line {line_no}" if line_no > 0 else "generated declarations"
                details.append(f'pass "{m["pass"]}", {where}: {m["message"]}')
            if not details:  # validation text only repeats a shader error, with whole-shader line numbers
                details += result.get("validation", [])
            raise HeadlessError("\n".join(details) or result.get("error", "unknown failure"))
        arrays = {}
        for name, meta in result["textures"].items():
            data = np.frombuffer(pathlib.Path(tmp, name + ".bin").read_bytes(), DTYPES[meta["format"]])
            arrays[name] = data.reshape(meta["height"], meta["width"], -1).copy()
        run.last = {"driver": driver["name"], "adapter": result.get("adapter"), "frames": result.get("frames")}
        return arrays
