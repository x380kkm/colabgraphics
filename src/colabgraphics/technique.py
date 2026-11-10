"""Describe a rendering technique and compile it into a program for the runtime.

A technique is a list of resources (textures), variables and passes. The author writes
only the algorithmic part of each shader; `compile()` works out everything else:

- the WGSL declarations for each resource a shader refers to by name,
- one uniform struct, `vars`, holding the technique's variables,
- which GPU usages every texture needs, and how many workgroups a compute pass dispatches.

The result is plain JSON that `js/runtime.js` executes, in a browser or headless.
"""
from __future__ import annotations

import base64
import math
import re
import zlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# format -> (bytes per pixel, WGSL texture type when read, usable as compute output)
FORMATS = {
    "r8unorm": (1, "texture_2d<f32>", False),
    "rg8unorm": (2, "texture_2d<f32>", False),
    "rgba8unorm": (4, "texture_2d<f32>", True),
    "rgba8unorm-srgb": (4, "texture_2d<f32>", False),
    "rgba8snorm": (4, "texture_2d<f32>", True),
    "rgba16float": (8, "texture_2d<f32>", True),
    "r32float": (4, "texture_2d<f32>", True),
    "rg32float": (8, "texture_2d<f32>", True),
    "rgba32float": (16, "texture_2d<f32>", True),
    "r32uint": (4, "texture_2d<u32>", True),
    "rgba8uint": (4, "texture_2d<u32>", True),
    "depth32float": (4, "texture_depth_2d", False),
}

# WGSL type -> (scalar, component count, alignment, size) under uniform layout rules
VAR_TYPES = {
    "f32": ("f32", 1, 4, 4),
    "i32": ("i32", 1, 4, 4),
    "u32": ("u32", 1, 4, 4),
    "vec2f": ("f32", 2, 8, 8),
    "vec3f": ("f32", 3, 16, 12),
    "vec4f": ("f32", 4, 16, 16),
}

SYS_STRUCT = """struct Sys {
  time: f32,          // seconds since the technique started
  frame: u32,         // frames rendered so far
  resolution: vec2f,  // size in pixels of what this pass writes to
  mouse: vec4f,       // xy: pointer position in pixels, z: 1 while pressed
}"""

RESERVED = {"vars", "sys", "linear_sampler", "nearest_sampler"}
IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class TechniqueError(Exception):
    """The technique description is inconsistent. The message says what and where."""


@dataclass(eq=False)
class Texture:
    name: str
    format: str
    size: tuple[int, int]
    data: str | None = None  # deflate-compressed pixels, base64

    def __repr__(self):
        return f"Texture({self.name!r}, {self.size[0]}x{self.size[1]}, {self.format})"


@dataclass(eq=False)
class Var:
    name: str
    type: str
    value: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    ui: str = "number"
    doc: str = ""


@dataclass(eq=False)
class Pass:
    kind: str
    name: str
    code: str
    reads: dict[str, Texture]
    options: dict[str, Any] = field(default_factory=dict)


def _strip_comments(code: str) -> str:
    code = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), code, flags=re.S)
    return re.sub(r"//[^\n]*", "", code)


def _uses(code: str, name: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9_.]){re.escape(name)}(?![A-Za-z0-9_])", code) is not None


def _pack(pixels: np.ndarray) -> str:
    return base64.b64encode(zlib.compress(np.ascontiguousarray(pixels).tobytes(), 6)).decode()


class Technique:
    def __init__(self, name: str = "technique"):
        self.name = name
        self.textures: list[Texture] = []
        self.vars: list[Var] = []
        self.passes: list[Pass] = []
        self.view: Texture | None = None

    # ---- resources -----------------------------------------------------------

    def _add_texture(self, tex: Texture) -> Texture:
        self._check_name(tex.name, "texture")
        if tex.format not in FORMATS:
            raise TechniqueError(f'texture "{tex.name}": unknown format "{tex.format}". Known formats: {", ".join(FORMATS)}')
        if any(t.name == tex.name for t in self.textures):
            raise TechniqueError(f'there is already a texture named "{tex.name}"')
        self.textures.append(tex)
        return tex

    def _check_name(self, name: str, what: str):
        if not IDENT.match(name):
            raise TechniqueError(f'{what} name "{name}" must be a plain identifier (letters, digits, underscore)')
        if name in RESERVED:
            raise TechniqueError(f'{what} name "{name}" is reserved: shaders use it for something the system provides')

    def image(self, name: str, source, srgb: bool = False) -> Texture:
        """A texture filled from an image file, a PIL image or a numpy array.

        uint8 data becomes rgba8unorm (or rgba8unorm-srgb); float data becomes rgba32float.
        """
        if isinstance(source, np.ndarray):
            pixels = source
        else:
            from PIL import Image  # only needed for files

            image = source if isinstance(source, Image.Image) else Image.open(source)
            pixels = np.asarray(image.convert("RGBA"))
        if pixels.ndim == 2:
            pixels = pixels[..., None]
        if pixels.ndim != 3 or pixels.shape[2] not in (1, 3, 4):
            raise TechniqueError(f'image "{name}": expected an array shaped (height, width[, 1|3|4]), got {pixels.shape}')
        h, w, c = pixels.shape
        is_float = np.issubdtype(pixels.dtype, np.floating)
        if not is_float and pixels.dtype != np.uint8:
            raise TechniqueError(f'image "{name}": expected uint8 or float data, got {pixels.dtype}')
        rgba = np.empty((h, w, 4), np.float32 if is_float else np.uint8)
        rgba[..., 3] = 1.0 if is_float else 255
        if c == 1:
            rgba[..., :3] = pixels
        else:
            rgba[..., :c] = pixels
        fmt = "rgba32float" if is_float else ("rgba8unorm-srgb" if srgb else "rgba8unorm")
        return self._add_texture(Texture(name, fmt, (w, h), _pack(rgba)))

    def texture(self, name: str, size: tuple[int, int] | None = None, like: Texture | None = None,
                format: str | None = None, scale: float = 1.0) -> Texture:
        """A texture the technique renders into. Give a size, or `like=` another texture."""
        if (size is None) == (like is None):
            raise TechniqueError(f'texture "{name}": give exactly one of size=(width, height) or like=<texture>')
        if like is not None:
            size = like.size
            format = format or ("rgba8unorm" if like.format == "rgba8unorm-srgb" else like.format)
        w, h = (max(1, round(s * scale)) for s in size)
        return self._add_texture(Texture(name, format or "rgba8unorm", (w, h)))

    def var(self, name: str, value, min=None, max=None, step=None, doc: str = "", color: bool = False) -> Var:
        """A value the reader can change while the technique runs. Shaders read it as vars.<name>."""
        self._check_name(name, "variable")
        if any(v.name == name for v in self.vars):
            raise TechniqueError(f'there is already a variable named "{name}"')
        if isinstance(value, bool):
            wgsl, ui = "u32", "checkbox"
        elif isinstance(value, int):
            wgsl, ui = "i32", "slider" if min is not None and max is not None else "number"
        elif isinstance(value, float):
            wgsl, ui = "f32", "slider" if min is not None and max is not None else "number"
        elif isinstance(value, (tuple, list)) and len(value) in (2, 3, 4):
            wgsl, ui = f"vec{len(value)}f", "color" if color else "number"
            value = [float(x) for x in value]
        else:
            raise TechniqueError(f'variable "{name}": the default must be a bool, int, float or a tuple of 2 to 4 numbers')
        var = Var(name, wgsl, value, min, max, step, ui, doc)
        self.vars.append(var)
        return var

    # ---- passes --------------------------------------------------------------

    def _add_pass(self, p: Pass) -> Pass:
        if any(q.name == p.name for q in self.passes):
            raise TechniqueError(f'there is already a pass named "{p.name}"')
        for wgsl_name, tex in p.reads.items():
            self._check_name(wgsl_name, f'pass "{p.name}": resource')
            self._owns(tex, p.name)
        self.passes.append(p)
        return p

    def _owns(self, tex, pass_name: str):
        if not isinstance(tex, Texture) or tex not in self.textures:
            raise TechniqueError(f'pass "{pass_name}": {tex!r} is not a texture of this technique')

    def compute(self, name: str, code: str, entry: str = "main", reads: dict | None = None,
                writes: dict | None = None, over: Texture | None = None) -> Pass:
        """Run a compute shader once per pixel of `over` (by default, the first texture it writes)."""
        reads, writes = dict(reads or {}), dict(writes or {})
        for wgsl_name, tex in writes.items():
            self._check_name(wgsl_name, f'pass "{name}": resource')
            self._owns(tex, name)
            if not FORMATS[tex.format][2]:
                raise TechniqueError(f'pass "{name}": a compute shader cannot write to "{tex.name}" '
                                     f'because its format {tex.format} is not a storage format')
            if tex in reads.values():
                raise TechniqueError(f'pass "{name}" both reads and writes "{tex.name}". '
                                     f'Write to a second texture instead and swap their roles in the next pass.')
        over = over or next(iter(writes.values()), None)
        if over is None:
            raise TechniqueError(f'pass "{name}": nothing to run over. Give writes={{...}} or over=<texture>.')
        self.view = next(iter(writes.values()), self.view)
        return self._add_pass(Pass("compute", name, code, reads, {"entry": entry, "writes": writes, "over": over}))

    def draw(self, name: str, code: str, targets: list[Texture], reads: dict | None = None,
             vertices: int = 3, instances: int = 1, vertex: str = "vs", fragment: str = "fs",
             clear: tuple | None = (0, 0, 0, 1), depth: Texture | bool | None = None,
             topology: str = "triangle-list", cull: str = "none") -> Pass:
        """Rasterize `vertices` vertices into `targets`. Geometry comes from the vertex shader."""
        if not targets:
            raise TechniqueError(f'pass "{name}": give at least one texture in targets=[...]')
        for tex in targets:
            self._owns(tex, name)
            if tex in (reads or {}).values():
                raise TechniqueError(f'pass "{name}" both reads and draws into "{tex.name}"')
        if depth is True:
            depth = self.texture(f"{name}_depth", like=targets[0], format="depth32float")
        elif depth:
            self._owns(depth, name)
            if depth.format != "depth32float":
                raise TechniqueError(f'pass "{name}": the depth texture must have format depth32float, "{depth.name}" is {depth.format}')
        self.view = targets[0]
        return self._add_pass(Pass("draw", name, code, dict(reads or {}), {
            "targets": list(targets), "vertices": vertices, "instances": instances, "vertex": vertex,
            "fragment": fragment, "clear": clear, "depth": depth or None, "topology": topology, "cull": cull}))

    # ---- compile -------------------------------------------------------------

    def _layout_vars(self):
        fields, offset, largest = [], 0, 4
        for v in self.vars:
            scalar, count, align, size = VAR_TYPES[v.type]
            offset = math.ceil(offset / align) * align
            fields.append({"name": v.name, "type": v.type, "scalar": scalar, "count": count, "offset": offset,
                           "value": int(v.value) if isinstance(v.value, bool) else v.value,
                           "min": v.min, "max": v.max, "step": v.step, "ui": v.ui, "doc": v.doc})
            offset += size
            largest = max(largest, align)
        return {"size": math.ceil(offset / largest) * largest if fields else 0, "fields": fields}

    def _preamble(self, p: Pass):
        """The declarations this pass's code needs, and the matching binding list."""
        code = _strip_comments(p.code)
        lines, bindings = [f'// ---- declared for you: pass "{p.name}" ----'], []

        def bind(entry, declaration):
            entry["binding"] = len(bindings)
            bindings.append(entry)
            lines.append(f"@group(0) @binding({entry['binding']}) {declaration}")

        used_vars = set(re.findall(r"(?<![A-Za-z0-9_.])vars\.([A-Za-z_][A-Za-z0-9_]*)", code))
        unknown = used_vars - {v.name for v in self.vars}
        if unknown:
            known = ", ".join(v.name for v in self.vars) or "none are defined"
            raise TechniqueError(f'pass "{p.name}": the shader uses vars.{sorted(unknown)[0]}, '
                                 f"which is not a variable of this technique (known: {known})")
        if used_vars:
            members = "\n".join(f"  {v.name}: {v.type}," for v in self.vars)
            lines.append(f"struct Vars {{\n{members}\n}}")
            bind({"kind": "vars"}, "var<uniform> vars: Vars;")
        if re.search(r"(?<![A-Za-z0-9_.])sys\.", code):
            lines.append(SYS_STRUCT)
            bind({"kind": "sys"}, "var<uniform> sys: Sys;")
        for wgsl_name, tex in p.reads.items():
            bind({"kind": "texture", "name": wgsl_name, "texture": tex.name}, f"var {wgsl_name}: {FORMATS[tex.format][1]};")
        for wgsl_name, tex in p.options.get("writes", {}).items():
            bind({"kind": "storage", "name": wgsl_name, "texture": tex.name},
                 f"var {wgsl_name}: texture_storage_2d<{tex.format}, write>;")
        for filt in ("linear", "nearest"):
            if _uses(code, f"{filt}_sampler"):
                bind({"kind": "sampler", "filter": filt}, f"var {filt}_sampler: sampler;")
        lines.append("// ---- your code ----")

        unused = [n for n in list(p.reads) + list(p.options.get("writes", {})) if not _uses(code, n)]
        return "\n".join(lines) + "\n", bindings, unused

    @staticmethod
    def _workgroup(p: Pass):
        code, entry = _strip_comments(p.code), p.options["entry"]
        for attrs, fn in re.findall(r"((?:@\w+(?:\s*\([^)]*\))?\s*)+)fn\s+(\w+)", code):
            if fn != entry:
                continue
            m = re.search(r"@workgroup_size\s*\(([^)]*)\)", attrs)
            if not m:
                break
            try:
                size = [int(x.strip().rstrip("u")) for x in m.group(1).split(",") if x.strip()]
            except ValueError:
                raise TechniqueError(f'pass "{p.name}": @workgroup_size({m.group(1)}) must use integer literals') from None
            return (size + [1, 1])[:3]
        raise TechniqueError(f'pass "{p.name}": no compute entry point "@compute @workgroup_size(...) fn {entry}" found in its shader')

    def compile(self) -> "Program":
        if not self.passes:
            raise TechniqueError("the technique has no passes yet: add one with compute(...) or draw(...)")
        passes, warnings = [], []
        for p in self.passes:
            preamble, bindings, unused = self._preamble(p)
            for n in unused:
                warnings.append(f'pass "{p.name}": "{n}" is connected but the shader never uses it')
            out = {"kind": p.kind, "name": p.name, "code": preamble + p.code, "bindings": bindings,
                   "preambleLines": preamble.count("\n")}
            if p.kind == "compute":
                out.update(entry=p.options["entry"],
                           dispatch={"texture": p.options["over"].name, "workgroup": self._workgroup(p)})
            else:
                o = p.options
                for stage in ("vertex", "fragment"):
                    if not re.search(rf"@{stage}[^{{]*?fn\s+{re.escape(o[stage])}\b", _strip_comments(p.code), flags=re.S):
                        raise TechniqueError(f'pass "{p.name}": no entry point "@{stage} fn {o[stage]}" found in its shader')
                out.update(vertex=o["vertex"], fragment=o["fragment"], vertexCount=o["vertices"],
                           instanceCount=o["instances"], topology=o["topology"], cull=o["cull"],
                           targets=[{"texture": t.name, "clear": list(o["clear"]) if o["clear"] else None} for t in o["targets"]],
                           depth={"texture": o["depth"].name} if o["depth"] else None)
            passes.append(out)
        return Program({
            "name": self.name,
            "textures": [{"name": t.name, "format": t.format, "size": list(t.size), **({"data": t.data} if t.data else {})}
                         for t in self.textures],
            "vars": self._layout_vars(),
            "passes": passes,
            "view": (self.view or self.textures[0]).name,
        }, warnings)

    # ---- notebook ------------------------------------------------------------

    def show(self, texture: Texture | None = None, **options):
        """Display the running technique. In a notebook this renders in the cell output."""
        from .display import show

        return show(self.compile(), view=texture.name if texture else None, **options)


class Program:
    """A compiled technique: JSON for the runtime, plus what the compiler noticed."""

    def __init__(self, json: dict, warnings: list[str]):
        self.json = json
        self.warnings = warnings

    def wgsl(self, pass_name: str) -> str:
        """The complete shader of one pass, including the generated declarations."""
        for p in self.json["passes"]:
            if p["name"] == pass_name:
                return p["code"]
        raise KeyError(pass_name)

    def __repr__(self):
        j = self.json
        return f'Program({j["name"]!r}: {len(j["textures"])} textures, {len(j["vars"]["fields"])} vars, {len(j["passes"])} passes)'
