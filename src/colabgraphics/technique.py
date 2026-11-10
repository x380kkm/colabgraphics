"""Describe a rendering technique and compile it into a program for the runtime.

A technique is a list of resources (textures), variables and passes. The author writes
each pass's shader in HLSL, and only its algorithmic part; `compile()` works out the rest:

- the HLSL declarations for every resource the shader can refer to by name, and one
  constant buffer, `vars`, holding the technique's variables,
- the translation of each entry point to WGSL, the language WebGPU runs, done by the
  Slang compiler (see slang.py),
- from the compiler's reflection data: which binding each resource got, where each
  variable sits in the constant buffer, and how many threads a compute group has.

The result is plain JSON that `js/runtime.js` executes, in a browser or headless.
"""
from __future__ import annotations

import base64
import re
import zlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import slang

# format -> (bytes per pixel, HLSL type of one texel, usable as compute output)
FORMATS = {
    "r8unorm": (1, "float", False),
    "rg8unorm": (2, "float2", False),
    "rgba8unorm": (4, "float4", True),
    "rgba8unorm-srgb": (4, "float4", False),
    "rgba8snorm": (4, "float4", True),
    "rgba16float": (8, "float4", True),
    "r32float": (4, "float", True),
    "rg32float": (8, "float2", True),
    "rgba32float": (16, "float4", True),
    "r32uint": (4, "uint", True),
    "rgba8uint": (4, "uint4", True),
    "depth32float": (4, "float", False),
}

# The only formats WebGPU lets a shader read and write through one binding.
READ_WRITE_FORMATS = ("r32float", "r32uint")

# HLSL type of a variable -> (scalar the runtime writes, number of scalars)
VAR_TYPES = {
    "float": ("f32", 1),
    "int": ("i32", 1),
    "uint": ("u32", 1),
    "float2": ("f32", 2),
    "float3": ("f32", 3),
    "float4": ("f32", 4),
    "float4x4": ("f32", 16),
}

# Every mesh uses this vertex layout: 44 bytes per vertex.
VERTEX_STRUCT = """struct Vertex
{
    float3 position : POSITION;
    float3 normal   : NORMAL;
    float2 uv       : TEXCOORD0;
    float3 color    : COLOR0;
};"""

SYS_STRUCT = """struct Sys
{
    float  time;        // seconds since the technique started
    uint   frame;       // frames rendered so far
    float2 resolution;  // size in pixels of what this pass writes to
    float4 mouse;       // xy: pointer position in pixels, z: 1 while pressed
};"""

RESERVED = {"vars", "sys", "linear_sampler", "nearest_sampler"}
IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DECLARED = "declared for you"   # the name the generated declarations carry in compiler messages

# What only a pixel shader can do (it needs screen-space derivatives), as it appears in the
# compiled WGSL, and the HLSL spelling to name in the message.
PIXEL_ONLY = {"textureSample": "Sample", "textureSampleBias": "SampleBias", "textureSampleCompare": "SampleCmp",
              "dpdx": "ddx", "dpdy": "ddy", "fwidth": "fwidth"}


class TechniqueError(Exception):
    """The technique description is inconsistent. The message says what and where."""


class ShaderError(TechniqueError):
    """A pass's shader does not compile. The message is the compiler's report, which names
    the pass and the line, counted from the start of that pass's code."""

    def __init__(self, pass_name: str, diagnostics: str, message: str | None = None):
        self.pass_name, self.diagnostics = pass_name, diagnostics
        super().__init__(message or f'pass "{pass_name}": the shader does not compile.\n\n{diagnostics}')


@dataclass(eq=False)
class Texture:
    name: str
    format: str
    size: tuple[int, int]
    data: str | None = None  # deflate-compressed pixels, base64

    def __repr__(self):
        return f"Texture({self.name!r}, {self.size[0]}x{self.size[1]}, {self.format})"


@dataclass(eq=False)
class Mesh:
    name: str
    vertex_count: int
    vertices: str             # interleaved float32 data, deflate-compressed, base64
    index_count: int = 0
    indices: str | None = None  # uint32 indices, same encoding

    def __repr__(self):
        shape = f"{self.index_count // 3} triangles, indexed" if self.index_count else f"{self.vertex_count // 3} triangles"
        return f"Mesh({self.name!r}, {self.vertex_count} vertices, {shape})"


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


def _pack(pixels: np.ndarray) -> str:
    return base64.b64encode(zlib.compress(np.ascontiguousarray(pixels).tobytes(), 6)).decode()


class Technique:
    def __init__(self, name: str = "technique"):
        self.name = name
        self.textures: list[Texture] = []
        self.vars: list[Var] = []
        self.meshes: list[Mesh] = []
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

    def mesh(self, name: str, positions, normals=None, uvs=None, colors=None, indices=None) -> Mesh:
        """Triangles for draw passes. Arrays hold one row per vertex; `indices` is optional.

        Without `indices`, every three vertices form a triangle. Missing normals are computed
        per triangle, missing uvs are zero, and missing colors are white. A single colour
        (three numbers) applies to all vertices.
        """
        self._check_name(name, "mesh")
        if any(m.name == name for m in self.meshes):
            raise TechniqueError(f'there is already a mesh named "{name}"')
        pos = np.asarray(positions, np.float32)
        if pos.ndim != 2 or pos.shape[1] != 3 or len(pos) == 0:
            raise TechniqueError(f'mesh "{name}": positions must be shaped (vertices, 3), got {pos.shape}')
        n = len(pos)
        idx = None
        if indices is not None:
            idx = np.asarray(indices).astype(np.uint32).reshape(-1)
            if len(idx) % 3 or (len(idx) and idx.max() >= n):
                raise TechniqueError(f'mesh "{name}": indices must come in threes and refer to existing vertices')
        elif n % 3:
            raise TechniqueError(f'mesh "{name}": without indices the vertex count must be a multiple of 3, got {n}')

        def per_vertex(values, width, what, default):
            if values is None:
                return default()
            a = np.asarray(values, np.float32)
            if a.shape == (width,):
                return np.broadcast_to(a, (n, width))
            if a.shape != (n, width):
                raise TechniqueError(f'mesh "{name}": {what} must be shaped ({n}, {width}), got {a.shape}')
            return a

        def face_normals():
            tri = pos[idx].reshape(-1, 3, 3) if idx is not None else pos.reshape(-1, 3, 3)
            fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-20)
            if idx is None:
                return np.repeat(fn, 3, axis=0)
            out = np.zeros((n, 3), np.float32)       # indexed: average the faces around each vertex
            np.add.at(out, idx, np.repeat(fn, 3, axis=0))
            return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-20)

        data = np.empty((n, 11), np.float32)
        data[:, 0:3] = pos
        data[:, 3:6] = per_vertex(normals, 3, "normals", face_normals)
        data[:, 6:8] = per_vertex(uvs, 2, "uvs", lambda: np.zeros((n, 2), np.float32))
        data[:, 8:11] = per_vertex(colors, 3, "colors", lambda: np.ones((n, 3), np.float32))
        mesh = Mesh(name, n, _pack(data), 0 if idx is None else len(idx), None if idx is None else _pack(idx))
        self.meshes.append(mesh)
        return mesh

    def var(self, name: str, value, min=None, max=None, step=None, doc: str = "", color: bool = False) -> Var:
        """A value the reader can change while the technique runs. Shaders read it as vars.<name>.

        The HLSL type follows the default: int, float, a tuple of 2 to 4 numbers (float2 to
        float4), or bool, which becomes a uint holding 0 or 1. A 4x4 numpy array becomes a
        float4x4 and gets no control in the viewer; give it in the usual maths layout, so
        that `mul(vars.m, float4(p, 1))` in the shader equals `m @ [x, y, z, 1]` in numpy.
        """
        self._check_name(name, "variable")
        if any(v.name == name for v in self.vars):
            raise TechniqueError(f'there is already a variable named "{name}"')
        if isinstance(value, np.ndarray) and value.shape == (4, 4):
            var = Var(name, "float4x4", _matrix(value), ui="hidden", doc=doc)
            self.vars.append(var)
            return var
        if isinstance(value, np.ndarray) and value.ndim == 1:
            value = tuple(float(x) for x in value)
        elif isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bool):
            hlsl, ui = "uint", "checkbox"
        elif isinstance(value, int):
            hlsl, ui = "int", "slider" if min is not None and max is not None else "number"
        elif isinstance(value, float):
            hlsl, ui = "float", "slider" if min is not None and max is not None else "number"
        elif isinstance(value, (tuple, list)) and len(value) in (2, 3, 4):
            hlsl, ui = f"float{len(value)}", "color" if color else "number"
            value = [float(x) for x in value]
        else:
            raise TechniqueError(f'variable "{name}": the default must be a bool, int, float, '
                                 f"a tuple of 2 to 4 numbers or a 4x4 array")
        var = Var(name, hlsl, value, min, max, step, ui, doc)
        self.vars.append(var)
        return var

    # ---- passes --------------------------------------------------------------

    def _add_pass(self, p: Pass) -> Pass:
        if any(q.name == p.name for q in self.passes):
            raise TechniqueError(f'there is already a pass named "{p.name}"')
        for hlsl_name, tex in p.reads.items():
            self._check_name(hlsl_name, f'pass "{p.name}": resource')
            self._owns(tex, p.name)
            if hlsl_name in p.options.get("writes", {}):
                raise TechniqueError(f'pass "{p.name}": the name "{hlsl_name}" is used for a texture it reads and for one it writes')
        self.passes.append(p)
        return p

    def _owns(self, tex, pass_name: str):
        if not isinstance(tex, Texture) or tex not in self.textures:
            raise TechniqueError(f'pass "{pass_name}": {tex!r} is not a texture of this technique')

    def compute(self, name: str, code: str, entry: str = "CSMain", reads: dict | None = None,
                writes: dict | None = None, over: Texture | None = None) -> Pass:
        """Run a compute shader once per pixel of `over` (by default, the first texture it writes).

        `code` is HLSL. `reads` and `writes` connect names used in the shader to textures:
        a name in `reads` is a Texture2D, a name in `writes` is a RWTexture2D. `entry` is
        the function to run.
        """
        reads, writes = dict(reads or {}), dict(writes or {})
        for hlsl_name, tex in writes.items():
            self._check_name(hlsl_name, f'pass "{name}": resource')
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
             vertices: int = 3, instances: int = 1, vertex: str = "VSMain", pixel: str = "PSMain",
             clear: tuple | None = (0, 0, 0, 1), depth: Texture | bool | None = None,
             topology: str = "triangle-list", cull: str = "none", mesh: Mesh | None = None) -> Pass:
        """Rasterize into `targets`: SV_Target0 goes to the first, SV_Target1 to the second, and so on.

        `code` is HLSL holding both the vertex shader and the pixel shader; `vertex` and
        `pixel` name the two functions. With `mesh=`, the vertex shader receives that mesh's
        vertices as a `Vertex` struct. Without it, `vertices` vertices are drawn and the
        vertex shader makes up the geometry from SV_VertexID.
        """
        if not targets:
            raise TechniqueError(f'pass "{name}": give at least one texture in targets=[...]')
        if mesh is not None and mesh not in self.meshes:
            raise TechniqueError(f'pass "{name}": {mesh!r} is not a mesh of this technique')
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
            "pixel": pixel, "clear": clear, "depth": depth or None, "topology": topology, "cull": cull,
            "mesh": mesh}))

    # ---- compile -------------------------------------------------------------

    def _declarations(self, p: Pass):
        """The HLSL placed in front of this pass's code, and what each declared name stands for.

        Everything the pass may use is declared; the compiler drops what the shader never
        touches and reports which names it did use.
        """
        lines = [f'// ---- declared for you: pass "{p.name}" ----']
        slots: dict[str, dict] = {}
        if self.vars:
            width = max(len(v.type) for v in self.vars)
            lines += ["struct Vars", "{"]
            for v in self.vars:
                doc = " ".join(v.doc.split())
                lines.append(f"    {v.type:<{width}} {v.name};" + (f"  // {doc}" if doc else ""))
            lines += ["};", "ConstantBuffer<Vars> vars;"]
            slots["vars"] = {"kind": "vars"}
        lines += [SYS_STRUCT, "ConstantBuffer<Sys> sys;"]
        slots["sys"] = {"kind": "sys"}
        if p.options.get("mesh"):
            lines.append(VERTEX_STRUCT)

        def connected(tex: Texture) -> str:
            return f'// texture "{tex.name}": {tex.size[0]} x {tex.size[1]}, {tex.format}'

        for name, tex in p.reads.items():
            lines.append(f"Texture2D<{FORMATS[tex.format][1]}> {name};  {connected(tex)}")
            slots[name] = {"kind": "texture", "name": name, "texture": tex.name}
        for name, tex in p.options.get("writes", {}).items():
            lines.append(f"RWTexture2D<{FORMATS[tex.format][1]}> {name};  {connected(tex)}")
            slots[name] = {"kind": "storage", "name": name, "texture": tex.name}
        lines += ["SamplerState linear_sampler;   // blends the four nearest texels",
                  "SamplerState nearest_sampler;  // takes the nearest texel"]
        slots["linear_sampler"] = {"kind": "sampler", "filter": "linear"}
        slots["nearest_sampler"] = {"kind": "sampler", "filter": "nearest"}
        return "\n".join(lines) + "\n", slots

    def _shader_error(self, p: Pass, diagnostics: str) -> ShaderError:
        missing = re.search(r"no function found matching entry point name '([^']+)'", diagnostics)
        located = [m for severity, m in slang.split_diagnostics(diagnostics)
                   if severity == "error" and "-->" in m and "entry point name" not in m]
        if missing and not located:
            option = "entry" if p.kind == "compute" else ("vertex" if missing.group(1) == p.options["vertex"] else "pixel")
            return ShaderError(p.name, diagnostics,
                               f'pass "{p.name}": its shader has no function named "{missing.group(1)}", which is the one '
                               f'this pass runs. Rename your function, or say which one to run with {option}="...".')
        if not self.vars and "undefined identifier 'vars'" in diagnostics:
            diagnostics += ("\n\nThis technique has no variables yet. Declare one with var(name, value) "
                            "and the shader can read it as vars.<name>.")
        return ShaderError(p.name, diagnostics)

    def _compile_pass(self, p: Pass):
        """Compile one pass. Returns its part of the program, warnings, and the reflection data."""
        declarations, slots = self._declarations(p)
        o = p.options
        entries = ((o["entry"], "compute"),) if p.kind == "compute" else ((o["vertex"], "vertex"), (o["pixel"], "fragment"))
        try:
            compiled = slang.compile(((DECLARED, declarations), (p.name, p.code)), entries)
        except slang.CompileError as e:
            raise self._shader_error(p, e.diagnostics) from None
        reflection = compiled.reflection

        stages = {}
        for (entry, stage), wgsl in zip(entries, compiled.wgsl):
            pixel_only = re.search(r"\b(" + "|".join(PIXEL_ONLY) + r")\s*\(", wgsl) if stage != "fragment" else None
            if pixel_only:
                hlsl = PIXEL_ONLY[pixel_only.group(1)]
                advice = ("use SampleLevel(sampler, uv, 0) instead, which names the mip level itself"
                          if hlsl.startswith("Sample") else "compute the difference yourself from neighbouring values")
                raise TechniqueError(f'pass "{p.name}": {hlsl}() needs screen-space derivatives, which exist only in a '
                                     f"pixel shader. In a {stage} shader, {advice}.")
            stages[stage] = {"entry": entry, "code": wgsl}

        # Which binding each declared name got, and which names the entry points really use.
        index = {q["name"]: q["binding"].get("index") for q in reflection["parameters"]}
        used = {b["name"] for e in reflection["entryPoints"] for b in e.get("bindings", []) if b["binding"].get("used")}
        bindings = []
        for name, slot in slots.items():
            if name not in used:
                continue
            if index.get(name) is None:
                raise slang.SlangError(f'pass "{p.name}": Slang {slang.version()} reported no binding for "{name}"')
            binding = dict(slot, binding=index[name])
            if slot["kind"] == "storage":
                tex = o["writes"][name]
                stages["compute"]["code"], reads = _retarget_storage(stages["compute"]["code"], binding["binding"], tex.format)
                if reads and tex.format not in READ_WRITE_FORMATS:
                    raise TechniqueError(
                        f'pass "{p.name}": the shader reads "{name}", the texture it also writes ("{tex.name}", {tex.format}). '
                        f"WebGPU allows reading and writing one texture in the same pass only for the formats "
                        f'{" and ".join(READ_WRITE_FORMATS)}. Read from a second texture instead.')
                if reads:
                    binding["access"] = "read-write"
            bindings.append(binding)

        warnings = [f'pass "{p.name}": "{n}" is connected but the shader never uses it'
                    for n in [*p.reads, *o.get("writes", {})] if n not in used]
        warnings += [f'pass "{p.name}": {message}'
                     for severity, message in slang.split_diagnostics(compiled.diagnostics) if severity == "warning"]

        out = {"kind": p.kind, "name": p.name, "stages": stages, "bindings": bindings}
        if p.kind == "compute":
            entry = next(e for e in reflection["entryPoints"] if e["name"] == o["entry"])
            out["dispatch"] = {"texture": o["over"].name, "workgroup": list(entry["threadGroupSize"])}
        else:
            out.update(vertexCount=o["vertices"], instanceCount=o["instances"], topology=o["topology"], cull=o["cull"],
                       targets=[{"texture": t.name, "clear": list(o["clear"]) if o["clear"] else None} for t in o["targets"]],
                       depth={"texture": o["depth"].name} if o["depth"] else None,
                       mesh=o["mesh"].name if o["mesh"] else None)
        return out, warnings, declarations, reflection

    def _layout_vars(self, reflection: dict):
        """Where each variable sits in the constant buffer. The compiler decides; this reads it back."""
        if not self.vars:
            return {"size": 0, "fields": []}
        layout = next(q for q in reflection["parameters"] if q["name"] == "vars")["type"]["elementVarLayout"]
        offsets = {f["name"]: f["binding"]["offset"] for f in layout["type"]["fields"]}
        fields = []
        for v in self.vars:
            scalar, count = VAR_TYPES[v.type]
            fields.append({"name": v.name, "type": v.type, "scalar": scalar, "count": count, "offset": offsets[v.name],
                           "value": int(v.value) if isinstance(v.value, bool) else v.value,
                           "min": v.min, "max": v.max, "step": v.step, "ui": v.ui, "doc": v.doc})
        return {"size": layout["binding"]["size"], "fields": fields}

    def compile(self) -> "Program":
        if not self.passes:
            raise TechniqueError("the technique has no passes yet: add one with compute(...) or draw(...)")
        passes, warnings, sources, reflection = [], [], {}, {}
        for p in self.passes:
            out, notes, declarations, reflection = self._compile_pass(p)
            passes.append(out)
            warnings += notes
            sources[p.name] = (declarations, p.code)
        return Program({
            "name": self.name,
            "textures": [{"name": t.name, "format": t.format, "size": list(t.size), **({"data": t.data} if t.data else {})}
                         for t in self.textures],
            "vars": self._layout_vars(reflection),   # every pass declares the same Vars, so any reflection will do
            "meshes": [{"name": m.name, "vertexCount": m.vertex_count, "vertices": m.vertices,
                        "indexCount": m.index_count, **({"indices": m.indices} if m.indices else {})} for m in self.meshes],
            "passes": passes,
            "view": (self.view or self.textures[0]).name,
        }, warnings, sources)

    # ---- notebook ------------------------------------------------------------

    def show(self, texture: Texture | None = None, **options):
        """Display the running technique. In a notebook this renders in the cell output."""
        from .display import show

        return show(self.compile(), view=texture.name if texture else None, **options)

    def read(self, texture, frames: int = 1, **vars):
        """Run the technique without a browser and return `texture` as an array (height, width, channels).

        Give a list of textures to get a list of arrays from the same run. Keyword arguments
        set variables for this run, for example `t.read(out, radius=4)`.
        """
        from . import headless

        many = isinstance(texture, (list, tuple))
        textures = list(texture) if many else [texture]
        for tex in textures:
            self._owns(tex, "read")
        values = {}
        for name, value in vars.items():
            var = next((v for v in self.vars if v.name == name), None)
            if var is None:
                known = ", ".join(v.name for v in self.vars) or "none are defined"
                raise TechniqueError(f'read(): "{name}" is not a variable of this technique (known: {known})')
            values[name] = _matrix(value) if var.type == "float4x4" else np.asarray(value).tolist()
        arrays = headless.run(self.compile(), read=[tex.name for tex in textures], frames=frames, vars=values)
        return [arrays[tex.name] for tex in textures] if many else arrays[texture.name]


def _matrix(value) -> list[float]:
    """The sixteen numbers of a float4x4, row after row: numpy's own order, and the order in
    which Slang lays a matrix out in a constant buffer."""
    m = np.asarray(value, np.float32)
    if m.shape != (4, 4):
        raise TechniqueError(f"a float4x4 variable needs a 4x4 array, got shape {m.shape}")
    return [float(x) for x in m.reshape(-1)]


def _retarget_storage(wgsl: str, binding: int, fmt: str):
    """Fix up the declaration of one texture a compute shader writes.

    Slang declares every RWTexture2D as read_write storage with a format guessed from the
    HLSL type. WebGPU wants the texture's real format, and write-only access unless the
    shader also reads it. Returns the new WGSL and whether the shader reads the texture.
    """
    found = re.search(rf"(@binding\({binding}\)\s*@group\(0\)\s*var\s+(\w+)\s*:\s*texture_storage_2d)<[^>]*>", wgsl)
    if not found:
        raise slang.SlangError(f"binding {binding} is not declared as a storage texture in the compiled WGSL; "
                               f"Slang {slang.version()} writes it in a form this code does not know")
    reads = re.search(rf"\btextureLoad\s*\(\s*\(*\s*{found.group(2)}\b", wgsl) is not None
    access = "read_write" if reads else "write"
    return wgsl[:found.start()] + f"{found.group(1)}<{fmt}, {access}>" + wgsl[found.end():], reads


class Program:
    """A compiled technique: JSON for the runtime, plus what the compiler noticed."""

    def __init__(self, json: dict, warnings: list[str], sources: dict[str, tuple[str, str]]):
        self.json = json
        self.warnings = warnings
        self._sources = sources   # pass name -> (generated declarations, the author's code)

    def declarations(self, pass_name: str) -> str:
        """The HLSL the system wrote in front of one pass's code."""
        return self._sources[pass_name][0]

    def hlsl(self, pass_name: str) -> str:
        """The complete HLSL of one pass: the generated declarations, then the author's code."""
        declarations, code = self._sources[pass_name]
        return declarations + "// ---- your code ----\n" + code

    def wgsl(self, pass_name: str, stage: str | None = None) -> str:
        """What one pass was compiled to: the WGSL the GPU runs.

        A draw pass has two shaders; `stage="vertex"` or `stage="pixel"` selects one.
        """
        stages = next(p for p in self.json["passes"] if p["name"] == pass_name)["stages"]
        if stage is not None:
            return stages["fragment" if stage == "pixel" else stage]["code"]
        if len(stages) == 1:
            return next(iter(stages.values()))["code"]
        return "\n".join(f'// ---- {"pixel" if s == "fragment" else s} shader: {v["entry"]} ----\n{v["code"]}'
                         for s, v in stages.items())

    def __repr__(self):
        j = self.json
        return f'Program({j["name"]!r}: {len(j["textures"])} textures, {len(j["vars"]["fields"])} vars, {len(j["passes"])} passes)'
