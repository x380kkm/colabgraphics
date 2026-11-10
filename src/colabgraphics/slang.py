"""HLSL in, WGSL out: a thin wrapper around the Slang compiler.

Slang (https://shader-slang.org) is the compiler the industry uses to take HLSL to other
targets, and it is what Gigi uses for its WebGPU output. Here it is driven through its C
API with ctypes, in the process, so nothing has to be started or parsed from a command line.

The compiler library comes from the `slangpy` pip package. Only the shared library inside
it is used; the package itself is never imported. Set COLABGRAPHICS_SLANG to the path of
another `libslang-compiler` (or the folder that holds it) to use a different build.
"""
from __future__ import annotations

import ctypes
import functools
import glob
import importlib.util
import json
import os
import re
from dataclasses import dataclass

# Values from slang.h
_TARGET_WGSL = 28
_LANGUAGE_SLANG = 1      # Slang's own language is a superset of HLSL
_STAGES = {"vertex": 1, "fragment": 5, "compute": 6}
_LIBRARY_NAMES = ("libslang-compiler.so*", "libslang.so*", "libslang-compiler*.dylib", "libslang.dylib",
                  "slang-compiler.dll", "slang.dll")


class SlangError(Exception):
    """The Slang compiler itself is missing or failed. Not a problem in the shader."""


class CompileError(Exception):
    """The shader did not compile. `diagnostics` is the compiler's own report."""

    def __init__(self, diagnostics: str):
        super().__init__(diagnostics)
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class Compiled:
    wgsl: tuple[str, ...]      # one WGSL module per entry point, in the order they were asked for
    reflection: dict           # what the compiler knows about parameters and entry points
    diagnostics: str           # warnings; empty when there are none


def _find_library() -> str:
    places = []
    override = os.environ.get("COLABGRAPHICS_SLANG")
    if override:
        if os.path.isfile(override):
            return override
        places.append(override)
    spec = importlib.util.find_spec("slangpy")   # locates the package without importing it
    if spec and spec.submodule_search_locations:
        places += list(spec.submodule_search_locations)
    for place in places:
        for name in _LIBRARY_NAMES:
            found = sorted(glob.glob(os.path.join(place, name)))
            if found:
                return found[0]
    raise SlangError("The Slang shader compiler was not found. Install it with:  pip install slangpy")


class _Api:
    """The handful of functions of Slang's C API that are needed, and one compiler session."""

    def __init__(self):
        self.path = _find_library()
        try:
            lib = ctypes.CDLL(self.path)
        except OSError as e:
            raise SlangError(f"could not load the Slang compiler from {self.path}: {e}") from None
        P, I, S = ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p

        def fn(name, restype, *argtypes):
            f = getattr(lib, name)
            f.restype, f.argtypes = restype, argtypes
            return f

        self.create_request = fn("spCreateCompileRequest", P, P)
        self.destroy_request = fn("spDestroyCompileRequest", None, P)
        self.set_target = fn("spSetCodeGenTarget", I, P, I)
        self.add_unit = fn("spAddTranslationUnit", I, P, I, S)
        self.add_source = fn("spAddTranslationUnitSourceString", None, P, I, S, S)
        self.add_entry = fn("spAddEntryPoint", I, P, I, S, I)
        self.compile = fn("spCompile", I, P)
        self.diagnostics = fn("spGetDiagnosticOutput", S, P)
        self.entry_source = fn("spGetEntryPointSource", S, P, I)
        self.reflection = fn("spGetReflection", P, P)
        self.reflection_json = fn("spReflection_ToJson", I, P, P, ctypes.POINTER(P))
        self.version = fn("spGetBuildTagString", S)().decode()
        self.session = fn("spCreateSession", P, S)(None)   # loads the core module: about 0.2 s, once
        if not self.session:
            raise SlangError(f"the Slang compiler at {self.path} could not start a session")


@functools.lru_cache(maxsize=1)
def _api() -> _Api:
    return _Api()


def _blob_bytes(blob) -> bytes:
    """Read and release an ISlangBlob. It is a COM object: a pointer to a table of functions
    [queryInterface, addRef, release, getBufferPointer, getBufferSize]."""
    P = ctypes.c_void_p
    table = ctypes.cast(ctypes.cast(blob, ctypes.POINTER(P))[0], ctypes.POINTER(P))
    pointer = ctypes.CFUNCTYPE(P, P)(table[3])(blob)
    size = ctypes.CFUNCTYPE(ctypes.c_size_t, P)(table[4])(blob)
    data = ctypes.string_at(pointer, size)
    ctypes.CFUNCTYPE(ctypes.c_uint32, P)(table[2])(blob)
    return data


def version() -> str:
    return _api().version


@functools.lru_cache(maxsize=256)
def compile(sources: tuple[tuple[str, str], ...], entries: tuple[tuple[str, str], ...]) -> Compiled:
    """Compile HLSL to WGSL.

    sources: (name, text) pairs that together form one shader; names appear in diagnostics.
    entries: (function name, stage) pairs; stage is "vertex", "fragment" or "compute".

    Results are remembered, so compiling the same shader again costs nothing. Raises
    CompileError with the compiler's report when the shader has errors.
    """
    api = _api()
    request = api.create_request(api.session)
    try:
        api.set_target(request, _TARGET_WGSL)
        unit = api.add_unit(request, _LANGUAGE_SLANG, b"")
        for name, text in sources:
            api.add_source(request, unit, name.encode(), text.encode())
        indices = [api.add_entry(request, unit, name.encode(), _STAGES[stage]) for name, stage in entries]
        failed = api.compile(request)
        diagnostics = (api.diagnostics(request) or b"").decode(errors="replace").strip()
        if failed:
            raise CompileError(diagnostics or "the Slang compiler reported a failure without any message")
        wgsl = tuple((api.entry_source(request, i) or b"").decode() for i in indices)
        blob = ctypes.c_void_p()
        api.reflection_json(api.reflection(request), request, ctypes.byref(blob))
        if not blob.value:
            raise SlangError("the Slang compiler returned no reflection data")
        return Compiled(wgsl, json.loads(_blob_bytes(blob)), diagnostics)
    finally:
        api.destroy_request(request)


def split_diagnostics(text: str) -> list[tuple[str, str]]:
    """Split a compiler report into (severity, message) items; notes stay with what they explain."""
    items: list[tuple[str, str]] = []
    for part in re.split(r"^(?=(?:error|warning|note)\b)", text, flags=re.M):
        part = part.strip("\n")
        if not part.strip():
            continue
        severity = re.match(r"\w+", part).group(0)
        if severity == "note" and items:
            items[-1] = (items[-1][0], items[-1][1] + "\n" + part)
        else:
            items.append((severity, part))
    return items
