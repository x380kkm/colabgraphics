"""End-to-end tests: build a technique in Python with HLSL shaders, run it in a real
browser engine or headless, read the result back and compare it with numpy.

    PYTHONPATH=src python3 -m pytest tests -q        (or run this file directly)
"""
import json
import pathlib
import subprocess
import tempfile

import numpy as np

import colabgraphics as cg

HERE = pathlib.Path(__file__).parent

BLUR = """
[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    int2 p = int2(id.xy);
    uint2 size;
    image.GetDimensions(size.x, size.y);
    float4 sum = 0;
    for (int i = -vars.radius; i <= vars.radius; i++)
        sum += image[clamp(p + i * DIR, int2(0, 0), int2(size) - 1)];
    result[p] = sum / float(2 * vars.radius + 1);
}
"""

TRIANGLE = """
struct VSOut
{
    float4 position : SV_Position;
    float3 color : COLOR0;
};

VSOut VSMain(uint i : SV_VertexID)
{
    static const float2 corners[3] = { float2(0.0, 0.8), float2(-0.8, -0.8), float2(0.8, -0.8) };
    static const float3 colors[3] = { float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 1) };
    VSOut o;
    o.position = float4(corners[i], 0, 1);
    o.color = colors[i];
    return o;
}

float4 PSMain(VSOut input) : SV_Target
{
    return float4(input.color * vars.brightness, 1);
}
"""


def run_in_browser_raw(program, texture="out", *sets, screenshot=None, out=None):
    """Open the program's page in headless Chromium; return what the page reported."""
    with tempfile.TemporaryDirectory() as tmp:
        page = pathlib.Path(tmp, "page.html")
        page.write_text(cg.standalone_html(program))
        cmd = ["node", str(HERE / "browser_run.cjs"), str(page), texture, str(out or pathlib.Path(tmp, "out.bin")), screenshot or "", *sets]
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        return json.loads(done.stdout.strip().splitlines()[-1])


def run_in_browser(technique, texture, *sets, screenshot=None):
    """Render in headless Chromium; return (info, pixels as an array shaped (h, w, channels))."""
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp, "out.bin")
        info = run_in_browser_raw(technique.compile(), texture, *sets, screenshot=screenshot, out=out)
        assert not info.get("failed"), info["failed"]
        assert not info["problems"], info["problems"]
        dtype = {"rgba8unorm": np.uint8, "rgba32float": np.float32}[info["format"]]
        return info, np.frombuffer(out.read_bytes(), dtype).reshape(info["height"], info["width"], -1)


def box_blur_reference(image, radius):
    """Separable clamped box blur on rgba8, quantizing after each pass like the GPU textures do."""
    def one_pass(a, axis):
        a = a.astype(np.float32) / 255
        n = a.shape[axis]
        acc = np.zeros_like(a)
        for i in range(-radius, radius + 1):
            acc += np.take(a, np.clip(np.arange(n) + i, 0, n - 1), axis=axis)
        return np.rint(acc / (2 * radius + 1) * 255).astype(np.uint8)
    return one_pass(one_pass(image, 1), 0)


def make_boxblur(image):
    t = cg.Technique("boxblur")
    src = t.image("src", image)
    tmp = t.texture("tmp", like=src)
    out = t.texture("out", like=src)
    t.var("radius", 3, min=0, max=16, doc="half width of the box, in pixels")
    t.compute("blur_h", "static const int2 DIR = int2(1, 0);" + BLUR, reads={"image": src}, writes={"result": tmp})
    t.compute("blur_v", "static const int2 DIR = int2(0, 1);" + BLUR, reads={"image": tmp}, writes={"result": out})
    return t


def random_image(seed, height, width):
    rng = np.random.default_rng(seed)
    return np.dstack([rng.integers(0, 256, (height, width, 3), dtype=np.uint8), np.full((height, width), 255, np.uint8)])


def expect_error(build, *fragments, kind=cg.TechniqueError):
    try:
        build()
    except kind as e:
        for fragment in fragments:
            assert fragment in str(e), f"expected {fragment!r} in:\n{e}"
        return e
    raise AssertionError(f"expected an error mentioning {fragments}")


def test_boxblur_matches_numpy():
    image = random_image(7, 48, 80)
    technique = make_boxblur(image)
    for radius in (3, 0, 9):
        info, got = run_in_browser(technique, "out", f"radius={radius}")
        expected = box_blur_reference(image, radius)
        worst = int(np.abs(got.astype(int) - expected.astype(int)).max())
        assert got.shape == expected.shape
        # Each pass may round a half differently from numpy, so allow one level per pass.
        assert worst <= 2, f"radius {radius}: differs from the reference by up to {worst}/255"
        # The single-pixel readout must agree with the full read-back.
        assert np.allclose(np.array(info["pixel"]) * 255, got[7, 10], atol=0.01), (info["pixel"], got[7, 10])


def test_triangle_covers_expected_pixels():
    t = cg.Technique("triangle")
    color = t.texture("color", size=(128, 128))
    t.var("brightness", 1.0, min=0.0, max=1.0)
    t.draw("tri", TRIANGLE, targets=[color], vertices=3)
    info, got = run_in_browser(t, "color")
    covered = got[..., :3].any(axis=-1)
    # The triangle spans 1.6 x 1.6 in clip space: half of base times height, as a share of the 2 x 2 square.
    expected_share = (0.5 * 1.6 * 1.6) / 4
    assert abs(covered.mean() - expected_share) < 0.01, covered.mean()
    assert not covered[0, 0] and not covered[-1, -1]          # corners stay cleared
    centre = got[80, 64, :3].astype(float) / 255               # inside: the three colours sum to one
    assert abs(centre.sum() - 1) < 0.03, centre
    info, dim = run_in_browser(t, "color", "brightness=0.5")
    assert abs(dim[80, 64, :3].astype(float).sum() / 255 - 0.5) < 0.03


def test_headless_matches_numpy_and_browser():
    image = random_image(11, 40, 56)
    technique = make_boxblur(image)
    out = next(t for t in technique.textures if t.name == "out")
    headless = technique.read(out, radius=4)
    expected = box_blur_reference(image, 4)
    assert int(np.abs(headless.astype(int) - expected.astype(int)).max()) <= 2
    _, browser = run_in_browser(technique, "out", "radius=4")
    assert int(np.abs(headless.astype(int) - browser.astype(int)).max()) <= 2


def test_variables_of_every_type_reach_the_shader():
    """Each variable is written into a pixel and read back, so a wrong offset in the constant
    buffer or a transposed matrix shows up as a wrong number."""
    matrix = np.arange(16, dtype=np.float32).reshape(4, 4) + 1      # not symmetric on purpose
    point = np.array([1.0, 2.0, 3.0, 1.0], np.float32)
    t = cg.Technique("vars")
    out = t.texture("out", size=(8, 1), format="rgba32float")
    t.var("count", 7)
    t.var("gain", 0.25)
    t.var("tint", (0.1, 0.2, 0.3))
    t.var("enabled", True)
    t.var("offset", (5.0, 6.0))
    t.var("m", matrix)
    t.var("last", 9.5)
    t.var("corner", (1.0, 2.0, 3.0, 4.0))
    t.compute("dump", """
[numthreads(1, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    float4 v = 0;
    switch (id.x)
    {
    case 0: v = float4(vars.count, vars.gain, vars.enabled, vars.last); break;
    case 1: v = float4(vars.tint, 0); break;
    case 2: v = float4(vars.offset, 0, 0); break;
    case 3: v = vars.corner; break;
    case 4: v = mul(vars.m, float4(1, 2, 3, 1)); break;      // matrix times column vector
    case 5: v = mul(float4(1, 2, 3, 1), vars.m); break;      // row vector times matrix
    case 6: v = vars.m[1]; break;                            // second row
    default: v = float4(sys.resolution, sys.frame, 0); break;
    }
    result[id.xy] = v;
}
""", writes={"result": out})
    got = t.read(out)[0]
    assert np.allclose(got[0], [7, 0.25, 1, 9.5]), got[0]
    assert np.allclose(got[1], [0.1, 0.2, 0.3, 0]), got[1]
    assert np.allclose(got[2], [5, 6, 0, 0]), got[2]
    assert np.allclose(got[3], [1, 2, 3, 4]), got[3]
    assert np.allclose(got[4], matrix @ point), (got[4], matrix @ point)
    assert np.allclose(got[5], point @ matrix), (got[5], point @ matrix)
    assert np.allclose(got[6], matrix[1]), got[6]
    assert np.allclose(got[7], [8, 1, 0, 0]), got[7]

    # Values given to read() replace the defaults for that run, matrices included.
    other = np.eye(4, dtype=np.float32)
    other[0, 3] = 10                                           # a translation along x
    got = t.read(out, count=-3, enabled=False, m=other, tint=(1, 0, 0))[0]
    assert np.allclose(got[0], [-3, 0.25, 0, 9.5]), got[0]
    assert np.allclose(got[1], [1, 0, 0, 0]), got[1]
    assert np.allclose(got[4], [11, 2, 3, 1]), got[4]


def test_fullscreen_sampling_and_reading_depth():
    image = random_image(3, 16, 24)
    t = cg.Technique("sample")
    src = t.image("src", image)
    copy = t.texture("copy", like=src)
    depth = t.texture("depth", like=src, format="depth32float")
    seen = t.texture("seen", like=src, format="r32float")
    # One triangle that covers the screen; uv (0, 0) is the top left corner, as in the image array.
    t.draw("copy", """
struct VSOut
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

VSOut VSMain(uint i : SV_VertexID)
{
    VSOut o;
    o.uv = float2((i << 1) & 2, i & 2);
    o.position = float4(o.uv * float2(2, -2) + float2(-1, 1), 0.25, 1);
    return o;
}

float4 PSMain(VSOut input) : SV_Target
{
    return image.Sample(nearest_sampler, input.uv);
}
""", targets=[copy], reads={"image": src}, depth=depth)
    t.compute("look", """
[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    result[id.xy] = z[id.xy];
}
""", reads={"z": depth}, writes={"result": seen})
    got, z = t.read([copy, seen])
    assert np.array_equal(got, image)
    assert np.allclose(z, 0.25), (z.min(), z.max())


def test_fullscreen_runs_a_pixel_shader_on_every_pixel():
    ys, xs = np.mgrid[0:8, 0:16]
    shaders = {
        "separate inputs": "float4 PSMain(float2 uv : TEXCOORD0, float4 position : SV_Position) : SV_Target"
                           " { return float4(uv, position.xy); }",
        "the struct": "float4 PSMain(Fullscreen input) : SV_Target { return float4(input.uv, input.position.xy); }",
    }
    for how, code in shaders.items():
        t = cg.Technique("fullscreen")
        out = t.texture("out", size=(16, 8), format="rgba32float")
        t.fullscreen("gradient", code, targets=[out])
        got = t.read(out)
        assert np.allclose(got[..., 0], (xs + 0.5) / 16, atol=1e-4), how      # uv: (0, 0) is the top left corner
        assert np.allclose(got[..., 1], (ys + 0.5) / 8, atol=1e-4), how
        assert np.allclose(got[..., 2], xs + 0.5) and np.allclose(got[..., 3], ys + 0.5), how   # pixel centres

    # No inputs at all, variables and sys, in the browser.
    t = cg.Technique("fullscreen")
    out = t.texture("out", size=(16, 8))
    t.var("level", 0.5)
    t.fullscreen("flat", "float4 PSMain() : SV_Target { return float4(vars.level, sys.resolution / 255.0, 1); }", targets=[out])
    _, got = run_in_browser(t, "out", "level=1.0")
    assert (got == [255, 16, 8, 255]).all(), got[0, 0]

    a = t.texture("a", size=(4, 4))
    expect_error(lambda: t.fullscreen("two", "float4 PSMain() : SV_Target { return 0; }", targets=[out, a]), "same size")


def test_a_texture_can_accumulate_over_frames():
    """r32float is one of the two formats WebGPU lets a pass read and write in place."""
    t = cg.Technique("count")
    total = t.texture("total", size=(8, 8), format="r32float")
    t.compute("add", """
[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    total[id.xy] = total[id.xy] + 1.0 + float(id.x);
}
""", writes={"total": total})
    got = t.read(total, frames=5)[..., 0]
    assert np.allclose(got, 5 * (1 + np.arange(8))[None, :]), got


def test_slang_features_beyond_hlsl_work():
    """The source language is Slang, a superset of HLSL: interfaces and generics compile too."""
    t = cg.Technique("slang")
    out = t.texture("out", size=(4, 1), format="rgba32float")
    t.var("k", 2.0)
    t.compute("p", """
interface IShape { float area(); }
struct Disc : IShape { float r; float area() { return 3.0 * r * r; } }
struct Box : IShape { float2 size; float area() { return size.x * size.y; } }

float twice<T : IShape>(T shape) { return 2.0 * shape.area(); }

[numthreads(1, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    Disc d; d.r = vars.k;
    Box b; b.size = float2(vars.k, 5);
    result[id.xy] = float4(twice(d), twice(b), id.x, 0);
}
""", writes={"result": out})
    got = t.read(out)[0]
    assert np.allclose(got[3], [24, 20, 3, 0]), got


def test_shader_errors_name_the_pass_and_the_authors_line():
    t = cg.Technique("bad")
    o = t.texture("o", size=(8, 8))
    code = "[numthreads(1, 1, 1)]\nvoid CSMain(uint3 id : SV_DispatchThreadID)\n{\n    o[id.xy] = float4(1, 1, 1, 1) * nope;\n}\n"
    t.compute("p", code, writes={"o": o})
    e = expect_error(t.compile, 'pass "p"', "p:4:", "nope", kind=cg.ShaderError)
    assert e.pass_name == "p"
    expect_error(lambda: t.read(o), "p:4:", kind=cg.ShaderError)     # every way of running it reports the same

    t = cg.Technique("bad")
    o = t.texture("o", size=(8, 8))
    t.compute("p", code.replace("* nope", "").replace("CSMain", "Main"), writes={"o": o})
    expect_error(t.compile, 'no function named "CSMain"', 'entry="..."', kind=cg.ShaderError)

    t = cg.Technique("bad")
    o = t.texture("o", size=(8, 8))
    t.compute("p", code.replace("nope", "vars.gain"), writes={"o": o})
    expect_error(t.compile, "undefined identifier 'vars'", "no variables yet", kind=cg.ShaderError)

    t = cg.Technique("bad")
    o = t.texture("o", size=(8, 8))
    t.var("gain", 1.0)
    t.compute("p", code.replace("nope", "vars.gian"), writes={"o": o})
    expect_error(t.compile, "'gian' is not a member of 'Vars'", kind=cg.ShaderError)


def test_a_compiled_shader_the_gpu_rejects_is_reported():
    """The HLSL is checked when it is compiled. If the WGSL made from it is still refused,
    the message must say which pass and quote the line."""
    from colabgraphics import headless

    program = make_boxblur(random_image(5, 8, 8)).compile()
    stage = program.json["passes"][1]["stages"]["compute"]
    stage["code"] = stage["code"].replace("textureStore(", "textureStoree(")
    expect_error(lambda: headless.run(program, read=["out"]), 'pass "blur_v": the compiled compute shader was rejected',
                 "textureStoree", "WGSL line", kind=headless.HeadlessError)
    info = run_in_browser_raw(program)
    assert 'pass "blur_v": this browser rejected the compiled compute shader' in info["failed"], info
    assert "WGSL line" in info["failed"] and "textureStoree" in info["failed"], info


def test_errors_name_the_problem():
    t = cg.Technique("bad")
    a = t.texture("a", size=(8, 8))
    b = t.texture("b", size=(8, 8))
    expect_error(lambda: t.compute("p", BLUR, reads={"image": a}, writes={"result": a}), "both reads and writes")
    expect_error(lambda: t.compute("p", BLUR, reads={"image": a}, writes={"image": b}), "reads and for one it writes")
    expect_error(lambda: t.texture("vars", size=(1, 1)), "reserved")
    expect_error(lambda: t.texture("a", size=(1, 1)), "already a texture")
    expect_error(lambda: t.var("m", np.zeros((3, 3))), "the default must be")

    # Reading the texture a pass writes is a WebGPU restriction; the message says which formats may.
    t.compute("p", "[numthreads(1, 1, 1)] void CSMain(uint3 id : SV_DispatchThreadID) { o[id.xy] = o[id.xy] * 0.5; }",
              writes={"o": a})
    expect_error(t.compile, 'reads "o"', "r32float", "second texture")

    # Sample() in a compute shader is the classic first mistake; say what to use instead.
    t = cg.Technique("bad")
    a = t.texture("a", size=(8, 8))
    b = t.texture("b", size=(8, 8))
    t.compute("p", "[numthreads(1, 1, 1)] void CSMain(uint3 id : SV_DispatchThreadID) "
                   "{ o[id.xy] = image.Sample(linear_sampler, float2(0.5, 0.5)); }", reads={"image": a}, writes={"o": b})
    expect_error(t.compile, "Sample()", "SampleLevel")


def test_warnings_are_collected():
    t = cg.Technique("warn")
    a = t.texture("a", size=(8, 8))
    b = t.texture("b", size=(8, 8))
    t.var("enabled", True)
    t.compute("p", "[numthreads(1, 1, 1)]\nvoid CSMain(uint3 id : SV_DispatchThreadID)\n{\n"
                   "    if (vars.enabled) o[id.xy] = float4(1, 0, 0, 1);\n}\n", reads={"image": a}, writes={"o": b})
    warnings = t.compile().warnings
    assert any('"image" is connected but the shader never uses it' in w for w in warnings), warnings
    assert any("implicit conversion" in w and "p:4:" in w for w in warnings), warnings


def test_the_generated_declarations_can_be_inspected():
    program = make_boxblur(random_image(1, 8, 8)).compile()
    declarations = program.declarations("blur_h")
    assert "int radius;  // half width of the box, in pixels" in declarations
    assert 'Texture2D<float4> image;  // texture "src": 8 x 8, rgba8unorm' in declarations
    assert 'RWTexture2D<float4> result;  // texture "tmp"' in declarations
    assert program.hlsl("blur_h").startswith(declarations) and "CSMain" in program.hlsl("blur_h")
    assert "texture_storage_2d<rgba8unorm, write>" in program.wgsl("blur_h")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
