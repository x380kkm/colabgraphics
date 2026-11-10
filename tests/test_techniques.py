"""End-to-end tests: build a technique in Python, run it in a real browser engine,
read the result back and compare it with a reference computed with numpy.

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
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let p = vec2i(id.xy);
  let size = vec2i(textureDimensions(image));
  var sum = vec4f(0);
  for (var i = -vars.radius; i <= vars.radius; i++) {
    sum += textureLoad(image, clamp(p + i * DIR, vec2i(0), size - 1), 0);
  }
  textureStore(result, p, sum / f32(2 * vars.radius + 1));
}
"""

TRIANGLE = """
struct Out { @builtin(position) pos: vec4f, @location(0) color: vec3f }

@vertex fn vs(@builtin(vertex_index) i: u32) -> Out {
  let corners = array(vec2f(0.0, 0.8), vec2f(-0.8, -0.8), vec2f(0.8, -0.8));
  let colors = array(vec3f(1, 0, 0), vec3f(0, 1, 0), vec3f(0, 0, 1));
  return Out(vec4f(corners[i], 0, 1), colors[i]);
}

@fragment fn fs(in: Out) -> @location(0) vec4f {
  return vec4f(in.color * vars.brightness, 1);
}
"""


def run_in_browser(technique, texture, *sets, screenshot=None):
    """Render in headless Chromium; return (info, pixels as an array shaped (h, w, channels))."""
    program = technique.compile()
    with tempfile.TemporaryDirectory() as tmp:
        page, out = pathlib.Path(tmp, "page.html"), pathlib.Path(tmp, "out.bin")
        page.write_text(cg.standalone_html(program))
        cmd = ["node", str(HERE / "browser_run.cjs"), str(page), texture, str(out), screenshot or "", *sets]
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        info = json.loads(done.stdout.strip().splitlines()[-1])
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
    t.compute("blur_h", "const DIR = vec2i(1, 0);" + BLUR, reads={"image": src}, writes={"result": tmp})
    t.compute("blur_v", "const DIR = vec2i(0, 1);" + BLUR, reads={"image": tmp}, writes={"result": out})
    return t


def test_boxblur_matches_numpy():
    rng = np.random.default_rng(7)
    image = np.dstack([rng.integers(0, 256, (48, 80, 3), dtype=np.uint8), np.full((48, 80), 255, np.uint8)])
    technique = make_boxblur(image)
    for radius in (3, 0, 9):
        info, got = run_in_browser(technique, "out", f"radius={radius}")
        expected = box_blur_reference(image, radius)
        worst = int(np.abs(got.astype(int) - expected.astype(int)).max())
        assert got.shape == expected.shape
        # Each pass may round a half differently from numpy, so allow one level per pass.
        assert worst <= 2, f"radius {radius}: differs from the reference by up to {worst}/255"


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
    rng = np.random.default_rng(11)
    image = np.dstack([rng.integers(0, 256, (40, 56, 3), dtype=np.uint8), np.full((40, 56), 255, np.uint8)])
    technique = make_boxblur(image)
    out = next(t for t in technique.textures if t.name == "out")
    headless = technique.read(out, radius=4)
    expected = box_blur_reference(image, 4)
    assert int(np.abs(headless.astype(int) - expected.astype(int)).max()) <= 2
    _, browser = run_in_browser(technique, "out", "radius=4")
    assert int(np.abs(headless.astype(int) - browser.astype(int)).max()) <= 2


def test_headless_reports_shader_errors_on_the_authors_line():
    from colabgraphics.headless import HeadlessError

    t = cg.Technique("bad")
    o = t.texture("o", size=(8, 8))
    code = "@compute @workgroup_size(1)\nfn main(@builtin(global_invocation_id) id: vec3u) {\n  textureStore(o, id.xy, vec4f(1.0) * nope);\n}\n"
    t.compute("p", code, writes={"o": o})
    try:
        t.read(o)
        raise AssertionError("expected a shader error")
    except HeadlessError as e:
        assert 'pass "p", line 3' in str(e) and "nope" in str(e), str(e)


def test_errors_name_the_problem():
    t = cg.Technique("bad")
    a = t.texture("a", size=(8, 8))
    for build, expect in [
        (lambda: t.compute("p1", "@compute @workgroup_size(1) fn main() { _ = vars.missing; }", writes={"o": a}), None),
        (lambda: t.compute("p2", BLUR, reads={"image": a}, writes={"result": a}), "both reads and writes"),
        (lambda: t.texture("vars", size=(1, 1)), "reserved"),
        (lambda: t.texture("a", size=(1, 1)), "already a texture"),
    ]:
        try:
            build()
            if expect is not None:
                raise AssertionError(f"expected an error mentioning {expect!r}")
        except cg.TechniqueError as e:
            assert expect in str(e), str(e)
    try:
        t.compile()
        raise AssertionError("expected an unknown-variable error")
    except cg.TechniqueError as e:
        assert "vars.missing" in str(e) and 'pass "p1"' in str(e), str(e)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
