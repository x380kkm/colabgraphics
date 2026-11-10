import numpy as np
import colabgraphics as cg

# A test picture made with numpy: coloured waves with a sharp grid on top.
y, x = np.mgrid[0:384, 0:512]
waves = np.stack([np.sin(x / 31 + y / 47), np.sin(x / 23 - y / 19 + 2), np.sin((x + y) / 37 + 4)], -1) * 0.5 + 0.5
grid = ((x // 32 + y // 32) % 2 == 0)[..., None] * 0.35
picture = (np.clip(waves * (0.65 + grid), 0, 1) * 255).astype(np.uint8)

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

t = cg.Technique("boxblur")
src = t.image("src", picture)
tmp = t.texture("tmp", like=src)
out = t.texture("out", like=src)
t.var("radius", 6, min=0, max=32, doc="half width of the box, in pixels")

t.compute("blur_h", "const DIR = vec2i(1, 0);" + BLUR, reads={"image": src}, writes={"result": tmp})
t.compute("blur_v", "const DIR = vec2i(0, 1);" + BLUR, reads={"image": tmp}, writes={"result": out})

t.show(out)
