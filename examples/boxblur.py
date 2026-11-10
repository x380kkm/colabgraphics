import numpy as np
import colabgraphics as cg

# A test picture made with numpy: coloured waves with a sharp grid on top.
y, x = np.mgrid[0:384, 0:512]
waves = np.stack([np.sin(x / 31 + y / 47), np.sin(x / 23 - y / 19 + 2), np.sin((x + y) / 37 + 4)], -1) * 0.5 + 0.5
grid = ((x // 32 + y // 32) % 2 == 0)[..., None] * 0.35
picture = (np.clip(waves * (0.65 + grid), 0, 1) * 255).astype(np.uint8)

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

t = cg.Technique("boxblur")
src = t.image("src", picture)
tmp = t.texture("tmp", like=src)
out = t.texture("out", like=src)
t.var("radius", 6, min=0, max=32, doc="half width of the box, in pixels")

t.compute("blur_h", "static const int2 DIR = int2(1, 0);" + BLUR, reads={"image": src}, writes={"result": tmp})
t.compute("blur_v", "static const int2 DIR = int2(0, 1);" + BLUR, reads={"image": tmp}, writes={"result": out})

# In a notebook the last line displays the running technique; as a script, save a page instead.
t.show(out).save("boxblur.snapshot.html")
print("wrote boxblur.snapshot.html")
