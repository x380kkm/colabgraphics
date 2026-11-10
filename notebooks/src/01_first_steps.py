# %% [markdown]
# # 第一步：三角形、变量、两个 pass
#
# 这本 notebook 里的画面都是活的：它们随文件一起保存，打开时就在你的浏览器里用 WebGPU 渲染，
# **不需要连接运行时**。想改代码再连。
#
# 写法只有两部分：用 Python 说明"有哪些纹理、哪些变量、依次跑哪些 pass"，用 WGSL 写每个 pass 的算法。
# 创建资源、绑定、管线、每帧调度这些，由系统生成。

# %%
%pip install -q git+https://github.com/x380kkm/colabgraphics@first-slice
import numpy as np
import colabgraphics as cg

# %% [markdown]
# ## 一个三角形
#
# 一张纹理，一个 draw pass。顶点位置直接写在顶点着色器里，所以连顶点缓冲都不需要。

# %%
TRIANGLE = """
struct Out { @builtin(position) pos: vec4f, @location(0) color: vec3f }

@vertex fn vs(@builtin(vertex_index) i: u32) -> Out {
  let corners = array(vec2f(0.0, 0.8), vec2f(-0.8, -0.8), vec2f(0.8, -0.8));
  let colors = array(vec3f(1, 0.3, 0.2), vec3f(0.2, 1, 0.4), vec3f(0.3, 0.4, 1));
  return Out(vec4f(corners[i], 0, 1), colors[i]);
}

@fragment fn fs(in: Out) -> @location(0) vec4f {
  return vec4f(in.color, 1);
}
"""

tri = cg.Technique("triangle")
color = tri.texture("color", size=(384, 384))
tri.draw("tri", TRIANGLE, targets=[color], vertices=3)
tri.show()

# %% [markdown]
# ## 变量
#
# `var()` 声明一个可以在运行时改动的值。着色器里用 `vars.名字` 读它，画面旁边会自动出现对应的控件。
# 下面的 `sys.time` 是系统提供的时间（秒）。

# %%
SPIN = """
struct Out { @builtin(position) pos: vec4f, @location(0) color: vec3f }

@vertex fn vs(@builtin(vertex_index) i: u32) -> Out {
  let corners = array(vec2f(0.0, 0.8), vec2f(-0.8, -0.8), vec2f(0.8, -0.8));
  let colors = array(vec3f(1, 0.3, 0.2), vec3f(0.2, 1, 0.4), vec3f(0.3, 0.4, 1));
  let a = sys.time * vars.speed;
  let p = corners[i] * vars.scale;
  let rotated = vec2f(p.x * cos(a) - p.y * sin(a), p.x * sin(a) + p.y * cos(a));
  return Out(vec4f(rotated, 0, 1), colors[i]);
}

@fragment fn fs(in: Out) -> @location(0) vec4f {
  return vec4f(mix(in.color, vars.tint.rgb, vars.tint_amount), 1);
}
"""

spin = cg.Technique("spinning")
color = spin.texture("color", size=(384, 384))
spin.var("speed", 1.0, min=-4.0, max=4.0, doc="转速，弧度每秒")
spin.var("scale", 0.8, min=0.1, max=1.2)
spin.var("tint", (1.0, 0.8, 0.2), color=True)
spin.var("tint_amount", 0.0, min=0.0, max=1.0)
spin.draw("tri", SPIN, targets=[color], vertices=3)
spin.show()

# %% [markdown]
# ## 系统替你写了什么
#
# 着色器里用到的 `vars`、`sys` 都没有声明过。编译时系统在你的代码前面补上了声明。
# 这不是黑箱，随时可以看：

# %%
print(spin.compile().wgsl("tri").split("// ---- your code ----")[0])

# %% [markdown]
# ## 两个 pass：可分离的 box blur
#
# 先横向模糊到一张中间纹理，再纵向模糊到结果。同一段着色器代码用两次，只是方向不同。
# 在着色器里，`image` 和 `result` 这两个名字由 `reads=` 和 `writes=` 接到具体的纹理上。
#
# 画面下方那一行是这个技术的流程。点里面的纹理名，可以看任意一步的中间结果。

# %%
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

blur = cg.Technique("boxblur")
src = blur.image("src", picture)
tmp = blur.texture("tmp", like=src)
out = blur.texture("out", like=src)
blur.var("radius", 6, min=0, max=32, doc="模糊半径，像素")

blur.compute("blur_h", "const DIR = vec2i(1, 0);" + BLUR, reads={"image": src}, writes={"result": tmp})
blur.compute("blur_v", "const DIR = vec2i(0, 1);" + BLUR, reads={"image": tmp}, writes={"result": out})
blur.show()

# %% [markdown]
# ## 用 numpy 核对结果
#
# `read()` 在运行时这一侧无头执行同一个技术，把纹理读回成 numpy 数组。这样 GPU 上的结果可以和
# 一份用 numpy 写的参考实现逐像素比较。（这一格需要连接运行时才能重新执行。）

# %%
def reference(image, radius):
    def one_pass(a, axis):
        a = a.astype(np.float32) / 255
        n = a.shape[axis]
        acc = sum(np.take(a, np.clip(np.arange(n) + i, 0, n - 1), axis=axis) for i in range(-radius, radius + 1))
        return np.rint(acc / (2 * radius + 1) * 255).astype(np.uint8)
    return one_pass(one_pass(image, 1), 0)

rgba = np.dstack([picture, np.full(picture.shape[:2], 255, np.uint8)])
for radius in (0, 4, 12):
    gpu = blur.read(out, radius=radius)
    worst = np.abs(gpu.astype(int) - reference(rgba, radius).astype(int)).max()
    print(f"radius {radius:2d}: GPU 与 numpy 的最大差 = {worst}/255")
