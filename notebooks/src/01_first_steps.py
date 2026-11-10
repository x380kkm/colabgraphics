# %% [markdown]
# # 第一步：像素着色器、变量、三角形、两个 pass
#
# 这本 notebook 里的画面都是活的：它们随文件一起保存，打开时就在你的浏览器里用 WebGPU 渲染，
# **不需要连接运行时**。想改代码再连。
#
# 写法只有两部分：用 Python 说明"有哪些纹理、哪些变量、依次跑哪些 pass"，用 HLSL 写每个 pass 的着色器。
# 创建资源、绑定、管线、每帧调度这些，由系统生成。
#
# 着色器语言用 HLSL，因为它是工业界写着色器最通用的语言。浏览器里的 WebGPU 只认 WGSL，
# 中间由 Slang 编译器翻译，这和 EA 的原型工具 Gigi 输出 WebGPU 时的做法相同。

# %%
%pip install -q git+https://github.com/x380kkm/colabgraphics@dev
import numpy as np
import colabgraphics as cg

# %% [markdown]
# ## 第一个着色器
#
# 像素着色器是一个函数：系统对画面上的每个像素调用它一次，它返回这个像素的颜色。
# `fullscreen()` 的意思就是"对一张纹理的每个像素跑一遍这个函数"。
#
# 参数 `uv` 是像素在画面里的位置，左上角是 (0, 0)，右下角是 (1, 1)。冒号后面的 `TEXCOORD0`、`SV_Target` 叫语义，
# 说明这个值从哪来、到哪去：`SV_Target` 表示返回值写进渲染目标。
#
# 把鼠标停在画面上可以读到那个像素的数值，可以拿来核对：红色分量是 `uv.x`，也就是（像素的横坐标 + 0.5）/ 384，
# 绿色分量同理是纵坐标。

# %%
first = cg.Technique("first")
color = first.texture("color", size=(384, 384))
first.fullscreen("shade", """
float4 PSMain(float2 uv : TEXCOORD0) : SV_Target
{
    return float4(uv, 0, 1);
}
""", targets=[color])
first.show()

# %% [markdown]
# ## 变量
#
# `var()` 声明一个可以在运行时改动的值。着色器里用 `vars.名字` 读它，画面旁边会自动出现对应的控件。
# 下面的 `sys.time` 是系统提供的时间（秒），所以画面会动。

# %%
RINGS = """
float4 PSMain(float2 uv : TEXCOORD0) : SV_Target
{
    float2 p = uv * 2.0 - 1.0;                       // 把原点移到画面中心
    float wave = cos(length(p) * vars.frequency - sys.time * vars.speed);
    float3 background = float3(0.05, 0.05, 0.1);
    return float4(lerp(background, vars.tint, 0.5 + 0.5 * wave), 1);
}
"""

rings = cg.Technique("rings")
color = rings.texture("color", size=(384, 384))
rings.var("frequency", 24.0, min=1.0, max=80.0, doc="每单位长度的弧度数")
rings.var("speed", 2.0, min=-8.0, max=8.0, doc="波向外走的速度，弧度每秒")
rings.var("tint", (1.0, 0.8, 0.2), color=True)
rings.fullscreen("shade", RINGS, targets=[color])
rings.show()

# %% [markdown]
# ## 系统替你写了什么
#
# 着色器里用到的 `vars`、`sys` 都没有声明过，顶点着色器也没有写。编译时系统在你的代码前面补上了这些。
# 这不是黑箱，随时可以看：

# %%
program = rings.compile()
print(program.declarations("shade"))

# %% [markdown]
# 编译的结果也可以看。下面是上面那个像素着色器被翻译成的 WGSL，浏览器实际运行的就是它：

# %%
print(program.wgsl("shade", stage="pixel"))

# %% [markdown]
# ## 自己写顶点着色器：一个三角形
#
# `draw()` 画几何体，这时顶点着色器和像素着色器都由你写：`VSMain` 对每个顶点运行一次，给出它在画面上的位置
# （`SV_Position`），`PSMain` 对三角形盖住的每个像素运行一次。顶点着色器的其他输出（这里是 `color`）
# 会在三角形内部插值后交给像素着色器。
#
# 顶点位置直接写在着色器里，用顶点编号 `SV_VertexID` 去取，所以连顶点缓冲都不需要。

# %%
TRIANGLE = """
struct VSOut
{
    float4 position : SV_Position;
    float3 color    : COLOR0;
};

VSOut VSMain(uint id : SV_VertexID)
{
    static const float2 corners[3] = { float2(0.0, 0.8), float2(-0.8, -0.8), float2(0.8, -0.8) };
    static const float3 colors[3]  = { float3(1, 0.3, 0.2), float3(0.2, 1, 0.4), float3(0.3, 0.4, 1) };
    float a = sys.time * vars.speed;
    float2 p = corners[id] * vars.scale;
    VSOut o;
    o.position = float4(p.x * cos(a) - p.y * sin(a), p.x * sin(a) + p.y * cos(a), 0, 1);
    o.color = colors[id];
    return o;
}

float4 PSMain(VSOut input) : SV_Target
{
    return float4(input.color, 1);
}
"""

tri = cg.Technique("triangle")
color = tri.texture("color", size=(384, 384))
tri.var("speed", 1.0, min=-4.0, max=4.0, doc="转速，弧度每秒")
tri.var("scale", 0.8, min=0.1, max=1.2)
tri.draw("tri", TRIANGLE, targets=[color], vertices=3)
tri.show()

# %% [markdown]
# ## 两个 pass：可分离的 box blur
#
# 先横向模糊到一张中间纹理，再纵向模糊到结果。同一段着色器代码用两次，只是方向不同。
# 这次用 compute pass：`CSMain` 对输出纹理的每个像素运行一次，`[numthreads(8, 8, 1)]` 是每组线程的大小。
# 在着色器里，`image` 和 `result` 这两个名字由 `reads=` 和 `writes=` 接到具体的纹理上：
# 读的是 `Texture2D`，写的是 `RWTexture2D`。
#
# 画面下方那一行是这个技术的流程。点里面的纹理名，可以看任意一步的中间结果。

# %%
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

blur = cg.Technique("boxblur")
src = blur.image("src", picture)
tmp = blur.texture("tmp", like=src)
out = blur.texture("out", like=src)
blur.var("radius", 6, min=0, max=32, doc="模糊半径，像素")

blur.compute("blur_h", "static const int2 DIR = int2(1, 0);" + BLUR, reads={"image": src}, writes={"result": tmp})
blur.compute("blur_v", "static const int2 DIR = int2(0, 1);" + BLUR, reads={"image": tmp}, writes={"result": out})
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

# %% [markdown]
# ## 写错了会怎样
#
# 着色器在 Python 这一侧编译，所以错误在运行这一格时就报出来，指明是哪个 pass 的第几行（从那段着色器代码的开头数）。
# 下面故意把 `radius` 拼错：

# %%
broken = cg.Technique("broken")
a = broken.image("a", picture)
b = broken.texture("b", like=a)
broken.var("radius", 6)
broken.compute("blur_h", "static const int2 DIR = int2(1, 0);" + BLUR.replace("2 * vars.radius", "2 * vars.raduis"),
               reads={"image": a}, writes={"result": b})
try:
    broken.compile()
except cg.ShaderError as error:
    print(error)
