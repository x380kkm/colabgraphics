# colabgraphics

在 Google Colab 里学习和原型化实时图形算法的环境。

它是一个 Colab 原生的项目，**设计上借鉴** [Gigi](https://github.com/electronicarts/gigi)
（EA 的渲染技术快速原型工具）的交互体验和编译器做法，但不运行 Gigi 本身，也不依赖它的编译器：
作者只描述"有哪些资源、哪些变量、依次跑哪些 pass"并写着色器的算法部分，
资源创建、绑定、管线、每帧调度、变量界面这些样板由系统生成。
描述和着色器写在 notebook 单元格里，渲染用 WebGPU 跑在读者自己的浏览器里。

当前阶段：**平台可行性验证完成，原生项目的设计进行中**。仓库里目前只有验证用的实验代码。

![一个 WebGPU 技术在未登录、未连接运行时的 Colab 里渲染](docs/images/gigi_boxblur_in_colab_no_runtime.jpg)

上图是真实的 Colab 页面：没有登录，没有连接运行时，单元格的输出是保存在 notebook 文件里的一段自包含 HTML，
其中的技术用 WebGPU 实时渲染。验证平台时用的负载是 Gigi 生成的 boxblur 示例，它在这里只是替身。

## 架构

| 部件 | 在哪里运行 | 作用 |
|---|---|---|
| 技术描述 | notebook 单元格 | 用 Python 声明资源、变量和 pass；着色器用 WGSL 只写算法部分 |
| 编译器 | Colab VM（Python） | 校验描述，推导尺寸与格式，为着色器生成绑定声明和变量结构体，产出给运行时执行的程序 |
| 运行时 | 浏览器，以及 VM 里的无头环境 | 同一份程序在读者的 GPU 上实时运行，也能在 VM 里无头运行以便同步读回、断言和 CI |
| 快照查看器 | 读者的浏览器 | `show()` 输出的自包含 HTML，存进 notebook，重开后不需要运行时 |
| 工作台 | 浏览器 ↔ VM | Gradio 式路径（VM 里的 HTTP 服务 + iframe）：改着色器、重编译、调试 |
| 调试层 | 运行时内 | 每个 pass 之后的资源快照、像素探针、计时、校验信息回指到单元格 |

编译器、运行时和调试层尚未实现；上表是设计方向，细节以后续提交为准。

## 已经实测的事实

Colab 前端（匿名打开真实 Colab，见 `experiments/colab-automation`）：

- 保存在 notebook 里的输出，重新打开时其中的 JavaScript 会再次执行；此时 `google.colab.kernel.accessAllowed` 为 `false`。
- WebGPU 在输出 iframe 里可用。
- 同一页面里所有单元格的输出 iframe 同源，可以互相调用；该域名每次打开页面都会变。
- 打开 notebook 时所有输出 iframe 立即创建，但只有可见的会收到动画帧。
- 单个 24 MB 的 HTML 输出可以渲染。同时带 HTML 和 PNG 的输出，Colab 显示 HTML。
- 输出 iframe 的 sandbox 不含 `allow-modals`，`allow` 为空，不能全屏；允许指针锁定和下载。

Colab VM（官方 [backend-info](https://github.com/googlecolab/backend-info)，2026-11-09）：

- Ubuntu 24.04.5，Python 3.13.18；gradio、anywidget 预装。
- apt 和 pip 清单里没有 Node，也没有 Vulkan 驱动。

无头运行 WebGPU（无 GPU 的 Linux，Dawn 的 Node 绑定，见 `experiments/dawn-node`）：

- 默认找不到 Vulkan 驱动；指向 SwiftShader 后可用，得到完整的 WebGPU 核心功能。
- 不装任何 Vulkan 驱动也行：Dawn 的 OpenGL ES 后端可以跑在 Mesa 的软件渲染上（需要让 `libEGL.so` 这个名字能被找到），
  得到的是 WebGPU 兼容模式，功能少一些。Colab 镜像预装了所需的 Mesa 包；这一点还没在 Colab 的 VM 里实测。
- 作为负载，Gigi 自带的 50 个 WebGPU 测试中 47 个跑完；12 个与 gold 逐位一致，22 个最多差 2/255，
  13 个只在边缘像素有差异，3 个光追测试超时。所以跨实现的结果比较必须用容差。
- Node 可以用 pip 装（PyPI 上的 `nodejs-wheel-binaries`）。

尚未在真实 Colab 里测过的：需要登录才能执行的部分，即内核与浏览器之间的通道
（`eval_js`、comm、端口代理）。`experiments/colab-probe/colab_probe.ipynb` 是为此准备的探针。

## 目录

```
experiments/
  colab-probe/        在 Colab 里手动运行的探针 notebook，及其本地排练
  colab-automation/   用 Playwright 匿名驱动真实 Colab；快照打包器 gigi_snapshot.py
  gradio-path/        Gradio 式路径的两个演示
  dawn-node/          用 Node + Dawn 无头运行 Gigi 的 WebGPU 测试
docs/images/          截图
```

## 复现实验

需要 Node、Playwright 自带的 Chromium，以及一份 Gigi 的检出（下面记作 `$GIGI`）。
WebGPU 在无 GPU 的机器上由 Chromium 自带的 SwiftShader 提供。

```bash
# 探针 notebook 的本地排练
cd experiments/colab-probe && python3 build.py && node test_local.cjs && python3 test_cells.py

# 匿名 Colab：保存的输出是否重新执行、WebGPU 是否可用
cd experiments/colab-automation && python3 build_saved_probe.py
node anon_saved.cjs saved_probe.ipynb /tmp/colab.png

# 把 Gigi 生成的应用打包成一段自包含输出，并放进真实 Colab
python3 gigi_snapshot.py $GIGI/_GeneratedCode/UnitTests/WebGPU UnitTests/Compute/boxblur /tmp/boxblur.html
node anon_gigi.cjs /tmp/boxblur.html /tmp/colab_gigi.png

# 用 Node + Dawn 无头运行 Gigi 的 WebGPU 测试（先在测试根目录 npm install webgpu pngjs）
python3 experiments/dawn-node/run_gigi_tests.py $GIGI/_GeneratedCode/UnitTests/WebGPU \
    /path/to/chrome-linux64/vk_swiftshader_icd.json
```

`anon_*.cjs` 的做法：打开 Colab 的 GitHub notebook 地址，在浏览器里拦截它取 `.ipynb` 的请求，
用本地文件作答。这样不需要发布任何东西，也不需要登录。
