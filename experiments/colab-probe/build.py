#!/usr/bin/env python3
"""Assemble colab_probe.ipynb from the snippets in this folder.

The browser-side snippets live in js/ so that test_local.cjs can run the very
same text in a local Chromium before the notebook ever reaches Colab.
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent


def read(name):
    text = (HERE / name).read_text()
    assert "'''" not in text, f"{name} must not contain triple single quotes"
    return text


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


def raw(name, text):
    """A Python raw-string assignment holding a snippet verbatim."""
    return f"{name} = r'''\n{text.rstrip()}\n'''\n"


cells = []

cells.append(md("""
# Gigi × Colab：环境探针

这个 notebook 不做任何渲染工作，只回答一组**只有在真实 Colab 里才能回答**的问题，
它们决定 Gigi 的 Colab 版本该怎么设计：

| 探针 | 回答的问题 |
|---|---|
| kernel | VM 里有什么：Python、glibc、Node、Vulkan 驱动、GPU |
| frame | 输出 iframe 里 WebGPU 能不能用，用的是哪块显卡 |
| sibling_frames | 一个单元格的输出能否访问另一个单元格的输出 |
| comm | 浏览器 → 内核的异步通道（comm、invokeFunction）及其二进制载荷 |
| eval_js | 单元格执行期间的同步往返：延迟、两个方向各能传多大 |
| kernel_http | 内核里的后台 HTTP 服务能否在单元格执行期间被浏览器访问 |
| node_dawn | 内核里能否用 Node + Dawn 无头运行 WebGPU（Gigi 自己的测试路径） |
| persistence | 保存并重新打开后，输出里的 WebGPU 画面是否还活着（需要人工看一眼） |

**用法**：代码执行程序 → 全部运行（约 1–2 分钟），然后把最后一个单元格打印的 JSON 贴回来。
报告里包含浏览器 User-Agent 和显卡型号；域名里的会话 ID 已被替换成 `<id>`。

`eval_js` 探针会发送最大 16 MB 的消息来找上限。如果连接因此断开，这本身就是结果：
重新连接后运行最后一个单元格即可，已完成的部分保存在 VM 的 `/content/gigi_colab_probe.json`。
"""))

cells.append(code("""
import glob, json, os, platform, shutil, subprocess, sys, time
from importlib import metadata

REPORT = {"probe_version": 1}
WORKDIR = "/content" if os.path.isdir("/content") else os.getcwd()
REPORT_PATH = os.path.join(WORKDIR, "gigi_colab_probe.json")


def save():
    with open(REPORT_PATH, "w") as f:
        json.dump(REPORT, f, indent=1, default=str)


def sh(cmd, timeout=30, limit=600):
    \"\"\"Run a shell command; return the tail of its output, or the error.\"\"\"
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (p.stdout + p.stderr).strip()[-limit:]
    except Exception as e:
        return f"<{type(e).__name__}: {e}>"


def version(name):
    try:
        return metadata.version(name)
    except Exception:
        return None


def probe(section, fn):
    \"\"\"Run one probe, store its result (or its error) in REPORT, and print it.\"\"\"
    t0 = time.time()
    try:
        REPORT[section] = fn()
    except Exception as e:
        REPORT[section] = {"error": f"{type(e).__name__}: {e}"}
    REPORT[section + "_seconds"] = round(time.time() - t0, 2)
    save()
    print(json.dumps(REPORT[section], indent=1, default=str)[:4000])


def kernel_facts():
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "libc": " ".join(platform.libc_ver()),
        "os": sh("grep PRETTY_NAME /etc/os-release"),
        "cpus": os.cpu_count(),
        "mem_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1),
        "disk_free_gb": round(shutil.disk_usage(WORKDIR).free / 2**30, 1),
        "colab_release": os.environ.get("COLAB_RELEASE_TAG"),
        "gpu": sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader"),
        "node": sh("node --version"),
        "npm": sh("npm --version"),
        "gcc": sh("gcc --version | head -1"),
        "cmake": sh("cmake --version | head -1"),
        "graphviz": sh("dot -V"),
        "vulkan_icd_files": sorted(glob.glob("/usr/share/vulkan/icd.d/*") + glob.glob("/etc/vulkan/icd.d/*")),
        "vulkan_libs": sh("ldconfig -p | grep -i -E 'libvulkan|libEGL' | head -6"),
        "packages": {n: version(n) for n in [
            "ipykernel", "ipython", "ipywidgets", "anywidget", "tornado",
            "jupyter-server", "numpy", "pillow", "jsonschema", "nbformat"]},
    }


probe("kernel", kernel_facts)
"""))

cells.append(md("""
## 两个"先放着"的输出

下面两个单元格各自创建一个输出 iframe，后面的探针要用到它们：

- **frame A**：一个带标记的输出，稍后从*另一个*单元格尝试访问它。
- **comm 探针**：在浏览器里发起 comm / invokeFunction。这两条通道走内核的 shell 通道，
  内核只在**两次单元格执行之间**处理它们，所以结果要等后面的单元格跑起来才会陆续到达。
"""))

cells.append(code(
    "from IPython.display import HTML, display\n\n"
    + raw("FRAME_A_HTML", read("js/frame_a.html"))
    + "\ndisplay(HTML(FRAME_A_HTML))\n"
))

cells.append(code(
    "import IPython\nfrom google.colab import output\n\n"
    + raw("COMM_HTML", read("js/comm.html"))
    + '''
COMM_LOG = []  # what the kernel received on the comm


def _on_comm_open(comm, open_msg):
    COMM_LOG.append({"event": "open", "buffers": [len(b) for b in open_msg.get("buffers", [])]})

    def on_msg(msg):
        data = msg["content"].get("data") or {}
        if "final" in data:  # the browser's own summary of the whole exchange
            REPORT["comm"] = data["final"]
            save()
            return
        COMM_LOG.append({"event": "msg", "buffers": [len(b) for b in msg.get("buffers", [])]})
        n = int(data.get("echo_bytes", 0))
        comm.send({"echo": n}, None, [bytes(n)])

    comm.on_msg(on_msg)


IPython.get_ipython().kernel.comm_manager.register_target("gigi_probe_comm", _on_comm_open)
output.register_callback("gigi_probe_callback", lambda *args, **kwargs: {"args": list(args), "kwargs": kwargs})
REPORT["comm"] = "no final message from the browser yet"
display(HTML(COMM_HTML))
'''
))

cells.append(md("""
## 输出 iframe 里有什么

`eval_js` 在**当前单元格**的输出 iframe 里执行一段 JS，并阻塞 Python 直到拿到返回值。
"""))

cells.append(code(
    raw("FRAME_JS", read("js/frame.js"))
    + '\nprobe("frame", lambda: output.eval_js(FRAME_JS, timeout_sec=90))\n'
))

cells.append(md("""
## 跨单元格访问

这段 JS 运行在*本单元格*的 iframe 里，目标是上面 frame A 所在的*另一个* iframe。
"""))

cells.append(code(
    raw("SIBLING_JS", read("js/sibling.js"))
    + '\nprobe("sibling_frames", lambda: output.eval_js(SIBLING_JS, timeout_sec=30))\n'
))

cells.append(md("""
## `eval_js` 的延迟与载荷上限

Colab 自己的上传代码把浏览器 → 内核的数据切成 100 KB 一块，下载代码按 1 MB 一块。
这里直接量一下两个方向各能走多大。
"""))

cells.append(code('''
def eval_js_limits():
    r = {}
    times = []
    for _ in range(20):
        t = time.perf_counter()
        output.eval_js("1", timeout_sec=10)
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    r["roundtrip_ms"] = {"min": round(times[0], 1), "median": round(times[10], 1), "max": round(times[-1], 1)}

    for direction in ("kernel_to_browser", "browser_to_kernel"):
        r[direction] = []
        for n in (100_000, 1_000_000, 4_000_000, 16_000_000):
            # Saved first, so a dropped connection still leaves a trace of where it happened.
            REPORT["eval_js_in_progress"] = {"direction": direction, "bytes": n, "so_far": r}
            save()
            t = time.perf_counter()
            step = {"bytes": n}
            try:
                if direction == "kernel_to_browser":
                    got = output.eval_js('("' + "x" * n + '").length', timeout_sec=60)
                    step["ok"] = got == n
                else:
                    got = output.eval_js(f'"x".repeat({n})', timeout_sec=60)
                    step["ok"] = isinstance(got, str) and len(got) == n
            except Exception as e:
                step["ok"] = False
                step["error"] = f"{type(e).__name__}: {e}"[:200]
            step["seconds"] = round(time.perf_counter() - t, 2)
            r[direction].append(step)
            if not step["ok"]:
                break
    REPORT.pop("eval_js_in_progress", None)
    return r


probe("eval_js", eval_js_limits)
'''))

cells.append(md("""
## 内核里的后台 HTTP 服务

服务跑在内核进程的一个后台线程里。浏览器通过 Colab 的端口代理访问它，
而此时主线程正卡在 `eval_js` 上等结果，也就是"单元格执行期间"。
"""))

cells.append(code(
    read("kernel_server.py").rstrip() + "\n\n\n"
    + raw("HTTP_JS", read("js/http.js"))
    + '''
_server, PORT = start_probe_server()


def kernel_http():
    browser = output.eval_js(HTTP_JS.replace("__PORT__", str(PORT)), timeout_sec=180)
    return {"port": PORT, "browser": browser, "kernel_saw": HTTP_LOG[:20]}


probe("kernel_http", kernel_http)
'''
))

cells.append(md("""
## 内核里的无头 WebGPU：Node + Dawn

Gigi 的 WebGPU 单元测试就是这样跑的（`npm install webgpu` 后 `node .`）。
没有 GPU 的 VM 需要一个软件 Vulkan 驱动；如果第一次找不到 adapter，就装 Mesa 的 lavapipe 再试。
"""))

cells.append(code(
    raw("NODE_PROBE", read("node_probe.mjs"))
    + '''

def node_dawn():
    r = {"node": sh("node --version"), "npm": sh("npm --version")}
    if not r["node"].startswith("v"):
        r["note"] = "no Node.js in this VM"
        return r
    d = os.path.join(WORKDIR, "gigi_probe_node")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "probe.mjs"), "w") as f:
        f.write(NODE_PROBE)
    t = time.time()
    r["npm_install"] = sh(f"cd {d} && (npm init -y >/dev/null 2>&1; npm install webgpu 2>&1 | tail -3)", timeout=600)
    r["npm_install_seconds"] = round(time.time() - t, 1)
    r["package_mb"] = sh(f"du -sm {d}/node_modules/webgpu | cut -f1")

    def run():
        text = sh(f"cd {d} && node probe.mjs 2>&1 | tail -12", timeout=180, limit=20000)
        for line in text.splitlines():
            if line.startswith("RESULT "):
                return json.loads(line[len("RESULT "):])
        return {"raw": text[-1500:]}

    r["first_run"] = run()
    if not r["first_run"].get("adapter"):
        t = time.time()
        install = "apt-get install -y -qq mesa-vulkan-drivers"
        r["apt_install"] = sh(f"({install} || (apt-get update -qq && {install})) 2>&1 | tail -3", timeout=900)
        r["apt_install_seconds"] = round(time.time() - t, 1)
        r["vulkan_icd_files"] = sorted(glob.glob("/usr/share/vulkan/icd.d/*"))
        r["after_lavapipe"] = run()
    return r


probe("node_dawn", node_dawn)
'''
))

cells.append(md("""
## 持久化：保存后重开，画面还活着吗

下面这个输出完全自包含，不依赖内核。运行后应当看到一块会动的彩色画面。

**人工检查**（这一步脚本测不了）：

1. 全部运行结束后保存（Ctrl/Cmd + S），等标题栏显示已保存。
2. 刷新页面，**不要**连接运行时。
3. 滚动到这个单元格：画面还在动吗？下面那行字里 `kernel access` 显示什么？

把观察到的结果和 JSON 一起告诉我即可。
"""))

cells.append(code(
    raw("PERSIST_HTML", read("js/persist.html"))
    + '\ndisplay(HTML(PERSIST_HTML.replace("__EMITTED_AT__", str(int(time.time())))))\n'
))

cells.append(md("""
## 报告

如果 `comm` 一项还显示 "no final message"，等几秒后单独重跑这个单元格
（comm 消息只有在内核空闲时才会被处理）。
"""))

cells.append(code('''
REPORT["comm_kernel_log"] = COMM_LOG
save()
print(json.dumps(REPORT, indent=1, default=str))
'''))

notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "cells": cells,
}

out = HERE / "colab_probe.ipynb"
out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(f"wrote {out} ({out.stat().st_size} bytes, {len(cells)} cells)")
