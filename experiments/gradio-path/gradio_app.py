"""Gradio as the shell around a Gigi-generated WebGPU app.

Rendering stays in the browser (the Gigi page runs WebGPU on the viewer's GPU).
Gradio provides the Python-defined controls and serves the generated files.

Usage: gradio_app.py <generated WebGPU root> <port>
"""
import sys

import gradio as gr

ROOT, PORT = sys.argv[1], int(sys.argv[2])

# Let Gradio's own web server hand out the generated folder (HTML, JS modules, PNGs).
gr.set_static_paths(paths=[ROOT])
APP_URL = f"/gradio_api/file={ROOT}/UnitTests/Compute/boxblur/index.html"

# Runs in the browser only: push the control values into the Gigi page's variable inputs.
# The Gigi page is same-origin with the Gradio page, so its DOM is reachable.
PUSH_VARIABLES_JS = """
(radius, enabled, srgb) => {
  const w = document.getElementById('gigi-view').contentWindow;
  const set = (id, prop, value) => {
    const el = w.document.getElementById(id);
    if (!el) return;
    el[prop] = value;
    el.dispatchEvent(new Event('change'));
  };
  set('inputBox_radius', 'value', radius);
  set('inputBox_enabled', 'checked', enabled);
  set('inputBox_sRGB', 'checked', srgb);
}
"""


def describe(radius):
    side = 2 * int(radius) + 1
    return f"radius={int(radius)} → {side}×{side} box, {side * 2} taps per pixel (separable)"


with gr.Blocks(title="Gigi boxblur") as demo:
    gr.Markdown("### Gigi boxblur · WebGPU in the browser, Gradio as the shell")
    with gr.Row():
        with gr.Column(scale=3):
            gr.HTML(f'<iframe id="gigi-view" src="{APP_URL}" style="width:100%;height:700px;border:0"></iframe>')
        with gr.Column(scale=1):
            radius = gr.Slider(0, 50, value=20, step=1, label="radius")
            enabled = gr.Checkbox(True, label="enabled")
            srgb = gr.Checkbox(True, label="sRGB")
            note = gr.Textbox(label="computed in Python", interactive=False, value=describe(20))
    for control in (radius, enabled, srgb):
        # fn=None: no round trip to the kernel at all for these.
        control.change(fn=None, inputs=[radius, enabled, srgb], js=PUSH_VARIABLES_JS)
    # And one ordinary Python callback, to show both kinds side by side.
    radius.change(fn=describe, inputs=radius, outputs=note)

demo.launch(server_name="127.0.0.1", server_port=PORT, share=False, inbrowser=False)
