// colabgraphics viewer: the reader-facing side of a running technique in a web page.
// It builds a small UI (canvas, a control per variable, a selector for which texture to
// look at) and drives a Runtime once per animation frame. Runtime comes from runtime.js;
// display.py concatenates the two files into one inline module.

function mountViewer(root, program, options = {}) {
  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  };
  root.classList.add('cg');
  const stage = root.appendChild(el('div', 'cg-stage'));
  const canvas = stage.appendChild(el('canvas'));
  const side = root.appendChild(el('div', 'cg-side'));
  const graph = root.appendChild(el('div', 'cg-graph'));
  const status = el('div', 'cg-status', 'waiting to become visible…');
  const errors = root.appendChild(el('pre', 'cg-errors'));
  errors.hidden = true;

  let viewName = options.view || program.view;
  const sizeCanvas = () => {
    const t = program.textures.find((x) => x.name === viewName);
    canvas.width = t.size[0];
    canvas.height = t.size[1];
  };
  sizeCanvas();

  // Which texture to look at.
  const viewRow = side.appendChild(el('label', 'cg-row'));
  viewRow.appendChild(el('span', 'cg-name', 'view'));
  const viewSelect = viewRow.appendChild(el('select'));
  for (const t of program.textures) {
    const o = viewSelect.appendChild(el('option', null, `${t.name}  (${t.size[0]}×${t.size[1]} ${t.format})`));
    o.value = t.name;
  }
  viewSelect.value = viewName;
  const chips = [];
  const setView = (name) => {
    viewName = name;
    viewSelect.value = name;
    sizeCanvas();
    for (const c of chips) c.classList.toggle('cg-current', c.dataset.texture === name);
  };
  viewSelect.addEventListener('change', () => setView(viewSelect.value));

  // The technique as a strip: each pass with what it reads and what it writes.
  // Click a texture to look at it.
  const chip = (name) => {
    const c = el('button', 'cg-chip', name);
    c.type = 'button';
    c.dataset.texture = name;
    c.title = 'show this texture';
    c.addEventListener('click', () => setView(name));
    chips.push(c);
    return c;
  };
  for (const pass of program.passes) {
    const node = graph.appendChild(el('div', 'cg-node'));
    const ins = node.appendChild(el('span', 'cg-io'));
    for (const b of pass.bindings) if (b.kind === 'texture') ins.appendChild(chip(b.texture));
    node.appendChild(el('span', 'cg-arrow', '→'));
    const box = node.appendChild(el('span', 'cg-pass', pass.name));
    box.title = pass.kind === 'compute' ? 'compute pass' : 'draw pass';
    node.appendChild(el('span', 'cg-arrow', '→'));
    const outs = node.appendChild(el('span', 'cg-io'));
    for (const b of pass.bindings) if (b.kind === 'storage') outs.appendChild(chip(b.texture));
    for (const t of pass.targets || []) outs.appendChild(chip(t.texture));
  }
  setView(viewName);

  // One control per variable. Changes apply on the next frame, with no round trip anywhere.
  const pending = new Map();
  const set = (name, value) => { pending.set(name, value); };
  const toHex = (v) => '#' + v.slice(0, 3).map((c) => Math.round(Math.min(Math.max(c, 0), 1) * 255).toString(16).padStart(2, '0')).join('');
  for (const f of program.vars.fields) {
    const row = side.appendChild(el('label', 'cg-row'));
    const name = row.appendChild(el('span', 'cg-name', f.name));
    if (f.doc) name.title = f.doc;
    if (f.ui === 'checkbox') {
      const box = row.appendChild(el('input'));
      box.type = 'checkbox';
      box.checked = !!f.value;
      box.addEventListener('change', () => set(f.name, box.checked ? 1 : 0));
    } else if (f.ui === 'slider') {
      const slider = row.appendChild(el('input'));
      slider.type = 'range';
      slider.min = f.min; slider.max = f.max;
      slider.step = f.step ?? (f.scalar === 'f32' ? (f.max - f.min) / 200 : 1);
      slider.value = f.value;
      const readout = row.appendChild(el('span', 'cg-readout', String(f.value)));
      slider.addEventListener('input', () => {
        const v = f.scalar === 'f32' ? parseFloat(slider.value) : parseInt(slider.value, 10);
        readout.textContent = f.scalar === 'f32' ? v.toFixed(3) : String(v);
        set(f.name, v);
      });
    } else if (f.ui === 'color') {
      const value = f.value.slice();
      const picker = row.appendChild(el('input'));
      picker.type = 'color';
      picker.value = toHex(value);
      picker.addEventListener('input', () => {
        for (let i = 0; i < 3; i++) value[i] = parseInt(picker.value.slice(1 + i * 2, 3 + i * 2), 16) / 255;
        set(f.name, value.slice());
      });
    } else {
      const value = Array.isArray(f.value) ? f.value.slice() : [f.value];
      value.forEach((v, i) => {
        const box = row.appendChild(el('input'));
        box.type = 'number';
        box.value = v;
        if (f.step != null) box.step = f.step; else if (f.scalar === 'f32') box.step = 'any';
        box.addEventListener('change', () => {
          value[i] = f.scalar === 'f32' ? parseFloat(box.value) : parseInt(box.value, 10);
          set(f.name, f.count === 1 ? value[0] : value.slice());
        });
      });
    }
  }
  side.appendChild(status);
  for (const w of options.warnings || []) side.appendChild(el('div', 'cg-warning', w));

  const mouse = [0, 0, 0, 0];
  const track = (e) => {
    const r = canvas.getBoundingClientRect();
    mouse[0] = (e.clientX - r.left) * canvas.width / r.width;
    mouse[1] = (e.clientY - r.top) * canvas.height / r.height;
  };
  canvas.addEventListener('pointermove', track);
  canvas.addEventListener('pointerdown', (e) => { track(e); mouse[2] = 1; });
  addEventListener('pointerup', () => { mouse[2] = 0; });

  const handle = { program, runtime: null, frames: 0, failed: null };
  const fail = (message) => {
    handle.failed = message;
    status.textContent = 'not running';
    errors.hidden = false;
    errors.textContent = message;
    if (options.fallback) {
      const img = el('img');
      img.src = options.fallback;
      img.alt = 'last rendered image';
      canvas.replaceWith(img);
    }
  };

  async function start() {
    if (!navigator.gpu) return fail('This browser does not provide WebGPU, so the technique cannot run here.');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return fail('WebGPU is present but no GPU adapter is available in this browser.');
    const device = await adapter.requestDevice();
    const context = canvas.getContext('webgpu');
    context.configure({ device, format: navigator.gpu.getPreferredCanvasFormat(), alphaMode: 'opaque' });
    const runtime = await new Runtime(device, program).init();
    handle.runtime = runtime;

    // Shader problems, reported against the author's own line numbers.
    const problems = (await runtime.shaderMessages()).filter((m) => m.type === 'error').map((m) => {
      const pass = program.passes.find((p) => p.name === m.pass);
      const line = m.line - pass.preambleLines;
      return `pass "${m.pass}", ${line > 0 ? 'line ' + line : 'generated declarations'}: ${m.message}`;
    });
    if (!problems.length) for (const e of runtime.errors) problems.push(e.message);
    if (problems.length) return fail(problems.join('\n'));
    device.lost.then((info) => { if (info.reason !== 'destroyed') fail('The GPU device was lost: ' + info.message); });

    status.textContent = `running · ${program.passes.length} passes`;
    const t0 = performance.now();
    let last = t0, smoothed = 0;
    const frame = (now) => {
      if (handle.failed) return;
      for (const [name, value] of pending) runtime.setVar(name, value);
      pending.clear();
      runtime.frame({ time: (now - t0) / 1000, mouse });
      runtime.present(context.getCurrentTexture(), viewName);
      handle.frames++;
      smoothed = smoothed ? smoothed * 0.9 + (now - last) * 0.1 : now - last;
      last = now;
      if (handle.frames % 20 === 0) status.textContent = `${(1000 / smoothed).toFixed(0)} fps · ${program.passes.length} passes · frame ${handle.frames}`;
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  }

  // Start on the first animation frame: pages that hold many outputs (a notebook) only
  // deliver animation frames to the visible ones, so off-screen outputs cost nothing.
  handle.ready = new Promise((resolve) => {
    requestAnimationFrame(() => start().catch((e) => fail(String(e && e.message || e))).then(() => resolve(handle)));
  });
  root.colabgraphics = handle;
  (self.colabgraphics = self.colabgraphics || []).push(handle);
  return handle;
}
