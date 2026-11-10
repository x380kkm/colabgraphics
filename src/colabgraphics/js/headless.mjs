// Run a compiled program without a browser: Node + Dawn (the "webgpu" npm package).
// Driven by headless.py, which writes a request file and reads the results back.
//
//   node headless.mjs <request.json>
//
// request: { program, frames, vars:{name:value}, read:[texture names], outDir,
//            webgpuModule: absolute path of the webgpu package's index.js,
//            flags: [dawn flags], compatibility: bool }
// Prints one line "RESULT {...}" and writes <outDir>/<texture>.bin for each texture read.
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { Runtime } from './runtime.js';

const request = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const out = { ok: false };
try {
  const { create, globals } = await import(pathToFileURL(request.webgpuModule).href);
  Object.assign(globalThis, globals);
  const gpu = create(request.flags || []);
  const adapter = await gpu.requestAdapter(request.compatibility ? { featureLevel: 'compatibility' } : {});
  if (!adapter) throw new Error('no GPU adapter: Dawn found no usable driver on this machine');
  out.adapter = adapter.info?.description || adapter.info?.vendor || 'unknown';
  const device = await adapter.requestDevice();

  const runtime = await new Runtime(device, request.program).init();
  const messages = await runtime.shaderMessages();
  out.shaderErrors = messages.filter((m) => m.type === 'error');
  out.validation = runtime.errors.map((e) => e.message);
  if (out.shaderErrors.length || out.validation.length) throw new Error('the program did not build');

  for (const [name, value] of Object.entries(request.vars || {})) runtime.setVar(name, value);
  device.pushErrorScope('validation');
  for (let i = 0; i < request.frames; i++) runtime.frame({ time: i / 60 });
  await device.queue.onSubmittedWorkDone();
  const runError = await device.popErrorScope();
  if (runError) { out.validation.push(runError.message); throw new Error('validation failed while running'); }

  out.textures = {};
  for (const name of request.read || []) {
    const t = await runtime.readTexture(name);
    fs.writeFileSync(path.join(request.outDir, name + '.bin'), t.bytes);
    out.textures[name] = { width: t.width, height: t.height, format: t.format, bytesPerPixel: t.bytesPerPixel };
  }
  out.frames = runtime.frameIndex;
  out.ok = true;
} catch (e) {
  out.error = String(e && e.message || e);
}
console.log('RESULT ' + JSON.stringify(out));
process.exit(0);
