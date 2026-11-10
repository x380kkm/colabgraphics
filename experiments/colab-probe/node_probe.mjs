// Runs in the Colab VM under Node, using the Dawn-based "webgpu" npm package:
// the same headless route Gigi's own WebGPU unit tests use ("node .").
import { create, globals } from 'webgpu';
Object.assign(globalThis, globals);

const out = {};
try {
  const gpu = create([]);
  const adapter = await gpu.requestAdapter();
  if (!adapter) {
    out.adapter = null;
  } else {
    const info = adapter.info || {};
    out.adapter = {
      vendor: info.vendor, architecture: info.architecture,
      device: info.device, description: info.description,
    };
    out.features = [...adapter.features].sort();
    const device = await adapter.requestDevice();
    device.pushErrorScope('validation');

    // 1024x1024 compute job writing a storage buffer, then readback.
    const N = 1024 * 1024;
    const t0 = performance.now();
    const module = device.createShaderModule({ code: `
      @group(0) @binding(0) var<storage, read_write> d: array<u32>;
      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) i: vec3<u32>) { d[i.x] = i.x * 3u + 1u; }` });
    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const storage = device.createBuffer({ size: N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readback = device.createBuffer({ size: N * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: storage } }],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup); pass.dispatchWorkgroups(N / 64); pass.end();
    encoder.copyBufferToBuffer(storage, 0, readback, 0, N * 4);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const data = new Uint32Array(readback.getMappedRange());
    out.computeOk = data[0] === 1 && data[5] === 16 && data[N - 1] === (N - 1) * 3 + 1;
    out.computeMs = Math.round(performance.now() - t0);
    const error = await device.popErrorScope();
    out.validationError = error ? error.message : null;
  }
} catch (e) {
  out.error = String(e);
}
console.log('RESULT ' + JSON.stringify(out));
process.exit(0);
