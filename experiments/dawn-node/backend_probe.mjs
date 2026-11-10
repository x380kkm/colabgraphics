// Which Dawn backends yield an adapter on this machine with no Vulkan driver configured?
import { create, globals } from 'webgpu';
Object.assign(globalThis, globals);
const backend = process.argv[2];
const out = { backend };
try {
  const gpu = create(backend ? ['backend=' + backend] : []);
  const adapter = await gpu.requestAdapter(backend === 'opengles' ? { featureLevel: 'compatibility' } : {});
  if (!adapter) { out.adapter = null; }
  else {
    out.adapter = { vendor: adapter.info?.vendor, architecture: adapter.info?.architecture, description: adapter.info?.description };
    out.features = [...adapter.features].sort();
    const device = await adapter.requestDevice();
    const module = device.createShaderModule({ code: `
      @group(0) @binding(0) var<storage, read_write> d: array<u32>;
      @compute @workgroup_size(4) fn main(@builtin(global_invocation_id) i: vec3<u32>) { d[i.x] = i.x * i.x + 1u; }` });
    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const buf = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const rb = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const bg = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: buf } }] });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(1); pass.end();
    enc.copyBufferToBuffer(buf, 0, rb, 0, 16);
    device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    out.compute = Array.from(new Uint32Array(rb.getMappedRange()));
  }
} catch (e) { out.error = String(e).slice(0, 300); }
console.log('RESULT ' + JSON.stringify(out));
process.exit(0);
