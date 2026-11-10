// Probe: can the Dawn Node binding find any WebGPU adapter on this GPU-less Linux machine?
import { create, globals } from 'webgpu';
Object.assign(globalThis, globals);

const gpu = create([]);
const adapter = await gpu.requestAdapter();
if (!adapter) {
  console.log('adapter: none');
} else {
  const info = adapter.info ?? {};
  console.log('adapter:', JSON.stringify({
    vendor: info.vendor, architecture: info.architecture,
    device: info.device, description: info.description,
  }));
  console.log('features:', [...adapter.features].sort().join(', '));
}
process.exit(0);
