// Runs inside the output frame of the current cell (via google.colab.output.eval_js).
// Reports what that frame can do. Identifiers in the origin are masked.
(async () => {
  const out = {};
  const mask = (s) => String(s).replace(/[a-z0-9]{10,}/gi, '<id>');
  out.origin = mask(location.origin);
  out.secure = self.isSecureContext;
  out.crossOriginIsolated = self.crossOriginIsolated;
  out.userAgent = navigator.userAgent;
  out.devicePixelRatio = self.devicePixelRatio;
  out.fullscreenEnabled = document.fullscreenEnabled;
  out.api = {
    BroadcastChannel: typeof BroadcastChannel,
    CompressionStream: typeof CompressionStream,
    OffscreenCanvas: typeof OffscreenCanvas,
    SharedArrayBuffer: typeof SharedArrayBuffer,
    clipboard: typeof navigator.clipboard,
  };
  try {
    out.framesInParent = parent.frames.length;
  } catch (e) { out.framesInParent = String(e); }
  try {
    out.allowedFeatures = document.featurePolicy ? document.featurePolicy.allowedFeatures().sort() : null;
  } catch (e) { out.allowedFeatures = String(e); }
  try {
    const c = (self.google || {}).colab || {};
    out.colabApi = {
      kernel: Object.keys(c.kernel || {}).sort(),
      output: Object.keys(c.output || {}).sort(),
      comms: Object.keys((c.kernel || {}).comms || {}).sort(),
      accessAllowed: (c.kernel || {}).accessAllowed,
    };
  } catch (e) { out.colabApi = String(e); }

  out.webgpu = { present: 'gpu' in navigator };
  if (navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        out.webgpu.adapter = null;
      } else {
        const info = adapter.info || {};
        out.webgpu.adapter = {
          vendor: info.vendor, architecture: info.architecture,
          device: info.device, description: info.description,
        };
        out.webgpu.features = [...adapter.features].sort();
        const L = adapter.limits;
        out.webgpu.limits = {};
        for (const k of ['maxTextureDimension2D', 'maxBufferSize', 'maxStorageBufferBindingSize',
          'maxStorageBuffersPerShaderStage', 'maxStorageTexturesPerShaderStage',
          'maxComputeInvocationsPerWorkgroup', 'maxColorAttachments', 'maxBindGroups']) {
          out.webgpu.limits[k] = L[k];
        }
        out.webgpu.preferredCanvasFormat = navigator.gpu.getPreferredCanvasFormat();

        // A tiny compute job with readback, to prove the device really works in this frame.
        const t0 = performance.now();
        const device = await adapter.requestDevice();
        const module = device.createShaderModule({ code: `
          @group(0) @binding(0) var<storage, read_write> d: array<u32>;
          @compute @workgroup_size(4)
          fn main(@builtin(global_invocation_id) i: vec3<u32>) { d[i.x] = i.x * i.x + 1u; }` });
        const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
        const storage = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const readback = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const bindGroup = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [{ binding: 0, resource: { buffer: storage } }],
        });
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup); pass.dispatchWorkgroups(1); pass.end();
        encoder.copyBufferToBuffer(storage, 0, readback, 0, 16);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        out.webgpu.compute = Array.from(new Uint32Array(readback.getMappedRange()));
        out.webgpu.computeExpected = [1, 2, 5, 10];
        out.webgpu.deviceAndComputeMs = Math.round(performance.now() - t0);
        device.destroy();
      }
    } catch (e) { out.webgpu.error = String(e); }
  }
  return out;
})()
