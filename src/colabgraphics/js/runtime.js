// colabgraphics runtime: executes a compiled technique ("program") on a GPUDevice.
//
// The same file runs in a browser (the reader's GPU) and under Node with Dawn (headless,
// for read-back and tests). It knows nothing about notebooks or the DOM.
//
// A program is plain JSON produced by the Python compiler:
//   textures  [{ name, format, size:[w,h], data? }]      data = deflate-compressed pixels, base64
//   vars      { size, fields:[{ name, type, offset, value, ... }] }
//   passes    [{ kind:'compute'|'draw', name, code, bindings:[...], ... }]
//   view      name of the texture shown by default
// The WGSL in `code` is complete: the compiler has already prepended the declarations
// for every resource the author's code refers to.

export const FORMATS = {
  // format: [bytes per pixel, sample type when bound as a texture, array type for read-back]
  'r8unorm': [1, 'float', Uint8Array],
  'rg8unorm': [2, 'float', Uint8Array],
  'rgba8unorm': [4, 'float', Uint8Array],
  'rgba8unorm-srgb': [4, 'float', Uint8Array],
  'rgba8snorm': [4, 'float', Int8Array],
  'rgba16float': [8, 'float', Uint16Array],
  'r32float': [4, 'unfilterable-float', Float32Array],
  'rg32float': [8, 'unfilterable-float', Float32Array],
  'rgba32float': [16, 'unfilterable-float', Float32Array],
  'r32uint': [4, 'uint', Uint32Array],
  'rgba8uint': [4, 'uint', Uint8Array],
  'depth32float': [4, 'depth', Float32Array],
};

const SYS_SIZE = 32;  // struct Sys { time: f32, frame: u32, resolution: vec2f, mouse: vec4f }

async function inflate(base64) {
  const packed = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
  const stream = new Blob([packed]).stream().pipeThrough(new DecompressionStream('deflate'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

function writeField(view, field, value) {
  const put = { f32: 'setFloat32', i32: 'setInt32', u32: 'setUint32' }[field.scalar];
  const values = Array.isArray(value) ? value : [value];
  for (let i = 0; i < field.count; i++) {
    const v = values[i] ?? 0;
    view[put](field.offset + i * 4, field.scalar === 'f32' ? v : Math.trunc(Number(v)), true);
  }
}

export class Runtime {
  constructor(device, program) {
    this.device = device;
    this.program = program;
    this.textures = new Map();   // name -> { texture, desc }
    this.passes = [];
    this.frameIndex = 0;
    this.values = {};
    this.errors = [];            // validation messages, each tagged with the pass it came from
    this.blits = new Map();
  }

  async init() {
    const { device, program } = this;
    device.pushErrorScope('validation');

    // Which usages does each texture need? Derived from how the passes use it.
    const usage = new Map(program.textures.map((t) => [t.name,
      GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST]));
    for (const pass of program.passes) {
      for (const b of pass.bindings) if (b.kind === 'storage') usage.set(b.texture, usage.get(b.texture) | GPUTextureUsage.STORAGE_BINDING);
      for (const t of pass.targets || []) usage.set(t.texture, usage.get(t.texture) | GPUTextureUsage.RENDER_ATTACHMENT);
      if (pass.depth) usage.set(pass.depth.texture, usage.get(pass.depth.texture) | GPUTextureUsage.RENDER_ATTACHMENT);
    }
    for (const desc of program.textures) {
      const texture = device.createTexture({ label: desc.name, size: desc.size, format: desc.format, usage: usage.get(desc.name) });
      this.textures.set(desc.name, { texture, desc });
      if (desc.data) {
        const pixels = await inflate(desc.data);
        device.queue.writeTexture({ texture }, pixels, { bytesPerRow: desc.size[0] * FORMATS[desc.format][0], rowsPerImage: desc.size[1] }, desc.size);
      }
    }

    // One uniform buffer holds every variable of the technique.
    this.varsData = new ArrayBuffer(Math.max(program.vars.size, 16));
    this.varsBuffer = device.createBuffer({ label: 'vars', size: this.varsData.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    for (const field of program.vars.fields) this.setVar(field.name, field.value);

    this.samplers = {
      linear: device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' }),
      nearest: device.createSampler({ addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' }),
    };

    for (const pass of program.passes) this.passes.push(this._buildPass(pass));
    const error = await device.popErrorScope();
    if (error) this.errors.push({ pass: null, message: error.message });
    return this;
  }

  _buildPass(pass) {
    const { device } = this;
    const stage = pass.kind === 'compute' ? GPUShaderStage.COMPUTE : GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
    const sysBuffer = device.createBuffer({ label: pass.name + '.sys', size: SYS_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const layoutEntries = [];
    const groupEntries = [];
    for (const b of pass.bindings) {
      const entry = { binding: b.binding, visibility: stage };
      let resource;
      if (b.kind === 'texture') {
        const { texture, desc } = this.textures.get(b.texture);
        entry.texture = { sampleType: FORMATS[desc.format][1], viewDimension: '2d' };
        resource = texture.createView();
      } else if (b.kind === 'storage') {
        const { texture, desc } = this.textures.get(b.texture);
        entry.storageTexture = { access: 'write-only', format: desc.format, viewDimension: '2d' };
        resource = texture.createView();
      } else if (b.kind === 'sampler') {
        entry.sampler = { type: b.filter === 'linear' ? 'filtering' : 'non-filtering' };
        resource = this.samplers[b.filter];
      } else if (b.kind === 'vars') {
        entry.buffer = { type: 'uniform' };
        resource = { buffer: this.varsBuffer };
      } else if (b.kind === 'sys') {
        entry.buffer = { type: 'uniform' };
        resource = { buffer: sysBuffer };
      } else {
        throw new Error(`pass "${pass.name}": unknown binding kind "${b.kind}"`);
      }
      layoutEntries.push(entry);
      groupEntries.push({ binding: b.binding, resource });
    }
    const bindGroupLayout = device.createBindGroupLayout({ label: pass.name, entries: layoutEntries });
    const layout = device.createPipelineLayout({ label: pass.name, bindGroupLayouts: [bindGroupLayout] });
    const module = device.createShaderModule({ label: pass.name, code: pass.code });
    const built = { pass, sysBuffer, module, resolution: [1, 1] };

    if (pass.kind === 'compute') {
      built.pipeline = device.createComputePipeline({ label: pass.name, layout, compute: { module, entryPoint: pass.entry } });
      const size = this.textures.get(pass.dispatch.texture).desc.size;
      built.resolution = size;
      built.groups = pass.dispatch.workgroup.map((w, i) => Math.ceil((size[i] ?? 1) / w));
    } else {
      const targets = pass.targets.map((t) => ({ format: this.textures.get(t.texture).desc.format }));
      built.resolution = this.textures.get(pass.targets[0].texture).desc.size;
      const descriptor = {
        label: pass.name, layout,
        vertex: { module, entryPoint: pass.vertex },
        fragment: { module, entryPoint: pass.fragment, targets },
        primitive: { topology: pass.topology, cullMode: pass.cull },
      };
      if (pass.depth) descriptor.depthStencil = { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' };
      built.pipeline = device.createRenderPipeline(descriptor);
    }
    built.bindGroup = device.createBindGroup({ label: pass.name, layout: bindGroupLayout, entries: groupEntries });
    return built;
  }

  /** Per-module compile messages (errors and warnings), with line numbers in the complete WGSL. */
  async shaderMessages() {
    const out = [];
    for (const built of this.passes) {
      const info = await built.module.getCompilationInfo();
      for (const m of info.messages) out.push({ pass: built.pass.name, type: m.type, line: m.lineNum, column: m.linePos, message: m.message });
    }
    return out;
  }

  setVar(name, value) {
    const field = this.program.vars.fields.find((f) => f.name === name);
    if (!field) throw new Error(`no variable named "${name}"`);
    this.values[name] = value;
    writeField(new DataView(this.varsData), field, value);
    this.device.queue.writeBuffer(this.varsBuffer, 0, this.varsData);
  }

  /** Run every pass once. `input` carries time in seconds and the mouse state. */
  frame(input = {}) {
    const { device } = this;
    const mouse = input.mouse || [0, 0, 0, 0];
    for (const built of this.passes) {
      const sys = new DataView(new ArrayBuffer(SYS_SIZE));
      sys.setFloat32(0, input.time ?? 0, true);
      sys.setUint32(4, this.frameIndex, true);
      sys.setFloat32(8, built.resolution[0], true);
      sys.setFloat32(12, built.resolution[1], true);
      for (let i = 0; i < 4; i++) sys.setFloat32(16 + i * 4, mouse[i] ?? 0, true);
      device.queue.writeBuffer(built.sysBuffer, 0, sys.buffer);
    }
    const encoder = device.createCommandEncoder();
    for (const built of this.passes) {
      const { pass } = built;
      encoder.pushDebugGroup(pass.name);
      if (pass.kind === 'compute') {
        const p = encoder.beginComputePass({ label: pass.name });
        p.setPipeline(built.pipeline);
        p.setBindGroup(0, built.bindGroup);
        p.dispatchWorkgroups(...built.groups);
        p.end();
      } else {
        const descriptor = {
          label: pass.name,
          colorAttachments: pass.targets.map((t) => ({
            view: this.textures.get(t.texture).texture.createView(),
            loadOp: t.clear ? 'clear' : 'load', storeOp: 'store', clearValue: t.clear || undefined,
          })),
        };
        if (pass.depth) {
          descriptor.depthStencilAttachment = {
            view: this.textures.get(pass.depth.texture).texture.createView(),
            depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store',
          };
        }
        const p = encoder.beginRenderPass(descriptor);
        p.setPipeline(built.pipeline);
        p.setBindGroup(0, built.bindGroup);
        p.draw(pass.vertexCount, pass.instanceCount);
        p.end();
      }
      encoder.popDebugGroup();
    }
    device.queue.submit([encoder.finish()]);
    this.frameIndex++;
  }

  /** Copy a texture back to the CPU. Returns tightly packed rows. */
  async readTexture(name) {
    const { device } = this;
    const { texture, desc } = this.textures.get(name);
    const [width, height] = desc.size;
    const [bytesPerPixel, , ArrayType] = FORMATS[desc.format];
    const stride = Math.ceil(width * bytesPerPixel / 256) * 256;
    const buffer = device.createBuffer({ size: stride * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow: stride, rowsPerImage: height }, [width, height]);
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const padded = new Uint8Array(buffer.getMappedRange());
    const bytes = new Uint8Array(width * height * bytesPerPixel);
    for (let y = 0; y < height; y++) bytes.set(padded.subarray(y * stride, y * stride + width * bytesPerPixel), y * width * bytesPerPixel);
    buffer.unmap();
    buffer.destroy();
    return { name, width, height, format: desc.format, bytesPerPixel, bytes, data: new ArrayType(bytes.buffer) };
  }

  /** Draw one of the technique's textures into a render target (the canvas, in a browser). */
  present(target, name) {
    const { device } = this;
    const { texture, desc } = this.textures.get(name);
    const sampleType = FORMATS[desc.format][1];
    const key = sampleType + '>' + target.format;
    if (!this.blits.has(key)) {
      const type = { uint: 'texture_2d<u32>', depth: 'texture_depth_2d' }[sampleType] || 'texture_2d<f32>';
      const load = {
        uint: 'vec4f(textureLoad(src, p, 0)) / 255.0',
        depth: 'vec4f(vec3f(textureLoad(src, p, 0)), 1.0)',
      }[sampleType] || 'textureLoad(src, p, 0)';
      const module = device.createShaderModule({ label: 'present', code: `
        @group(0) @binding(0) var src: ${type};
        @group(0) @binding(1) var<uniform> target_size: vec2u;
        @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
          let p = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
          return vec4f(p[i], 0, 1);
        }
        @fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
          // The target may be a different size from the texture: nearest-neighbour scale.
          let p = vec2i(pos.xy * vec2f(textureDimensions(src)) / vec2f(target_size));
          let c = ${load};
          return vec4f(c.rgb, 1.0);
        }` });
      const pipeline = device.createRenderPipeline({
        label: 'present', layout: 'auto',
        vertex: { module, entryPoint: 'vs' },
        fragment: { module, entryPoint: 'fs', targets: [{ format: target.format }] },
      });
      const sizeBuffer = device.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.blits.set(key, { pipeline, sizeBuffer });
    }
    const { pipeline, sizeBuffer } = this.blits.get(key);
    device.queue.writeBuffer(sizeBuffer, 0, new Uint32Array([target.width, target.height]));
    const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: texture.createView() },
      { binding: 1, resource: { buffer: sizeBuffer } },
    ] });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({ colorAttachments: [{ view: target.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }] });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  destroy() {
    for (const { texture } of this.textures.values()) texture.destroy();
    this.varsBuffer.destroy();
    for (const built of this.passes) built.sysBuffer.destroy();
  }
}
