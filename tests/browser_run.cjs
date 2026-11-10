// Open a colabgraphics page in headless Chromium (WebGPU on SwiftShader), let the
// technique run, and read one of its textures back.
//
// Usage: node browser_run.cjs <page.html> <texture> <out.bin> [screenshot.png] [var=value ...]
// Prints one line of JSON describing the run; the texture bytes go to <out.bin>.
const fs = require('fs');
const http = require('http');
const { chromium } = require('playwright');

const [page_, texture, outBin, shot, ...sets] = process.argv.slice(2);
const FLAGS = ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=swiftshader',
  '--use-vulkan=swiftshader', '--ignore-gpu-blocklist', '--disable-vulkan-surface'];

(async () => {
  const html = fs.readFileSync(page_);
  const server = await new Promise((r) => { const s = http.createServer((q, res) => {
    res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' }); res.end(html); }).listen(0, '127.0.0.1', () => r(s)); });
  const browser = await chromium.launch({ args: FLAGS });
  const page = await browser.newPage({ viewport: { width: 1100, height: 800 } });
  const problems = [];
  page.on('console', (m) => { if (m.type() === 'error') problems.push(m.text().slice(0, 300)); });
  page.on('pageerror', (e) => problems.push('pageerror: ' + e.message.slice(0, 300)));
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  await page.waitForFunction(() => self.colabgraphics && self.colabgraphics.length > 0, null, { timeout: 30000 });
  const result = await page.evaluate(async ({ texture, sets }) => {
    const handle = await self.colabgraphics[0].ready;
    if (handle.failed) return { failed: handle.failed };
    for (const s of sets) {
      const [name, value] = s.split('=');
      handle.runtime.setVar(name, JSON.parse(value));
    }
    const before = handle.frames;
    await new Promise((resolve) => { const wait = () => (handle.frames >= before + 3 ? resolve() : requestAnimationFrame(wait)); wait(); });
    const t = await handle.runtime.readTexture(texture);
    const pixel = await handle.runtime.readPixel(texture, 10, 7);   // the viewer's pixel readout uses this
    let binary = '';
    for (let i = 0; i < t.bytes.length; i += 0x8000) binary += String.fromCharCode.apply(null, t.bytes.subarray(i, i + 0x8000));
    return { width: t.width, height: t.height, format: t.format, bytesPerPixel: t.bytesPerPixel, frames: handle.frames, pixel,
      status: document.querySelector('.cg-status').textContent, base64: btoa(binary) };
  }, { texture, sets });
  if (shot) await page.screenshot({ path: shot });
  if (result.base64) { fs.writeFileSync(outBin, Buffer.from(result.base64, 'base64')); delete result.base64; }
  console.log(JSON.stringify({ ...result, problems }));
  await browser.close();
  server.close();
  process.exit(result.failed ? 1 : 0);
})().catch((e) => { console.log(JSON.stringify({ failed: String(e).slice(0, 800) })); process.exit(1); });
