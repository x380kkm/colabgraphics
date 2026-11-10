// Two questions about saved outputs in real Colab (anonymous, no runtime):
//   1. many WebGPU outputs in one notebook: are their frames created lazily, do all of
//      them get a working device, does the output-frame origin change between page loads?
//   2. how large can one stored HTML output be and still render?
// Usage: node anon_many.cjs <canvases> <padding MB, comma separated>
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

const [nCanvases = '24', paddings = '8,24'] = process.argv.slice(2);
const COLAB_URL = 'https://colab.research.google.com/github/googlecolab/colabtools/blob/main/notebooks/colab-github-demo.ipynb';
const persist = fs.readFileSync(path.join(__dirname, '..', 'colab-probe', 'js', 'persist.html'), 'utf8')
  .replaceAll('__EMITTED_AT__', String(Math.floor(Date.now() / 1000) - 60));

const cell = (n, html) => ({ cell_type: 'code', metadata: {}, execution_count: n, source: [`show(${n})`],
  outputs: [{ output_type: 'display_data', metadata: {}, data: { 'text/html': [html] } }] });
const cells = [{ cell_type: 'markdown', metadata: {}, source: ['# Many saved WebGPU outputs'] }];
for (let i = 0; i < parseInt(nCanvases, 10); i++) {
  cells.push(cell(i + 1, `<script>self.gigiCell = ${i};</script>` + persist));
}
for (const mb of paddings.split(',').filter(Boolean).map(Number)) {
  const pad = '<!--' + 'x'.repeat(mb * 1024 * 1024) + '-->';
  cells.push(cell(100 + mb, `<div id="gigi-big">big output: ${mb} MB of padding arrived</div><script>self.gigiBig = ${mb};</script>` + pad));
}
const notebook = JSON.stringify({ nbformat: 4, nbformat_minor: 0,
  metadata: { colab: { provenance: [] }, kernelspec: { name: 'python3', display_name: 'Python 3' } }, cells });

async function survey(page) {
  const rows = [];
  for (const f of page.frames()) {
    if (!/outputframe\.html/.test(f.url())) continue;
    const r = await f.evaluate(() => ({ origin: location.origin, cell: self.gigiCell, big: self.gigiBig,
      frames: self.gigiProbeFrames ? self.gigiProbeFrames() : null,
      status: (document.getElementById('gigi-probe-persist') || {}).textContent || null })).catch(() => null);
    if (r) rows.push(r);
  }
  return rows;
}
const summarize = (rows) => ({
  outputFrames: rows.length,
  canvasFrames: rows.filter((r) => r.cell !== undefined && r.cell !== null).length,
  animating: rows.filter((r) => r.frames > 0).length,
  failed: rows.filter((r) => r.status && /FAILED/.test(r.status)).map((r) => r.status).slice(0, 3),
  bigRendered: rows.filter((r) => r.big).map((r) => r.big).sort((a, b) => a - b),
});

(async () => {
  const browser = await chromium.launch({ args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan',
    '--use-angle=swiftshader', '--use-vulkan=swiftshader', '--ignore-gpu-blocklist', '--disable-vulkan-surface'] });
  const out = { notebookMB: +(notebook.length / 1e6).toFixed(1), loads: [] };
  for (let load = 0; load < 2; load++) {
    const context = await browser.newContext({ viewport: { width: 1400, height: 1000 }, locale: 'en-US',
      userAgent: 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36' });
    const page = await context.newPage();
    await page.route(/raw\.githubusercontent\.com\/.*\.ipynb/, (route) => route.fulfill({
      status: 200, contentType: 'text/plain; charset=utf-8', headers: { 'access-control-allow-origin': '*' }, body: notebook }));
    const t0 = Date.now();
    await page.goto(COLAB_URL, { waitUntil: 'domcontentloaded', timeout: 120000 });
    await page.waitForTimeout(15000);
    const initial = await survey(page);
    // Scroll the notebook to the bottom in steps.
    await page.mouse.move(700, 600);
    for (let i = 0; i < 40; i++) { await page.mouse.wheel(0, 900); await page.waitForTimeout(350); }
    await page.waitForTimeout(6000);
    const scrolled = await survey(page);
    const origins = [...new Set(scrolled.map((r) => r.origin))];
    out.loads.push({ loadMs: Date.now() - t0, beforeScroll: summarize(initial), afterScroll: summarize(scrolled),
      // Host labels only: enough to see what changes between loads.
      originLabels: origins.map((o) => new URL(o).hostname.split('.')[0]) });
    await context.close();
    if (load === 0 && process.argv[4] === 'once') break;
  }
  console.log(JSON.stringify(out, null, 1));
  await browser.close();
})().catch((e) => { console.error(String(e).slice(0, 2000)); process.exit(1); });
