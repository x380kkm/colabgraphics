// Put a colabgraphics snapshot into the real Colab front end as a SAVED cell output,
// without signing in and without a runtime, and report whether it runs there.
//
// Colab is opened on a public GitHub notebook address; its request for the .ipynb file is
// answered locally with a one-cell notebook whose output is the snapshot. Nothing is
// published anywhere.
//
// Usage: node colab_anon.cjs <cell source file> <snapshot fragment.html> <screenshot.png>
const fs = require('fs');
const { chromium } = require('playwright');

const [sourcePath, fragmentPath, shot] = process.argv.slice(2);
const COLAB_URL = 'https://colab.research.google.com/github/googlecolab/colabtools/blob/main/notebooks/colab-github-demo.ipynb';
const FLAGS = ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=swiftshader',
  '--use-vulkan=swiftshader', '--ignore-gpu-blocklist', '--disable-vulkan-surface'];

const notebook = JSON.stringify({
  nbformat: 4, nbformat_minor: 0,
  metadata: { colab: { provenance: [] }, kernelspec: { name: 'python3', display_name: 'Python 3' }, language_info: { name: 'python' } },
  cells: [{ cell_type: 'code', metadata: {}, execution_count: 1,
    source: fs.readFileSync(sourcePath, 'utf8').split(/(?<=\n)/),
    outputs: [{ output_type: 'display_data', metadata: {},
      data: { 'text/html': [fs.readFileSync(fragmentPath, 'utf8')], 'text/plain': ['<colabgraphics technique>'] } }] }],
});

(async () => {
  const browser = await chromium.launch({ args: FLAGS });
  const context = await browser.newContext({ viewport: { width: 1400, height: 1500 }, deviceScaleFactor: 2, locale: 'en-US',
    userAgent: 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36' });
  const page = await context.newPage();
  await page.route(/raw\.githubusercontent\.com\/.*\.ipynb/, (route) => route.fulfill({
    status: 200, contentType: 'text/plain; charset=utf-8', headers: { 'access-control-allow-origin': '*' }, body: notebook }));
  await page.goto(COLAB_URL, { waitUntil: 'domcontentloaded', timeout: 90000 });
  await page.getByRole('button', { name: /close/i }).first().click({ timeout: 8000 }).catch(() => {});

  let frame = null;
  for (let i = 0; i < 60 && !frame; i++) {
    await page.waitForTimeout(1000);
    for (const f of page.frames()) {
      if (/outputframe\.html/.test(f.url()) && await f.evaluate(() => !!(self.colabgraphics && self.colabgraphics.length)).catch(() => false)) { frame = f; break; }
    }
  }
  if (!frame) { console.log(JSON.stringify({ ok: false, reason: 'no output frame with a technique appeared' })); await browser.close(); process.exit(1); }
  await frame.locator('canvas').scrollIntoViewIfNeeded().catch(() => {});
  await page.waitForTimeout(5000);
  const state = await frame.evaluate(() => {
    const h = self.colabgraphics[0];
    return { failed: h.failed, frames: h.frames, status: document.querySelector('.cg-status').textContent,
      kernelAccess: google.colab.kernel.accessAllowed, canvas: [document.querySelector('canvas').width, document.querySelector('canvas').height] };
  });
  await page.screenshot({ path: shot });
  console.log(JSON.stringify({ ok: !state.failed && state.frames > 0, notebookKB: Math.round(notebook.length / 1024), ...state }));
  await browser.close();
  process.exit(!state.failed && state.frames > 0 ? 0 : 1);
})().catch((e) => { console.log(JSON.stringify({ ok: false, reason: String(e).slice(0, 600) })); process.exit(1); });
