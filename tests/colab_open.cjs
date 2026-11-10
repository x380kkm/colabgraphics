// Open a notebook from this repository in the real Colab front end, the way a reader
// would (public GitHub link, not signed in, no runtime), and check that every stored
// colabgraphics output is running.
//
// Usage: node colab_open.cjs <colab url> <screenshot prefix>
const { chromium } = require('playwright');

const [url, shotPrefix] = process.argv.slice(2);
const FLAGS = ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=swiftshader',
  '--use-vulkan=swiftshader', '--ignore-gpu-blocklist', '--disable-vulkan-surface'];

(async () => {
  const browser = await chromium.launch({ args: FLAGS });
  const context = await browser.newContext({ viewport: { width: 1400, height: 1100 }, deviceScaleFactor: 2, locale: 'zh-CN',
    userAgent: 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36' });
  const page = await context.newPage();
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 90000 });
  await page.waitForTimeout(12000);
  await page.getByRole('button', { name: /close|关闭/i }).first().click({ timeout: 3000 }).catch(() => {});

  // Each stored output lives in its own frame. Bring each into view in turn so it starts.
  const viewers = [];
  for (const f of page.frames()) {
    if (!/outputframe\.html/.test(f.url())) continue;
    if (!(await f.evaluate(() => !!(self.colabgraphics && self.colabgraphics.length)).catch(() => false))) continue;
    viewers.push(f);
  }
  const report = [];
  for (const [i, f] of viewers.entries()) {
    // Scroll the output's frame element into view in the notebook page itself, then give
    // the technique time to start: on a software GPU, building pipelines takes a moment.
    const element = await f.frameElement();
    await element.scrollIntoViewIfNeeded().catch(() => {});
    await f.waitForFunction(() => self.colabgraphics[0].frames > 20 || self.colabgraphics[0].failed, null, { timeout: 45000 }).catch(() => {});
    const state = await f.evaluate(() => {
      const h = self.colabgraphics[0];
      return { name: h.program.name, failed: h.failed, frames: h.frames, status: document.querySelector('.cg-status').textContent,
        controls: [...document.querySelectorAll('.cg-name')].map((e) => e.textContent) };
    });
    report.push(state);
    await page.screenshot({ path: `${shotPrefix}_${i + 1}_${state.name}.png` });
  }
  const ok = report.length > 0 && report.every((r) => !r.failed && r.frames > 0);
  console.log(JSON.stringify({ ok, title: await page.title(), viewers: report }, null, 1));
  await browser.close();
  process.exit(ok ? 0 : 1);
})().catch((e) => { console.log(JSON.stringify({ ok: false, reason: String(e).slice(0, 600) })); process.exit(1); });
