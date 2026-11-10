// Show a Gigi snapshot fragment as a SAVED cell output in real Colab, anonymously and
// with no runtime: wrap the fragment in a one-cell notebook, hand that notebook to Colab
// in place of the GitHub file it asks for, and watch the output frame.
//
// Usage: node anon_gigi.cjs <snapshot.html> <screenshot.png> [title]
const fs = require('fs');
const { chromium } = require('playwright');

const [snapPath, shot, title = 'Gigi boxblur'] = process.argv.slice(2);
const COLAB_URL = 'https://colab.research.google.com/github/googlecolab/colabtools/blob/main/notebooks/colab-github-demo.ipynb';

const fragment = fs.readFileSync(snapPath, 'utf8');
const notebook = JSON.stringify({
  nbformat: 4, nbformat_minor: 0,
  metadata: { colab: { provenance: [] }, kernelspec: { name: 'python3', display_name: 'Python 3' }, language_info: { name: 'python' } },
  cells: [
    { cell_type: 'markdown', metadata: {}, source: [`# ${title}\n`, '\n',
      'The output below is stored in the notebook file. No runtime is connected and nothing was executed in this session.'] },
    { cell_type: 'code', metadata: {}, execution_count: 1, source: ['technique.show()'],
      outputs: [{ output_type: 'display_data', metadata: {}, data: { 'text/html': [fragment], 'text/plain': ['<Gigi technique snapshot>'] } }] },
  ],
});

(async () => {
  const browser = await chromium.launch({ args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan',
    '--use-angle=swiftshader', '--use-vulkan=swiftshader', '--ignore-gpu-blocklist', '--disable-vulkan-surface'] });
  const context = await browser.newContext({
    viewport: { width: 1400, height: 1150 }, deviceScaleFactor: 2, locale: 'en-US',
    userAgent: 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36',
  });
  const page = await context.newPage();
  await page.route(/raw\.githubusercontent\.com\/.*\.ipynb/, (route) => route.fulfill({
    status: 200, contentType: 'text/plain; charset=utf-8', headers: { 'access-control-allow-origin': '*' }, body: notebook }));
  const problems = [];
  page.on('console', (m) => { if (m.type() === 'error' && !/401|requestStorageAccess/.test(m.text())) problems.push(m.text().slice(0, 240)); });

  const t0 = Date.now();
  await page.goto(COLAB_URL, { waitUntil: 'domcontentloaded', timeout: 90000 });
  // Close the promo banner so more of the notebook is visible.
  await page.getByRole('button', { name: /close/i }).first().click({ timeout: 8000 }).catch(() => {});

  // Find the output frame that holds the snapshot.
  let frame = null;
  for (let i = 0; i < 60 && !frame; i++) {
    await page.waitForTimeout(1000);
    for (const f of page.frames()) {
      if (!/outputframe\.html/.test(f.url())) continue;
      if (await f.evaluate(() => !!self.gigiSnapshot).catch(() => false)) { frame = f; break; }
    }
  }
  if (!frame) {
    // Diagnose: what does the page show, and what is in each output frame?
    await page.screenshot({ path: shot });
    const diag = [];
    for (const f of page.frames()) {
      if (!/outputframe\.html/.test(f.url())) continue;
      diag.push(await f.evaluate(() => ({ html: document.body ? document.body.innerHTML.length : -1,
        text: document.body ? document.body.innerText.slice(0, 200) : null,
        scripts: document.scripts.length, snapshotData: !!document.getElementById('gigi-snapshot-data') })).catch((e) => ({ error: String(e).slice(0, 120) })));
    }
    console.log(JSON.stringify({ rendered: false, reason: 'snapshot output frame never appeared', outputFrames: diag,
      pageText: (await page.evaluate(() => document.body.innerText)).slice(0, 700), problems }, null, 1));
    await browser.close();
    process.exit(1);
  }
  const foundAfterMs = Date.now() - t0;
  let rendered = true;
  await frame.waitForFunction(() => /FPS: \d/.test((document.getElementById('FPS') || {}).textContent || ''), null, { timeout: 60000 })
    .catch(() => { rendered = false; });
  const firstFrameAfterMs = Date.now() - t0;
  await frame.locator('#mainCanvas').scrollIntoViewIfNeeded().catch(() => {});
  await page.waitForTimeout(4000);

  const el = await frame.frameElement();
  const elementInfo = await el.evaluate((e) => ({ sandbox: e.getAttribute('sandbox'), allow: e.getAttribute('allow'),
    width: Math.round(e.getBoundingClientRect().width), height: Math.round(e.getBoundingClientRect().height) }));
  const state = await frame.evaluate(() => ({
    snapshot: self.gigiSnapshot,
    kernelAccess: google.colab.kernel.accessAllowed,
    fps: document.getElementById('FPS').textContent.replace(/\s+/g, ' ').trim(),
    canvas: [document.getElementById('mainCanvas').width, document.getElementById('mainCanvas').height],
    radius: document.getElementById('inputBox_radius').value,
    viewing: [...document.getElementById("Viewing").options].map((o) => o.value).filter(Boolean),
    scripts: [...document.scripts].map((s) => ({ type: s.type, id: s.id, chars: s.textContent.length, src: (s.src || "").slice(0, 60) })),
  }));
  await page.screenshot({ path: shot });
  console.log(JSON.stringify({ rendered, foundAfterMs, firstFrameAfterMs, notebookBytes: notebook.length, elementInfo, state, problems }, null, 1));
  await browser.close();
  process.exit(rendered ? 0 : 1);
})().catch((e) => { console.error(String(e).slice(0, 2000)); process.exit(1); });
