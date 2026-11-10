// Open Colab anonymously, but answer its request for the notebook file with a local
// .ipynb, so no notebook has to be published anywhere. Then report on every output frame.
//
// Usage: node anon_saved.cjs <local.ipynb> <screenshot.png> [emittedAtSecondsAgo]
const fs = require('fs');
const { chromium } = require('playwright');

const [nbPath, shot, ageSec = '3600'] = process.argv.slice(2);
const COLAB_URL = 'https://colab.research.google.com/github/googlecolab/colabtools/blob/main/notebooks/colab-github-demo.ipynb';
const WEBGPU_FLAGS = ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=swiftshader',
  '--use-vulkan=swiftshader', '--ignore-gpu-blocklist', '--disable-vulkan-surface'];
const mask = (s) => String(s).replace(/[a-z0-9]{10,}/gi, '<id>');

(async () => {
  // Pretend the outputs were produced a while ago.
  const emittedAt = Math.floor(Date.now() / 1000) - parseInt(ageSec, 10);
  const notebook = fs.readFileSync(nbPath, 'utf8').replaceAll('__EMITTED_AT__', String(emittedAt));

  const browser = await chromium.launch({ args: WEBGPU_FLAGS });
  const context = await browser.newContext({
    viewport: { width: 1400, height: 1000 }, deviceScaleFactor: 2, locale: 'en-US',
    userAgent: 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36',
  });
  const page = await context.newPage();
  let injected = 0;
  await page.route(/raw\.githubusercontent\.com\/.*\.ipynb/, async (route) => {
    injected++;
    await route.fulfill({ status: 200, contentType: 'text/plain; charset=utf-8',
      headers: { 'access-control-allow-origin': '*' }, body: notebook });
  });
  const consoleErrors = [];
  page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text().slice(0, 160)); });

  await page.goto(COLAB_URL, { waitUntil: 'domcontentloaded', timeout: 90000 });
  await page.waitForTimeout(12000);
  // Dismiss the banner if present, then scroll through the notebook so lazily created outputs render.
  for (let i = 0; i < 6; i++) { await page.mouse.wheel(0, 700); await page.waitForTimeout(1200); }
  await page.waitForTimeout(8000);

  // Attributes of every <iframe> element, as the parent page sees them.
  const iframeElements = await page.evaluate(() => {
    const out = [];
    const walk = (root) => {
      for (const el of root.querySelectorAll('*')) {
        if (el.tagName === 'IFRAME') {
          const r = el.getBoundingClientRect();
          out.push({ src: el.getAttribute('src') || '', sandbox: el.getAttribute('sandbox'), allow: el.getAttribute('allow'),
            size: [Math.round(r.width), Math.round(r.height)] });
        }
        if (el.shadowRoot) walk(el.shadowRoot);
      }
    };
    walk(document);
    return out;
  });

  // What each output frame itself reports.
  const frames = [];
  for (const f of page.frames()) {
    if (!/outputframe\.html/.test(f.url())) continue;
    let info;
    try {
      info = await f.evaluate(() => ({
        origin: location.origin,
        marker: self.gigiProbeMarker || null,
        jsOutputRan: !!self.gigiJsOutputRan,
        persist: document.getElementById('gigi-probe-persist') ? document.getElementById('gigi-probe-persist').textContent : null,
        persistFrames: self.gigiProbeFrames ? self.gigiProbeFrames() : null,
        frameA: document.getElementById('gigi-probe-a') ? document.getElementById('gigi-probe-a').textContent : null,
        bundleHtml: !!document.getElementById('gigi-bundle-html'),
        images: [...document.images].map((i) => [i.naturalWidth, i.naturalHeight]),
        jsNote: document.getElementById('gigi-js-output') ? document.getElementById('gigi-js-output').textContent : null,
        facts: self.gigiProbeResult || null,
        text: document.body ? document.body.innerText.slice(0, 120) : null,
      }));
    } catch (e) { info = { error: String(e).slice(0, 200) }; }
    const el = iframeElements.find((x) => x.src === f.url());
    frames.push({ url: f.url(), sandbox: el && el.sandbox, allow: el && el.allow, size: el && el.size, ...info });
  }

  await page.screenshot({ path: shot, fullPage: false });
  const origins = [...new Set(frames.map((f) => f.origin))];
  console.log(JSON.stringify({
    injectedRequests: injected, title: await page.title(),
    outputFrames: frames.length, distinctOrigins: origins.length, originsMasked: origins.map(mask),
    frames: frames.map((f) => ({ ...f, url: mask(f.url), origin: mask(f.origin) })),
    consoleErrors: consoleErrors.slice(0, 8),
  }, null, 1));
  await browser.close();
})().catch((e) => { console.error(String(e).slice(0, 2000)); process.exit(1); });
