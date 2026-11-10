// Open a public notebook in Colab WITHOUT signing in, and report what the page looks like
// from the outside: which frames exist, what attributes their <iframe> elements carry,
// which hosts were contacted, and how the notebook file itself was fetched.
//
// Usage: node anon_open.cjs <colab url> <screenshot.png> [seconds to wait]
const { chromium } = require('playwright');

const [url, shot, waitSec = '25'] = process.argv.slice(2);
const WEBGPU_FLAGS = ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=swiftshader',
  '--use-vulkan=swiftshader', '--ignore-gpu-blocklist', '--disable-vulkan-surface'];

(async () => {
  const browser = await chromium.launch({ args: WEBGPU_FLAGS });
  const context = await browser.newContext({
    viewport: { width: 1400, height: 1000 },
    userAgent: 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36',
    locale: 'en-US',
  });
  const page = await context.newPage();
  const hosts = new Map();
  const notebookFetches = [];
  const failures = [];
  page.on('request', (r) => {
    const u = new URL(r.url());
    hosts.set(u.host, (hosts.get(u.host) || 0) + 1);
    if (/ipynb|contents\/|raw\.githubusercontent|api\.github/.test(r.url())) notebookFetches.push(`${r.method()} ${r.url().slice(0, 220)}`);
  });
  page.on('requestfailed', (r) => failures.push(`${r.failure()?.errorText} ${r.url().slice(0, 140)}`));
  const consoleErrors = [];
  page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text().slice(0, 200)); });

  const resp = await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 90000 });
  await page.waitForTimeout(parseInt(waitSec, 10) * 1000);

  // <iframe> elements can sit inside shadow roots, so walk the composed tree.
  const iframes = await page.evaluate(() => {
    const out = [];
    const walk = (root) => {
      for (const el of root.querySelectorAll('*')) {
        if (el.tagName === 'IFRAME') {
          const r = el.getBoundingClientRect();
          out.push({ src: (el.getAttribute('src') || '').slice(0, 200), sandbox: el.getAttribute('sandbox'),
            allow: el.getAttribute('allow'), allowfullscreen: el.hasAttribute('allowfullscreen'),
            name: el.getAttribute('name'), cls: el.className, size: [Math.round(r.width), Math.round(r.height)] });
        }
        if (el.shadowRoot) walk(el.shadowRoot);
      }
    };
    walk(document);
    return out;
  });
  const frames = page.frames().map((f) => f.url().slice(0, 200));
  const text = await page.evaluate(() => document.body.innerText.slice(0, 600));
  await page.screenshot({ path: shot });
  console.log(JSON.stringify({
    status: resp && resp.status(), finalUrl: page.url().slice(0, 200), title: await page.title(),
    bodyText: text, frames, iframes, notebookFetches: notebookFetches.slice(0, 12),
    hosts: [...hosts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 40),
    failures: failures.slice(0, 15), consoleErrors: consoleErrors.slice(0, 10),
  }, null, 1));
  await browser.close();
})().catch((e) => { console.error(String(e).slice(0, 2000)); process.exit(1); });
