// The "Gradio path", rehearsed locally.
//
// In Colab, Gradio starts an HTTP server inside the kernel process and shows it in the
// cell output through an iframe whose URL comes from google.colab.kernel.proxyPort.
// Here: a static HTTP server stands in for the kernel-side server (serving a folder of
// Gigi-generated WebGPU apps, unmodified), and a page on another origin embeds it in an
// iframe the way the cell output would.
//
// Usage: node demo.cjs <generated WebGPU root> <relative path of app index.html> <screenshot.png>
const http = require('http');
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

const [root, appPath, shot] = process.argv.slice(2);
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.png': 'image/png',
  '.json': 'application/json', '.csv': 'text/csv', '.ico': 'image/x-icon' };

function listen(handler, host) {
  return new Promise((resolve) => { const s = http.createServer(handler).listen(0, host, () => resolve(s)); });
}

(async () => {
  const requests = [];
  const kernel = await listen((req, res) => {
    const rel = decodeURIComponent(new URL(req.url, 'http://x').pathname);
    const file = path.join(root, path.normalize(rel));
    if (!file.startsWith(path.resolve(root)) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      requests.push(`404 ${rel}`);
      res.writeHead(404); res.end('not found'); return;
    }
    requests.push(`200 ${rel}`);
    res.writeHead(200, { 'content-type': MIME[path.extname(file)] || 'application/octet-stream' });
    fs.createReadStream(file).pipe(res);
  }, '127.0.0.1');
  const appUrl = `http://127.0.0.1:${kernel.address().port}/${appPath}`;

  const cell = await listen((req, res) => {
    res.writeHead(200, { 'content-type': 'text/html' });
    res.end(`<!doctype html><meta charset="utf-8">
      <body style="margin:0;font:14px system-ui;background:#f6f6f6">
      <div style="padding:8px 12px;color:#444">cell output (origin: localhost) &rarr; iframe &rarr; kernel-side HTTP server (origin: 127.0.0.1)</div>
      <iframe src="${appUrl}" style="border:0;width:100%;height:900px;background:white"></iframe>`);
  }, 'localhost');

  const browser = await chromium.launch({ args: [
    '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=swiftshader',
    '--use-vulkan=swiftshader', '--ignore-gpu-blocklist', '--disable-vulkan-surface',
  ] });
  const page = await browser.newPage({ viewport: { width: 1280, height: 960 }, deviceScaleFactor: 2 });
  const problems = [];
  page.on('console', (m) => { if (m.type() === 'error') problems.push('console: ' + m.text()); });
  page.on('pageerror', (e) => problems.push('pageerror: ' + e.message));
  page.on('dialog', async (d) => { problems.push('dialog: ' + d.message()); await d.dismiss(); });
  await page.goto(`http://localhost:${cell.address().port}/`);
  const frame = page.frames().find((f) => f.url().startsWith('http://127.0.0.1'));
  await frame.waitForFunction(() => /FPS: \d/.test(document.getElementById('FPS').textContent), null, { timeout: 60000 });
  await page.waitForTimeout(1500);

  const info = await frame.evaluate(() => ({
    fps: document.getElementById('FPS').textContent.replace(/\s+/g, ' ').trim(),
    canvas: [document.getElementById('mainCanvas').width, document.getElementById('mainCanvas').height],
    viewing: [...document.getElementById('Viewing').options].map((o) => o.value).filter(Boolean),
    radius: document.getElementById('inputBox_radius') ? document.getElementById('inputBox_radius').value : null,
    loading: document.getElementById('LoadingText').textContent,
  }));
  await page.screenshot({ path: shot });
  console.log(JSON.stringify({ appUrl, info, problems, servedOk: requests.filter((r) => r.startsWith('200')).length,
    notFound: requests.filter((r) => r.startsWith('404')) }, null, 1));
  await browser.close(); kernel.close(); cell.close();
})().catch((e) => { console.error(e); process.exit(1); });
