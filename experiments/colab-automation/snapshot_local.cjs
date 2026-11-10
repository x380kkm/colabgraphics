// Check a self-contained snapshot fragment locally before sending it to Colab:
// serve it as the only content of a sandboxed cross-origin iframe (the same sandbox
// tokens Colab's output frames were measured to carry) and wait for the first frames.
// Usage: node snapshot_local.cjs <snapshot.html> <screenshot.png>
const http = require('http');
const fs = require('fs');
const { chromium } = require('playwright');

const [snapPath, shot] = process.argv.slice(2);
const SANDBOX = 'allow-downloads allow-forms allow-pointer-lock allow-popups allow-popups-to-escape-sandbox allow-same-origin allow-scripts allow-storage-access-by-user-activation';
const listen = (h, host) => new Promise((r) => { const s = http.createServer(h).listen(0, host, () => r(s)); });

(async () => {
  const fragment = fs.readFileSync(snapPath, 'utf8');
  const frameServer = await listen((req, res) => {
    res.writeHead(200, { 'content-type': 'text/html' });
    res.end('<!doctype html><meta charset="utf-8"><body><div id="output-area">' + fragment + '</div>');
  }, '127.0.0.1');
  const topServer = await listen((req, res) => {
    res.writeHead(200, { 'content-type': 'text/html' });
    res.end(`<!doctype html><body style="margin:0"><iframe sandbox="${SANDBOX}" allow="" src="http://127.0.0.1:${frameServer.address().port}/" style="border:0;width:100%;height:1000px"></iframe>`);
  }, 'localhost');
  const browser = await chromium.launch({ args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan',
    '--use-angle=swiftshader', '--use-vulkan=swiftshader', '--ignore-gpu-blocklist', '--disable-vulkan-surface'] });
  const page = await browser.newPage({ viewport: { width: 1300, height: 1000 } });
  const problems = [];
  page.on('console', (m) => { if (m.type() === 'error') problems.push(m.text().slice(0, 300)); });
  page.on('pageerror', (e) => problems.push('pageerror: ' + e.message.slice(0, 300)));
  await page.goto(`http://localhost:${topServer.address().port}/`);
  const frame = page.frames().find((f) => f.url().startsWith('http://127.0.0.1'));
  let ok = true;
  await frame.waitForFunction(() => /FPS: \d/.test((document.getElementById('FPS') || {}).textContent || ''), null, { timeout: 45000 })
    .catch(() => { ok = false; });
  await page.waitForTimeout(1000);
  const state = await frame.evaluate(() => ({ snapshot: self.gigiSnapshot, fps: (document.getElementById('FPS') || {}).textContent,
    canvas: [document.getElementById('mainCanvas').width, document.getElementById('mainCanvas').height] }));
  await page.screenshot({ path: shot });
  console.log(JSON.stringify({ rendered: ok, state, problems }, null, 1));
  await browser.close(); frameServer.close(); topServer.close();
  process.exit(ok ? 0 : 1);
})().catch((e) => { console.error(String(e).slice(0, 1500)); process.exit(1); });
