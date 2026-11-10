// Local rehearsal of the browser-side probe snippets.
//
// Colab itself is not reachable from the development container, so this builds the
// same shape locally: a top page on one origin that embeds "cell output" iframes
// from another origin, each with a mock of the google.colab API. The snippets in
// js/ are the exact text the notebook ships. page/frame.evaluate(string) stands in
// for google.colab.output.eval_js.
//
// Usage: node test_local.cjs
const http = require('http');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const { chromium } = require('playwright');

const HERE = __dirname;
const read = (name) => fs.readFileSync(path.join(HERE, name), 'utf8');

const MOCK_COLAB = `<script>
window.google = { colab: {
  kernel: {
    accessAllowed: true,
    proxyPort: async (port) => 'http://127.0.0.1:' + port + '/',
    invokeFunction: async (name, args, kwargs) => ({ mock: true, name, args, kwargs }),
    comms: {
      registerTarget() {},
      open: async (target, data, buffers) => {
        const queue = []; let wake = null;
        window.mockCommSent = [];
        return {
          close() {},
          send: async (d, opts) => {
            window.mockCommSent.push(d);
            if (d && d.echo_bytes !== undefined) {
              queue.push({ data: { echo: d.echo_bytes }, buffers: [new ArrayBuffer(d.echo_bytes)] });
              if (wake) { wake(); wake = null; }
            }
          },
          messages: { [Symbol.asyncIterator]() { return { next: async () => {
            while (!queue.length) await new Promise((r) => { wake = r; });
            return { value: queue.shift(), done: false };
          } }; } },
        };
      },
    },
  },
  output: { setIframeHeight() {}, getActiveOutputArea() { return document.body; } },
} };
</script>`;

const page = (body) => `<!doctype html><meta charset="utf-8"><body>${MOCK_COLAB}${body}`;

function serve(handler, host) {
  return new Promise((resolve) => {
    const s = http.createServer(handler).listen(0, host, () => resolve(s));
  });
}

function startKernelServer() {
  return new Promise((resolve, reject) => {
    const code = 'import kernel_server, time\n' +
      's, p = kernel_server.start_probe_server()\nprint(p, flush=True)\ntime.sleep(600)\n';
    const proc = spawn('python3', ['-c', code], { cwd: HERE });
    proc.stdout.once('data', (d) => resolve({ proc, port: parseInt(String(d), 10) }));
    proc.stderr.on('data', (d) => process.stderr.write('[kernel_server] ' + d));
    proc.once('exit', (c) => reject(new Error('kernel server exited ' + c)));
  });
}

const checks = [];
const check = (name, ok, detail) => { checks.push({ name, ok }); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`); };

async function main() {
  const kernel = await startKernelServer();
  const pages = {
    '/cellA': page(read('js/frame_a.html')),
    '/cellB': page('<div>cell B</div>'),
    '/cellC': page(read('js/comm.html')),
    '/cellD': page(read('js/persist.html').replace('__EMITTED_AT__', String(Math.floor(Date.now() / 1000) - 42))),
  };
  const frameServer = await serve((req, res) => {
    const body = pages[req.url.split('?')[0]];
    res.writeHead(body ? 200 : 404, { 'content-type': 'text/html' });
    res.end(body || 'not found');
  }, '127.0.0.1');
  const frameOrigin = `http://127.0.0.1:${frameServer.address().port}`;
  const topServer = await serve((req, res) => {
    res.writeHead(200, { 'content-type': 'text/html' });
    res.end('<!doctype html><meta charset="utf-8"><body>' + Object.keys(pages).map((p) =>
      `<iframe src="${frameOrigin}${p}" width="420" height="330" sandbox="allow-scripts allow-same-origin"></iframe>`).join('\n'));
  }, 'localhost');

  const browser = await chromium.launch({ args: [
    '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=swiftshader',
    '--use-vulkan=swiftshader', '--ignore-gpu-blocklist', '--disable-vulkan-surface',
  ] });
  const top = await browser.newPage({ viewport: { width: 900, height: 720 } });
  top.on('console', (m) => { if (m.type() === 'error') console.log('[browser error]', m.text()); });
  await top.goto(`http://localhost:${topServer.address().port}/`);
  await top.waitForLoadState('networkidle');
  const frameOf = (p) => top.frames().find((f) => f.url().endsWith(p));

  // frame.js
  const frame = await frameOf('/cellB').evaluate(read('js/frame.js'));
  console.log('\nframe.js →', JSON.stringify(frame, null, 1));
  check('frame.js: secure context', frame.secure === true);
  check('frame.js: WebGPU compute readback', JSON.stringify(frame.webgpu.compute) === '[1,2,5,10]', JSON.stringify(frame.webgpu.adapter));
  check('frame.js: colab api listed', Array.isArray(frame.colabApi.kernel) && frame.colabApi.kernel.includes('proxyPort'));
  check('frame.js: result is JSON-serializable', !!JSON.stringify(frame));

  // sibling.js
  const sibling = await frameOf('/cellB').evaluate(read('js/sibling.js'));
  console.log('\nsibling.js →', JSON.stringify(sibling));
  check('sibling.js: direct call into another cell frame', sibling.direct === 'pong:direct');
  check('sibling.js: BroadcastChannel between cell frames', sibling.broadcast === 'pong:broadcast');

  // http.js against the real kernel_server.py
  const httpResult = await frameOf('/cellB').evaluate(read('js/http.js').replaceAll('__PORT__', String(kernel.port)));
  console.log('\nhttp.js →', JSON.stringify(httpResult, null, 1));
  check('http.js: GET small', httpResult.getSmall.ok && httpResult.getSmall.status === 200);
  check('http.js: GET with credentials', httpResult.getSmallWithCredentials.ok);
  check('http.js: GET 8 MB', httpResult.get8MB.ok && httpResult.get8MB.bytes === 8000000);
  check('http.js: POST 8 MB', httpResult.post8MB.ok && /8000000/.test(httpResult.post8MB.body));
  check('http.js: WebSocket echo', httpResult.webSocket.ok && httpResult.webSocket.echo === 'ping', JSON.stringify(httpResult.webSocket));
  check('http.js: service-worker route fails cleanly outside Colab', httpResult.serviceWorkerLocalhost.ok === false);

  // comm.html (against the mock)
  await frameOf('/cellC').waitForFunction(() => document.getElementById('gigi-probe-comm').textContent.includes('{'), null, { timeout: 30000 });
  const commText = await frameOf('/cellC').evaluate(() => document.getElementById('gigi-probe-comm').textContent);
  const commSent = await frameOf('/cellC').evaluate(() => window.mockCommSent);
  console.log('\ncomm.html →', commText);
  const final = commSent.find((m) => m.final);
  check('comm.html: three echo round trips', final && final.final.comm.length === 3 && final.final.comm.every((s) => s.ok));
  check('comm.html: invokeFunction result captured', final && /"mock":true/.test(final.final.invoke.result));

  // persist.html
  await top.waitForTimeout(1500);
  const frames = await frameOf('/cellD').evaluate(() => (self.gigiProbeFrames ? self.gigiProbeFrames() : -1));
  const persistText = await frameOf('/cellD').evaluate(() => document.getElementById('gigi-probe-persist').textContent);
  console.log('\npersist.html →', persistText, '| frames:', frames);
  check('persist.html: WebGPU canvas animates', frames > 0);
  check('persist.html: reports output age', /output is 4\d s old|output is 4\ds old/.test(persistText));
  await top.screenshot({ path: path.join(HERE, 'test_local.png') });

  await browser.close();
  frameServer.close(); topServer.close(); kernel.proc.kill();
  const failed = checks.filter((c) => !c.ok);
  console.log(`\n${checks.length - failed.length}/${checks.length} checks passed`);
  process.exit(failed.length ? 1 : 0);
}

main().catch((e) => { console.error(e); process.exit(1); });
