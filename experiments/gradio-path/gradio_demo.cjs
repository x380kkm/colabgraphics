// Drive the Gradio-hosted Gigi viewer in Chromium and check that a Gradio slider
// really changes what WebGPU renders.
// Usage: node gradio_demo.cjs <python> <generated WebGPU root> <screenshot prefix>
const path = require('path');
const { spawn } = require('child_process');
const { chromium } = require('playwright');

const [python, root, shotPrefix] = process.argv.slice(2);
const PORT = 7861;

function startGradio() {
  return new Promise((resolve, reject) => {
    const proc = spawn(python, [path.join(__dirname, 'gradio_app.py'), root, String(PORT)],
      { env: { ...process.env, GRADIO_ANALYTICS_ENABLED: 'False', NO_PROXY: '127.0.0.1,localhost', no_proxy: '127.0.0.1,localhost' } });
    let log = '';
    const onData = (d) => { log += d; if (/Running on local URL/.test(log)) resolve(proc); };
    proc.stdout.on('data', onData); proc.stderr.on('data', onData);
    proc.once('exit', (c) => reject(new Error('gradio exited ' + c + '\n' + log)));
    setTimeout(() => reject(new Error('gradio did not start\n' + log)), 90000);
  });
}

// Mean colour of the Gigi canvas region, read from a screenshot-free path: sample the
// rendered page pixels through a 2D canvas in the top document.
async function canvasSignature(frame) {
  return frame.evaluate(async () => {
    const c = document.getElementById('mainCanvas');
    const bitmap = await createImageBitmap(c);
    const probe = new OffscreenCanvas(64, 64);
    const ctx = probe.getContext('2d');
    ctx.drawImage(bitmap, 0, 0, 64, 64);
    const data = ctx.getImageData(0, 0, 64, 64).data;
    // Sum of absolute horizontal differences: a sharp image scores high, a blurred one low.
    let edge = 0;
    for (let y = 0; y < 64; y++) for (let x = 1; x < 64; x++) {
      const i = (y * 64 + x) * 4;
      edge += Math.abs(data[i] - data[i - 4]) + Math.abs(data[i + 1] - data[i - 3]) + Math.abs(data[i + 2] - data[i - 2]);
    }
    return Math.round(edge / (64 * 63 * 3) * 100) / 100;
  });
}

(async () => {
  const gradio = await startGradio();
  const browser = await chromium.launch({ args: [
    '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=swiftshader',
    '--use-vulkan=swiftshader', '--ignore-gpu-blocklist', '--disable-vulkan-surface',
  ] });
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 }, deviceScaleFactor: 2 });
  const problems = [];
  page.on('console', (m) => { if (m.type() === 'error') problems.push('console: ' + m.text().slice(0, 200)); });
  page.on('pageerror', (e) => problems.push('pageerror: ' + e.message.slice(0, 200)));
  await page.goto(`http://127.0.0.1:${PORT}/`);
  await page.waitForSelector('#gigi-view', { timeout: 60000 });
  const frame = await (await page.$('#gigi-view')).contentFrame();
  await frame.waitForFunction(() => /FPS: \d/.test(document.getElementById('FPS').textContent), null, { timeout: 60000 });
  await page.waitForTimeout(1500);

  const before = { radius: await frame.$eval('#inputBox_radius', (e) => e.value), edge: await canvasSignature(frame) };
  await page.screenshot({ path: shotPrefix + '_radius20.png' });

  // Move the Gradio slider to 2 through its number box, as a person would type it.
  const numberBox = page.locator('input[type="number"]').first();
  await numberBox.fill('2');
  await numberBox.press('Enter');
  await page.waitForTimeout(2500);
  const after = { radius: await frame.$eval('#inputBox_radius', (e) => e.value), edge: await canvasSignature(frame) };
  const pythonNote = await page.locator('textarea').first().inputValue().catch(() => null);
  await page.screenshot({ path: shotPrefix + '_radius2.png' });

  console.log(JSON.stringify({ before, after, pythonNote, problems }, null, 1));
  await browser.close();
  gradio.kill();
})().catch((e) => { console.error(String(e).slice(0, 3000)); process.exit(1); });
