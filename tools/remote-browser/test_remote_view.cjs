// End-to-end check of the remote view, with a headless browser playing the person:
// open the remote view page, enter the access code, wait for a frame, click into a form
// field on the remote screen and type. The form page reports what it received to its
// own tiny server, which is how this test knows the input arrived.
//
// Expects tools/remote-browser/start.sh to be running with the form page as its URL:
//   node test_remote_view.cjs serve              -> serves the form page, prints what it receives
//   node test_remote_view.cjs drive <port> <code> <formPort> <screenshot.png>
const http = require('http');
const { chromium } = require('playwright');

const FORM = `<!doctype html><meta charset="utf-8"><title>remote view test form</title>
<body style="font: 20px system-ui; margin: 40px">
<h1>remote view test</h1>
<input id="field" style="font: inherit; width: 600px; padding: 12px" placeholder="type here" autofocus>
<button id="btn" style="font: inherit; padding: 12px 24px">Submit</button>
<p id="echo"></p>
<script>
  const field = document.getElementById('field');
  field.addEventListener('input', () => { document.getElementById('echo').textContent = 'typed: ' + field.value; });
  document.getElementById('btn').addEventListener('click', () => {
    fetch('/got?v=' + encodeURIComponent(field.value));
    document.getElementById('echo').textContent = 'submitted: ' + field.value;
  });
</script>`;

if (process.argv[2] === 'serve') {
  const server = http.createServer((req, res) => {
    if (req.url.startsWith('/got')) console.log('GOT ' + decodeURIComponent(req.url.split('v=')[1] || ''));
    res.writeHead(200, { 'content-type': 'text/html' });
    res.end(FORM);
  }).listen(parseInt(process.argv[3] || '8799', 10), '127.0.0.1', () => console.log('form on ' + server.address().port));
} else {
  const [, , , port, code, formPort, shot] = process.argv;
  (async () => {
    const browser = await chromium.launch();
    const page = await browser.newPage({ viewport: { width: 1320, height: 1000 } });
    await page.goto(`http://127.0.0.1:${port}/`);
    await page.fill('#code', 'wrong-code');
    await page.click('button[type=submit]');
    await page.waitForTimeout(500);
    const rejected = await page.textContent('#gate-error');
    await page.fill('#code', code);
    await page.click('button[type=submit]');
    await page.waitForSelector('#view', { state: 'visible', timeout: 10000 });
    await page.waitForTimeout(2500);

    // Find the form field on the remote screen: it is drawn at a known place in the page,
    // but the window manager adds decorations, so locate it by colour instead of guessing.
    const canvas = page.locator('#screen');
    const box = await canvas.boundingBox();
    const size = await canvas.evaluate((c) => [c.width, c.height]);
    const title = await page.textContent('#status');
    // Click the input (roughly: left part of the page, below the heading), type, then click Submit.
    const toClient = (x, y) => [box.x + x * box.width / size[0], box.y + y * box.height / size[1]];
    const target = await canvas.evaluate((c) => {
      // The text field is a wide white box with a darker 1px border; scan for its left edge row.
      const d = c.getContext('2d').getImageData(0, 0, c.width, c.height);
      const lum = (x, y) => { const i = (y * c.width + x) * 4; return d.data[i] + d.data[i + 1] + d.data[i + 2]; };
      for (let y = 120; y < c.height - 40; y += 2) {
        let run = 0;
        for (let x = 30; x < 700; x++) { if (lum(x, y) < 600) run++; else run = 0; if (run > 300) return { x: 200, y: y + 20 }; }
      }
      return null;
    });
    const click = async (x, y) => { const [cx, cy] = toClient(x, y); await page.mouse.click(cx, cy); };
    await click(target ? target.x : 200, target ? target.y : 260);
    await page.waitForTimeout(400);
    await canvas.focus();
    await page.keyboard.type('Hello 你好 #42');
    await page.waitForTimeout(600);
    // Also the paste box, which is the route meant for secrets.
    await page.fill('#typed', ' via-box');
    await page.click('#send');
    await page.waitForTimeout(800);
    await canvas.focus();
    await page.keyboard.press('Tab');
    await page.keyboard.press('Enter');   // focus moved to Submit; Enter presses it
    await page.waitForTimeout(1500);
    await page.screenshot({ path: shot });
    console.log(JSON.stringify({ rejectedWrongCode: rejected, title, remoteScreen: size, fieldFoundAt: target }, null, 1));
    await browser.close();
  })().catch((e) => { console.error(String(e).slice(0, 1500)); process.exit(1); });
}
