// Runs in an output frame WHILE the Python cell is still executing.
// Talks to a small HTTP server that lives in a background thread of the kernel,
// through Colab's kernel-port proxy. The port number is substituted on the last line.
(async (port) => {
  const out = {};
  const mask = (s) => String(s).replace(/[a-z0-9]{10,}/gi, '<id>');
  const timed = async (label, fn) => {
    const t0 = performance.now();
    try {
      const extra = await fn();
      out[label] = Object.assign({ ok: true, ms: Math.round(performance.now() - t0) }, extra || {});
    } catch (e) {
      out[label] = { ok: false, ms: Math.round(performance.now() - t0), error: String(e).slice(0, 200) };
    }
  };
  let base = null;
  await timed('proxyPort', async () => {
    base = await google.colab.kernel.proxyPort(port, { cache: false });
    return { urlPattern: mask(base) };
  });
  if (base) {
    const u = (p) => new URL(p, base).toString();
    await timed('getSmall', async () => {
      const r = await fetch(u('small'));
      return { status: r.status, body: (await r.text()).slice(0, 80) };
    });
    await timed('getSmallWithCredentials', async () => {
      const r = await fetch(u('small'), { credentials: 'include' });
      return { status: r.status };
    });
    await timed('get8MB', async () => {
      const r = await fetch(u('bytes?n=8000000'));
      return { status: r.status, bytes: (await r.arrayBuffer()).byteLength };
    });
    await timed('post8MB', async () => {
      const r = await fetch(u('upload'), { method: 'POST', body: new Blob([new Uint8Array(8000000)]) });
      return { status: r.status, body: (await r.text()).slice(0, 80) };
    });
    await timed('webSocket', () => new Promise((resolve, reject) => {
      const ws = new WebSocket(u('ws').replace(/^http/, 'ws'));
      const timer = setTimeout(() => { try { ws.close(); } catch (e) {} reject(new Error('timeout')); }, 8000);
      ws.onopen = () => ws.send('ping');
      ws.onmessage = (ev) => { clearTimeout(timer); ws.close(); resolve({ echo: String(ev.data) }); };
      ws.onerror = () => { clearTimeout(timer); reject(new Error('websocket error')); };
    }));
  }
  // The route google.colab.html uses: the Colab service worker maps https://localhost:<port>.
  await timed('serviceWorkerLocalhost', async () => {
    const r = await fetch('https://localhost:' + port + '/small');
    return { status: r.status, body: (await r.text()).slice(0, 80) };
  });
  return out;
})(__PORT__)
