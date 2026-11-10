// Runs in the output frame of a LATER cell. Tries to reach "frame A", which an
// earlier cell created, by two routes:
//   direct    - walk parent.frames and call a function on the sibling's window
//               (what Colab's own google.colab.html proxy does)
//   broadcast - BroadcastChannel, which only needs the frames to share an origin
(async () => {
  const out = { direct: null, broadcast: null };
  try {
    for (let i = 0; i < parent.frames.length; i++) {
      try {
        const w = parent.frames[i];
        if (w !== self && w.gigiProbeMarker === 'frame-A') {
          out.direct = w.gigiProbePing('direct');
          break;
        }
      } catch (e) { /* cross-origin frame: skip */ }
    }
  } catch (e) { out.directError = String(e); }
  try {
    const bc = new BroadcastChannel('gigi-probe');
    out.broadcast = await new Promise((resolve) => {
      const timer = setTimeout(() => resolve(null), 2000);
      bc.onmessage = (ev) => {
        if (ev.data && ev.data.pong) { clearTimeout(timer); resolve('pong:' + ev.data.pong); }
      };
      bc.postMessage({ ping: 'broadcast' });
    });
    bc.close();
  } catch (e) { out.broadcastError = String(e); }
  return out;
})()
