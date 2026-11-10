#!/usr/bin/env python3
"""Turn a Gigi-generated WebGPU app folder into ONE self-contained HTML fragment.

Prototype of the "snapshot" output: everything the page needs (JS modules, WGSL inside
them, CSS, input images) is inlined, so the fragment can be stored as a notebook cell
output and run again later with no kernel and no web server.

How: each ES module becomes a Blob URL, with its import specifiers rewritten to the Blob
URLs of the modules it depends on; asset fetches are answered from inlined data URLs.

Usage: gigi_snapshot.py <generated WebGPU root> <app dir relative to root> <out.html>
"""
import base64
import json
import mimetypes
import pathlib
import posixpath
import re
import sys

root = pathlib.Path(sys.argv[1]).resolve()
app_rel = sys.argv[2].strip("/")
out_path = pathlib.Path(sys.argv[3])
app = root / app_rel

index = (app / "index.html").read_text()
head_script = re.search(r'<script type="module">(.*?)</script>', index, re.S).group(1)
body = re.search(r"<body>(.*?)</body>", index, re.S).group(1)
css_href = re.search(r'<link rel="stylesheet" href="([^"]+)"', index).group(1)
css = (app / css_href).resolve().read_text()

IMPORT_RE = re.compile(r'''(\bfrom\s*|\bimport\s*\(\s*|\bimport\s+)(["'])(\.{1,2}/[^"']+)\2''')

modules = {}  # path relative to root -> source


def collect(source, base_dir):
    """Record every relative module this source imports (recursively)."""
    for m in IMPORT_RE.finditer(source):
        rel = posixpath.normpath(posixpath.join(base_dir, m.group(3)))
        if rel not in modules:
            modules[rel] = (root / rel).read_text()
            collect(modules[rel], posixpath.dirname(rel))


collect(head_script, app_rel)

# Assets: every non-code file next to the app (and in its assets/ folder).
assets = {}
for f in sorted(list(app.glob("*")) + list(app.glob("assets/*"))):
    if f.is_file() and not f.is_symlink() and f.suffix.lower() in (".png", ".bin", ".csv", ".bvh", ".jpg", ".hdr"):
        mime = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        assets[f.relative_to(app).as_posix().lower()] = f"data:{mime};base64," + base64.b64encode(f.read_bytes()).decode()

payload = {"modules": modules, "main": head_script, "mainDir": app_rel, "assets": assets}

LOADER = r"""
const P = __PAYLOAD__;
const IMPORT_RE = /(\bfrom\s*|\bimport\s*\(\s*|\bimport\s+)(["'])(\.{1,2}\/[^"']+)\2/g;
const norm = (base, spec) => {
  const parts = (base ? base.split('/') : []).concat(spec.split('/'));
  const out = [];
  for (const p of parts) { if (p === '..') out.pop(); else if (p !== '.' && p !== '') out.push(p); }
  return out.join('/');
};
const urls = {};
const blobFor = (path) => {
  if (urls[path]) return urls[path];
  const dir = path.split('/').slice(0, -1).join('/');
  const src = P.modules[path].replace(IMPORT_RE, (m, lead, q, spec) => lead + q + blobFor(norm(dir, spec)) + q);
  return (urls[path] = URL.createObjectURL(new Blob([src], { type: 'text/javascript' })));
};
// Answer asset requests from the inlined copies (names compared case-insensitively,
// with "a/../b" collapsed, which is what the Windows-authored paths rely on).
const realFetch = self.fetch.bind(self);
self.fetch = (input, init) => {
  if (typeof input === 'string' && !/^(https?:|data:|blob:)/.test(input)) {
    const key = norm('', input).toLowerCase();
    if (P.assets[key]) return realFetch(P.assets[key], init);
  }
  return realFetch(input, init);
};
const main = P.main.replace(IMPORT_RE, (m, lead, q, spec) => lead + q + blobFor(norm(P.mainDir, spec)) + q);
self.gigiSnapshot = { started: Date.now(), modules: Object.keys(P.modules).length, assets: Object.keys(P.assets).length };
self.addEventListener('error', (e) => { self.gigiSnapshot.error = String(e.message); });
self.addEventListener('unhandledrejection', (e) => { self.gigiSnapshot.error = String(e.reason); });
import(URL.createObjectURL(new Blob([main], { type: 'text/javascript' })))
  .then(() => { self.gigiSnapshot.mainLoaded = true; })
  .catch((e) => { self.gigiSnapshot.error = String(e); });
"""

# JSON is a valid JavaScript expression; "</" is escaped so the data cannot end the script tag.
payload_js = json.dumps(payload).replace("</", "<\\/")
html = (
    f"<style>{css}</style>\n{body}\n"
    f'<script type="module">{LOADER.replace("__PAYLOAD__", payload_js)}</script>\n'
)
out_path.write_text(html)
print(f"wrote {out_path}: {len(html) / 1e6:.2f} MB, {len(modules)} modules "
      f"({sum(map(len, modules.values())) / 1e3:.0f} KB), {len(assets)} assets")
