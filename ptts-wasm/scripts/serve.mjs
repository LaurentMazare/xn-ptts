#!/usr/bin/env node
// Serve a directory over HTTP with the two headers that make a page cross-origin isolated,
// which the package's threaded build needs. `python3 -m http.server` sends neither, and a
// demo served by it would always run on one thread.
//
//   node scripts/serve.mjs site 8080

import { createReadStream, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join, normalize, resolve, sep } from 'node:path';

const root = resolve(process.argv[2] ?? '.');
const port = Number(process.argv[3] ?? 8080);
const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json',
  '.wasm': 'application/wasm',
};

createServer((req, res) => {
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  let path;
  try {
    path = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
  } catch {
    // A malformed escape, which would otherwise throw out of the handler and stop the server.
    return res.writeHead(400).end();
  }
  let file = normalize(join(root, path));
  // `normalize` resolves `..`, including an escaped `..%2f`. Comparing against `root` plus a
  // separator keeps the result inside `root`, not in a sibling whose name starts the same.
  if (file !== root && !file.startsWith(root + sep)) return res.writeHead(403).end();
  try {
    if (statSync(file).isDirectory()) file = join(file, 'index.html');
    const { size } = statSync(file);
    res.writeHead(200, { 'Content-Type': TYPES[extname(file)] ?? 'application/octet-stream', 'Content-Length': size });
    createReadStream(file).pipe(res);
  } catch {
    res.writeHead(404).end();
  }
}).listen(port, () => console.log(`serving ${root} on http://localhost:${port}, cross-origin isolated`));
