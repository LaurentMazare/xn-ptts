// The download cache's own logic, against a fake Cache API and a fake network, so it runs
// under plain `node --test` with no browser and no model.

import { test } from 'node:test';
import assert from 'node:assert/strict';

const { fetchBytes } = await import('../fetch.js');

function fakeNetwork(body) {
  const store = new Map();
  let fetches = 0;
  globalThis.caches = {
    async open() {
      return {
        match: async (url) => store.get(url)?.clone(),
        put: async (url, response) => void store.set(url, new Response(await response.arrayBuffer())),
      };
    },
  };
  globalThis.fetch = async () => {
    fetches++;
    return new Response(body, { headers: { 'content-length': String(body.length) } });
  };
  return { fetches: () => fetches };
}

test('fetchBytes downloads once, then serves from the cache', async () => {
  const body = new Uint8Array(3000).map((_, i) => i % 251);
  const net = fakeNetwork(body);
  const progress = [];
  const first = await fetchBytes('https://x/model', { onProgress: (p) => progress.push(p) });
  const second = await fetchBytes('https://x/model', { onProgress: (p) => progress.push(p) });
  assert.deepEqual(first, body);
  assert.deepEqual(second, body);
  assert.equal(net.fetches(), 1);
  assert.equal(progress.at(-1).cached, true);
  assert.equal(progress.find((p) => !p.cached).total, 3000);

  await fetchBytes('https://x/model', { cache: false });
  assert.equal(net.fetches(), 2, 'cache: false always downloads');
});

test('fetchBytes still works where the Cache API is missing', async () => {
  fakeNetwork(new Uint8Array([7, 8]));
  delete globalThis.caches;
  assert.deepEqual(await fetchBytes('https://x/y'), new Uint8Array([7, 8]));
});

test('fetchBytes reports HTTP errors', async () => {
  globalThis.fetch = async () => new Response('nope', { status: 404 });
  await assert.rejects(fetchBytes('https://x/missing', { cache: false }), /HTTP 404/);
});

// ---- wav ----
