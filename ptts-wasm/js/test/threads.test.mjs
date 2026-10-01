// The thread-count policy in threads.js, without a browser.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { chooseThreads, MAX_THREADS, MOBILE_THREADS } from '../threads.js';

const DESKTOP = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/154.0 Safari/537.36';
const ANDROID = 'Mozilla/5.0 (Linux; Android 16; SM-S901B) AppleWebKit/537.36 Chrome/154.0 Mobile Safari/537.36';

test('one thread without cross-origin isolation, whatever was asked for', () => {
  for (const requested of ['auto', 1, 4]) {
    const got = chooseThreads({ requested, isolated: false, hardwareConcurrency: 10, userAgent: DESKTOP });
    assert.equal(got.threads, 1);
  }
  assert.match(chooseThreads({ isolated: false, hardwareConcurrency: 10 }).reason, /not cross-origin isolated/);
});

test('auto: capped on a desktop, lower on a phone, never more than the cores', () => {
  assert.deepEqual(chooseThreads({ isolated: true, hardwareConcurrency: 10, userAgent: DESKTOP }), {
    threads: MAX_THREADS,
    reason: 'default',
  });
  assert.deepEqual(chooseThreads({ isolated: true, hardwareConcurrency: 8, userAgent: ANDROID }), {
    threads: MOBILE_THREADS,
    reason: 'phone default',
  });
  assert.equal(chooseThreads({ isolated: true, hardwareConcurrency: 2, userAgent: DESKTOP }).threads, 2);
  assert.equal(chooseThreads({ isolated: true, hardwareConcurrency: undefined, userAgent: DESKTOP }).threads, 1);
});

test('a requested count is honoured up to the number of cores', () => {
  assert.deepEqual(chooseThreads({ requested: 1, isolated: true, hardwareConcurrency: 10 }), { threads: 1, reason: 'requested' });
  assert.equal(chooseThreads({ requested: 6, isolated: true, hardwareConcurrency: 10 }).threads, 6);
  assert.equal(chooseThreads({ requested: 16, isolated: true, hardwareConcurrency: 10 }).threads, 10);
});
