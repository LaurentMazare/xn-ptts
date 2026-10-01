// The thread-count policy in threads.js, without a browser.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { chooseThreads, DEFAULT_THREADS } from '../threads.js';

test('one thread without cross-origin isolation, whatever was asked for', () => {
  for (const requested of ['auto', 1, 4]) {
    assert.equal(chooseThreads({ requested, isolated: false, hardwareConcurrency: 10 }).threads, 1);
  }
  assert.match(chooseThreads({ isolated: false, hardwareConcurrency: 10 }).reason, /not cross-origin isolated/);
});

test('auto: the default count, never more than the cores', () => {
  assert.deepEqual(chooseThreads({ isolated: true, hardwareConcurrency: 10 }), { threads: DEFAULT_THREADS, reason: 'default' });
  assert.deepEqual(chooseThreads({ isolated: true, hardwareConcurrency: 2 }), { threads: 2, reason: 'default' });
});

test('a requested count is honoured up to the number of cores', () => {
  assert.deepEqual(chooseThreads({ requested: 1, isolated: true, hardwareConcurrency: 10 }), { threads: 1, reason: 'requested' });
  assert.equal(chooseThreads({ requested: 6, isolated: true, hardwareConcurrency: 10 }).threads, 6);
  assert.equal(chooseThreads({ requested: 16, isolated: true, hardwareConcurrency: 10 }).threads, 10);
});

test('one thread, with its own reason, when the browser reports one core or none', () => {
  for (const hardwareConcurrency of [undefined, 0, 1]) {
    for (const requested of ['auto', 6]) {
      assert.deepEqual(chooseThreads({ requested, isolated: true, hardwareConcurrency }), {
        threads: 1,
        reason: 'the browser reports one core or none',
      });
    }
  }
});
