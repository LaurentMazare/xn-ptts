// WAV encoding and PCM joining: plain functions, plain assertions.

import { test } from 'node:test';
import assert from 'node:assert/strict';

const { encodeWav, concatPcm } = await import('../wav.js');

test('encodeWav writes a 16-bit mono header and clips', async () => {
  const wav = new DataView(await encodeWav(new Float32Array([0, 1, -1, 2]), 24000).arrayBuffer());
  const ascii = (o) => String.fromCharCode(...[0, 1, 2, 3].map((i) => wav.getUint8(o + i)));
  assert.equal(ascii(0), 'RIFF');
  assert.equal(ascii(8), 'WAVE');
  assert.equal(wav.getUint16(22, true), 1);
  assert.equal(wav.getUint32(24, true), 24000);
  assert.equal(wav.getUint32(40, true), 8);
  assert.deepEqual([0, 1, 2, 3].map((i) => wav.getInt16(44 + 2 * i, true)), [0, 32767, -32768, 32767]);
});

test('concatPcm joins frames in order', () => {
  assert.deepEqual(concatPcm([new Float32Array([1]), new Float32Array([2, 3])]), new Float32Array([1, 2, 3]));
  assert.deepEqual(concatPcm([]), new Float32Array(0));
});
