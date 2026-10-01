// How many CPU threads generation gets. Kept free of browser globals so the node tests can
// cover it; `worker.js` passes in what it reads from `self` and `navigator`.
//
// The default is deliberately small and the same everywhere. Decoding one utterance issues
// many small operators, and past a few threads handing them out costs more than splitting
// saves; a thread that lands on an efficiency core holds back every operator it shares.
// Three stays within the big cores of current phones and laptops. A page that knows its
// devices better passes `threads` itself.

/** Threads in all by default: two workers plus the one that owns the model. */
export const DEFAULT_THREADS = 3;

/**
 * @param {object} env
 * @param {number | 'auto'} [env.requested] `LoadOptions.threads`.
 * @param {boolean} env.isolated Whether the page is cross-origin isolated, which wasm threads
 *   need: without it there is no `SharedArrayBuffer`.
 * @param {number} [env.hardwareConcurrency]
 * @returns {{ threads: number, reason: string }} Threads in all, counting the one that owns
 *   the model, and why that many.
 */
export function chooseThreads({ requested = 'auto', isolated, hardwareConcurrency }) {
  if (requested === 1) return { threads: 1, reason: 'requested' };
  if (!isolated) return { threads: 1, reason: 'the page is not cross-origin isolated' };
  if (!(hardwareConcurrency > 1)) return { threads: 1, reason: 'the browser reports one core or none' };
  if (requested !== 'auto') return { threads: Math.min(requested, hardwareConcurrency), reason: 'requested' };
  return { threads: Math.min(DEFAULT_THREADS, hardwareConcurrency), reason: 'default' };
}
