// How many CPU threads generation gets. Kept free of browser globals so the node tests can
// cover it; `worker.js` passes in what it reads from `self` and `navigator`.
//
// The count is deliberately not one per core. Decoding one utterance issues many small
// operators, and past a few threads handing them out costs more than splitting saves. On
// phones it is lower still: a thread that lands on an efficiency core holds back every
// operator it shares, so the default stays within the big cores a phone is sure to have.

/**
 * Threads in all on a desktop, at most: three workers plus the one that owns the model.
 * Laptop chips commonly have four performance cores, and a fifth thread lands on an
 * efficiency core and slows the frame down rather than up.
 */
export const MAX_THREADS = 4;
/** Threads in all on a phone: two workers plus the one that owns the model. */
export const MOBILE_THREADS = 3;

// iPads on iPadOS 13 and later send a desktop Mac user agent, so they fall under the desktop
// cap, which their cores can take.
const MOBILE = /Android|iPhone|iPad|Mobile/;

/**
 * @param {object} env
 * @param {number | 'auto'} [env.requested] `LoadOptions.threads`.
 * @param {boolean} env.isolated Whether the page is cross-origin isolated, which wasm threads
 *   need: without it there is no `SharedArrayBuffer`.
 * @param {number} [env.hardwareConcurrency]
 * @param {string} [env.userAgent]
 * @returns {{ threads: number, reason: string }} Threads in all, counting the one that owns
 *   the model, and why that many.
 */
export function chooseThreads({ requested = 'auto', isolated, hardwareConcurrency, userAgent = '' }) {
  if (requested === 1) return { threads: 1, reason: 'requested' };
  if (!isolated) return { threads: 1, reason: 'the page is not cross-origin isolated' };
  if (!(hardwareConcurrency > 1)) return { threads: 1, reason: 'the browser reports one core or none' };
  const cores = hardwareConcurrency;
  if (requested !== 'auto') return { threads: Math.min(requested, cores), reason: 'requested' };
  if (MOBILE.test(userAgent)) return { threads: Math.min(MOBILE_THREADS, cores), reason: 'phone default' };
  return { threads: Math.min(MAX_THREADS, cores), reason: 'default' };
}
