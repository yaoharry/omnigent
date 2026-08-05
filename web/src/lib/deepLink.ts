/**
 * `omnigent://` deep links, from the web side.
 *
 * The scheme is defined and consumed by the native shells — the desktop
 * parser (`electron/src/deepLink.js`, spec prose in `electron/README.md`
 * under "Deep links") and the iOS one (`ios/Omnigent/DeepLink.swift`).
 * This module is only the *emitter*, and must stay within what those
 * parsers accept:
 *
 *   - host (with port when non-default), never an http/https scheme — the
 *     shells infer it (`http` for loopback, `https` for remote) so a link
 *     and a pasted URL can't disagree;
 *   - path exactly `/c/<session_id>` — v1 accepts nothing else;
 *   - no workspace mount (`/ml/omnigents`), which is server-determined and
 *     rediscovered on open.
 */

import { getOmnigentTransformShareLink } from "@/lib/host";

/** Where a user without the desktop app installed goes to get it. */
export const DESKTOP_DOWNLOAD_URL = "https://omnigent.ai/download/mac";

/**
 * The absolute, basename-rebased web URL for a session — the thing a user
 * would copy to share it.
 *
 * In the embed the host transform returns a full URL (origin included);
 * standalone has no transform, so we prepend the origin ourselves.
 */
export function getShareableLink(sessionId: string, rebasePath: (path: string) => string): string {
  const path = rebasePath(`/c/${sessionId}`);
  const transform = getOmnigentTransformShareLink();
  return transform ? transform(path) : `${window.location.origin}${path}`;
}

/**
 * The `omnigent://<host>/c/<session_id>` link that opens this session in a
 * native shell.
 *
 * The host is read back off the shareable URL rather than off
 * `window.location` directly, so an embedded (host-transformed) origin and a
 * standalone one name the same server the shells key their window reuse off.
 */
export function getDeepLink(sessionId: string, rebasePath: (path: string) => string): string {
  const url = getShareableLink(sessionId, rebasePath);
  try {
    const { host } = new URL(url);
    return `omnigent://${host}/c/${sessionId}`;
  } catch {
    // Unparseable transform output: fall back to the current origin's host.
    return `omnigent://${window.location.host}/c/${sessionId}`;
  }
}
