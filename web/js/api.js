/* Thin fetch layer. Every call returns parsed JSON or throws an ApiError whose
   message is the server's `detail`, because that is what the UI wants to show. */

const BASE = '/api';

export class ApiError extends Error {
  constructor(message, { status, detail } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

async function request(method, path, body, { raw = false } = {}) {
  const options = { method, headers: {} };
  if (body !== undefined) {
    options.headers['Content-Type'] = 'application/json';
    options.body = JSON.stringify(body);
  }
  const response = await fetch(BASE + path, options);
  if (raw) {
    if (!response.ok) throw new ApiError(await response.text(), { status: response.status });
    return response.text();
  }
  if (response.status === 204) return null;

  const text = await response.text();
  let payload = null;
  try { payload = text ? JSON.parse(text) : null; } catch { payload = { detail: text }; }

  if (!response.ok) {
    const detail = payload?.detail;
    const message = typeof detail === 'string'
      ? detail
      : detail?.message || payload?.message || `${method} ${path} failed (${response.status})`;
    throw new ApiError(message, { status: response.status, detail });
  }
  return payload;
}

export const get = (path) => request('GET', path);
export const getText = (path) => request('GET', path, undefined, { raw: true });
export const post = (path, body) => request('POST', path, body ?? {});
export const put = (path, body) => request('PUT', path, body ?? {});
export const patch = (path, body) => request('PATCH', path, body ?? {});
export const del = (path) => request('DELETE', path);

/* --- server-sent events ------------------------------------------------- */

/** Subscribe to an SSE endpoint. `handlers` maps event name -> callback.
 *  Returns a close function. Reconnects are handled by EventSource itself
 *  except for terminal streams, which call close() from their `end` handler. */
export function stream(path, handlers = {}) {
  const source = new EventSource(BASE + path);
  for (const [event, handler] of Object.entries(handlers)) {
    if (event === 'error') { source.addEventListener('error', handler); continue; }
    source.addEventListener(event, (message) => {
      let payload = null;
      try { payload = message.data ? JSON.parse(message.data) : null; } catch { payload = message.data; }
      handler(payload, message);
    });
  }
  return () => source.close();
}

/* --- POST that streams a response body (chat completions) --------------- */

export async function* postStream(path, body, signal) {
  const response = await fetch(BASE + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok || !response.body) {
    const text = await response.text();
    throw new ApiError(text || `stream failed (${response.status})`, { status: response.status });
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    // SSE frames are separated by a blank line; a frame may hold several
    // `data:` lines which the spec says to join with newlines.
    let split;
    while ((split = buffer.indexOf('\n\n')) !== -1) {
      const frame = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);
      const data = frame
        .split('\n')
        .filter((line) => line.startsWith('data:'))
        .map((line) => line.slice(5).trim())
        .join('\n');
      if (data) yield data;
    }
  }
}
