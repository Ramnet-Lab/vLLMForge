/* Tiny DOM toolkit. No framework, no build step — the whole point is that this
   directory can be edited on the box and reloaded. `h` is the only abstraction
   that matters: it builds elements from (tag, props, children). */

export function h(tag, props = null, ...children) {
  const el = document.createElement(tag);
  if (props) {
    for (const [key, value] of Object.entries(props)) {
      if (value === null || value === undefined || value === false) continue;
      if (key === 'class') el.className = value;
      else if (key === 'style' && typeof value === 'object') Object.assign(el.style, value);
      else if (key === 'dataset') Object.assign(el.dataset, value);
      else if (key.startsWith('on') && typeof value === 'function') {
        el.addEventListener(key.slice(2).toLowerCase(), value);
      } else if (key === 'html') el.innerHTML = value;
      else if (key in el && key !== 'list') el[key] = value;
      else el.setAttribute(key, value === true ? '' : value);
    }
  }
  append(el, children);
  return el;
}

function append(el, children) {
  for (const child of children.flat(4)) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}

export const frag = (...children) => {
  const f = document.createDocumentFragment();
  append(f, children);
  return f;
};

export function clear(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
  return el;
}

export function mount(el, ...children) {
  clear(el);
  append(el, children);
  return el;
}

/* --- formatting -------------------------------------------------------- */

const UNITS = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB'];

export function bytes(value, digits = 1) {
  if (!Number.isFinite(value) || value <= 0) return '0 B';
  const index = Math.min(UNITS.length - 1, Math.floor(Math.log(value) / Math.log(1024)));
  const scaled = value / 1024 ** index;
  return `${scaled.toFixed(index === 0 ? 0 : digits)} ${UNITS[index]}`;
}

export function count(value) {
  if (!Number.isFinite(value)) return '—';
  if (value >= 1e9) return `${(value / 1e9).toFixed(1)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}k`;
  return String(value);
}

export function duration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '—';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  if (m < 60) return `${m}m ${Math.round(seconds % 60)}s`;
  const hrs = Math.floor(m / 60);
  if (hrs < 24) return `${hrs}h ${m % 60}m`;
  return `${Math.floor(hrs / 24)}d ${hrs % 24}h`;
}

export function ago(timestamp) {
  if (!timestamp) return '—';
  const seconds = Date.now() / 1000 - (typeof timestamp === 'number' ? timestamp : Date.parse(timestamp) / 1000);
  if (!Number.isFinite(seconds)) return '—';
  if (seconds < 0) return 'just now';
  return `${duration(seconds)} ago`;
}

export function when(timestamp) {
  if (!timestamp) return '—';
  const date = typeof timestamp === 'number' ? new Date(timestamp * 1000) : new Date(timestamp);
  return Number.isNaN(date.getTime()) ? '—' : date.toLocaleString();
}

export const pct = (value, digits = 0) => `${(value * 100).toFixed(digits)}%`;

/* --- components -------------------------------------------------------- */

export const badge = (status, label) =>
  h('span', { class: `badge ${status || 'absent'}` }, label ?? status ?? 'unknown');

export const stat = (label, value, hint) =>
  h('div', { class: 'stat' },
    h('span', { class: 'label' }, label),
    h('span', { class: 'value' }, value),
    hint ? h('span', { class: 'hint' }, hint) : null);

export function panel(title, { sub, actions, body, foot, flush } = {}) {
  return h('section', { class: 'panel' },
    title || actions
      ? h('div', { class: 'panel-head' },
          h('div', { class: 'row' }, h('h2', null, title), sub ? h('span', { class: 'sub' }, sub) : null),
          actions ? h('div', { class: 'row' }, actions) : null)
      : null,
    h('div', { class: `panel-body${flush ? ' flush' : ''}` }, body),
    foot ? h('div', { class: 'panel-foot' }, foot) : null);
}

export const notice = (level, ...content) =>
  h('div', { class: `notice ${level}` },
    h('span', { class: 'mark' }, { ok: '✓', warn: '⚠', danger: '✕', info: 'ℹ' }[level] || 'ℹ'),
    h('div', null, ...content));

export const empty = (title, message, action) =>
  h('div', { class: 'empty' }, h('h3', null, title), message ? h('p', null, message) : null, action);

export const spinner = () => h('span', { class: 'spinner' });

export function field(label, control, { help, flag, inline, changed } = {}) {
  return h('div', { class: `field${inline ? ' inline' : ''}${changed ? ' changed' : ''}` },
    h('label', null,
      h('span', null, label),
      flag ? h('span', { class: 'flagname' }, flag) : null),
    control,
    help ? h('span', { class: 'help' }, help) : null);
}

/* --- toasts ------------------------------------------------------------ */

export function toast(message, { level = 'ok', title = '', timeout = 5200 } = {}) {
  const host = document.getElementById('toasts');
  const el = h('div', { class: `toast ${level}` },
    title ? h('div', { class: 't-title' }, title) : null,
    h('div', null, message));
  el.addEventListener('click', () => el.remove());
  host.append(el);
  if (timeout) setTimeout(() => el.remove(), timeout);
  return el;
}

/* --- modal ------------------------------------------------------------- */

export function modal(title, body, { actions = [], wide = false } = {}) {
  // A dialog per call rather than one shared element: a confirm opened from
  // inside an open modal used to replace its content, and closing the confirm
  // tore down the modal underneath it too.
  const dialog = h('dialog', { class: 'modal' });
  if (wide) dialog.style.maxWidth = 'min(96vw, 1240px)';

  const close = () => {
    dialog.close();
    dialog.remove();
  };

  mount(dialog,
    h('div', { class: 'modal-head' },
      h('h2', null, title),
      h('button', { class: 'icon-btn', onClick: close, title: 'Close' }, '✕')),
    h('div', { class: 'modal-body' }, body),
    actions.length ? h('div', { class: 'modal-foot' }, actions) : null);

  document.body.append(dialog);
  dialog.showModal();
  return { close, dialog };
}

export function confirmDialog(title, message, { danger = true, confirmLabel = 'Confirm' } = {}) {
  return new Promise((resolve) => {
    const ctl = modal(title, h('p', { class: 'muted' }, message), {
      actions: [
        h('button', { onClick: () => { ctl.close(); resolve(false); } }, 'Cancel'),
        h('button', {
          class: danger ? 'btn-danger' : 'btn-primary',
          onClick: () => { ctl.close(); resolve(true); },
        }, confirmLabel),
      ],
    });
    ctl.dialog.addEventListener('close', () => resolve(false), { once: true });
  });
}

/* --- log rendering ----------------------------------------------------- */

const LINE_CLASS = [
  [/^\$ /, 'l-cmd'],
  [/\b(ERROR|CRITICAL|Traceback|Exception|failed|FAILED|error:)\b/, 'l-err'],
  [/\b(WARNING|WARN|warn)\b/, 'l-warn'],
  [/\b(DONE|Completed|succeeded|Application startup complete)\b/, 'l-ok'],
];

export function logLine(text) {
  const cls = LINE_CLASS.find(([re]) => re.test(text));
  return h('div', { class: cls ? cls[1] : '' }, text);
}

export function logBox(lines = []) {
  const box = h('div', { class: 'logbox' }, lines.map(logLine));
  box.append = (text) => {
    const stuck = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
    Element.prototype.append.call(box, logLine(text));
    while (box.childElementCount > 4000) box.firstElementChild.remove();
    if (stuck) box.scrollTop = box.scrollHeight;
  };
  queueMicrotask(() => { box.scrollTop = box.scrollHeight; });
  return box;
}

/* --- misc -------------------------------------------------------------- */

export function copyButton(text, label = 'Copy') {
  return h('button', {
    class: 'btn-sm btn-ghost',
    onClick: async (event) => {
      try {
        await navigator.clipboard.writeText(typeof text === 'function' ? text() : text);
        const btn = event.currentTarget;
        const original = btn.textContent;
        btn.textContent = 'Copied';
        setTimeout(() => { btn.textContent = original; }, 1300);
      } catch {
        toast('Clipboard is unavailable in this context', { level: 'warn' });
      }
    },
  }, label);
}

export function debounce(fn, ms = 260) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}

export const escapeHtml = (value) =>
  String(value).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

/** Register a view's own CSS once. Views own their layout rules this way
 *  instead of all appending to one stylesheet. */
export function ensureStyles(id, css) {
  if (document.getElementById(`style-${id}`)) return;
  document.head.append(h('style', { id: `style-${id}` }, css));
}

const SVG_NS = 'http://www.w3.org/2000/svg';

/** Build an SVG element. Separate from h() because SVG needs its own namespace
 *  and takes attributes rather than properties — createElement would produce an
 *  HTMLUnknownElement that renders as nothing. */
export function svg(tag, props = null, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  if (props) {
    for (const [key, value] of Object.entries(props)) {
      if (value === null || value === undefined || value === false) continue;
      if (key.startsWith('on') && typeof value === 'function') {
        el.addEventListener(key.slice(2).toLowerCase(), value);
      } else {
        el.setAttribute(key, value === true ? '' : String(value));
      }
    }
  }
  for (const child of children.flat(3)) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}
