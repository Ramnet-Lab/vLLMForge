/* Jobs: every long-running container in one place — downloads, image builds,
   fine-tunes, Heretic runs.

   The progress dict is per-kind and deliberately unschema'd (see app/jobs.py),
   so nothing here knows what a "trial" or a "shard" is: it prefers percent,
   falls back to step/total, and otherwise just names the phase. New job kinds
   therefore render sensibly on the day they are added. */

import { get, getText, post, stream } from '../api.js';
import {
  h, mount, bytes, duration, ago, when, badge, panel, notice, empty,
  toast, modal, confirmDialog, logBox, ensureStyles,
} from '../ui.js';

const TERMINAL = new Set(['succeeded', 'failed', 'cancelled']);
const KINDS = [
  ['all', 'All'],
  ['download', 'Downloads'],
  ['finetune', 'Fine-tune'],
  ['heretic', 'Heretic'],
  ['build', 'Builds'],
];
const PAGE = 100;
const MAX_LIMIT = 500;

let ctx = null;
let closeAll = null;
let closeDetail = null;
let ticker = null;
const state = { jobs: [], byId: new Map(), kind: 'all', selected: null, limit: PAGE };
const els = {};

/* --- data --------------------------------------------------------------- */

/* One unfiltered fetch rather than one per kind tab: the tab counts and the
   sidebar badge both need every kind, and filtering 100 rows in the browser is
   free next to a round trip. */
async function load() {
  const payload = await get(`/jobs?limit=${state.limit}`);
  state.jobs = payload.jobs || [];
  state.byId = new Map(state.jobs.map((job) => [job.id, job]));
}

function upsert(job) {
  const existing = state.byId.get(job.id);
  state.byId.set(job.id, job);
  if (existing) state.jobs[state.jobs.indexOf(existing)] = job;
  else state.jobs.unshift(job);
}

const visible = () =>
  state.kind === 'all' ? state.jobs : state.jobs.filter((job) => job.kind === state.kind);

const activeCount = () => state.jobs.filter((job) => !TERMINAL.has(job.status)).length;

/* --- progress ----------------------------------------------------------- */

const finite = (value) => (Number.isFinite(value) ? value : null);

/** Fraction complete, 0-100, or null when the job's progress cannot say. */
const doneOf = (progress) => finite(progress.step) ?? finite(progress.files_done);
const totalOf = (progress) =>
  finite(progress.total) ?? finite(progress.total_steps) ?? finite(progress.files_total);

function percentOf(progress) {
  const direct = finite(progress.percent);
  if (direct !== null) return Math.max(0, Math.min(100, direct));
  const done = doneOf(progress);
  const total = totalOf(progress);
  if (done === null || !total) return null;
  return Math.max(0, Math.min(100, (100 * done) / total));
}

function stepLabel(progress) {
  const done = doneOf(progress);
  const total = totalOf(progress);
  return done !== null && total ? `${done}/${total}` : '';
}

function progressBlock(job) {
  const progress = job.progress || {};
  const percent = percentOf(progress);
  const running = !TERMINAL.has(job.status);
  if (percent === null && !running && !progress.phase) return null;

  const tone = job.status === 'failed' ? ' failed' : job.status === 'cancelled' ? ' stalled' : '';
  const bar = percent === null
    ? h('div', { class: `progress indeterminate${tone}` }, h('span', null))
    : h('div', { class: `progress${tone}` }, h('span', { style: { width: `${percent}%` } }));

  const steps = stepLabel(progress);
  const right = percent === null
    ? steps
    : `${percent.toFixed(percent < 10 ? 1 : 0)}%${steps ? `  ${steps}` : ''}`;
  return h('div', { class: 'job-progress' },
    h('div', { class: 'job-progress-label' },
      h('span', { class: 'truncate' }, progress.phase || (running ? job.status : '')),
      h('span', { class: 'nowrap' }, right)),
    bar);
}

/* Not every worker's progress bar reaches us as a transient `progress-line`:
   huggingface_hub's downloader newline-terminates each redraw, so the same bar
   arrives as an ordinary log line several times a second. Collapse a redraw onto
   its predecessor — matching the shape, not the numbers — so a long download
   leaves one moving bar instead of two thousand lines of history. */
const BAR = /[\u2588\u258f\u258e\u258d\u258c\u258b\u258a\u2589]|\d+%\||it\/s|[KMGT]?B\/s/;
const BLOCKS = /[\u2588\u258f\u258e\u258d\u258c\u258b\u258a\u2589\s]+/g;
const barShape = (line) => line.replace(/[\d.,:%]+/g, '#').replace(BLOCKS, ' ');

// tqdm's multi-bar redraws carry bare cursor-movement escapes, which render as
// stray glyphs in a <div>.
const ANSI = /\u001b\[[0-9;?]*[ -/]*[@-~]/g;
const plain = (line) => line.replace(ANSI, '');

function appendLog(raw) {
  const line = plain(raw);
  const shape = BAR.test(line) ? barShape(line) : null;
  const repeat = shape && shape === els.lastShape && els.log.lastElementChild;
  if (repeat) els.log.lastElementChild.remove();
  els.lastShape = shape;
  els.log.append(line);
}

/* --- list --------------------------------------------------------------- */

function elapsedOf(job) {
  const start = job.started_at || job.created_at;
  if (!start) return '—';
  return duration((job.finished_at || Date.now() / 1000) - start);
}

function jobRow(job) {
  return h('button', {
    class: 'job-row',
    'aria-current': job.id === state.selected ? 'true' : 'false',
    onClick: () => select(job.id),
  },
    h('div', { class: 'row' },
      badge(job.status, job.status),
      h('span', { class: 'tag' }, job.kind),
      h('span', { class: 'spacer' }),
      h('span', { class: 'faint small nowrap' }, ago(job.created_at))),
    h('div', { class: 'j-title truncate', title: job.title }, job.title),
    progressBlock(job),
    h('div', { class: 'j-meta' },
      h('span', null, elapsedOf(job)),
      job.exit_code ? h('span', { class: 'l-err' }, `exit ${job.exit_code}`) : null,
      h('span', { class: 'mono' }, job.id)));
}

function renderList() {
  const rows = visible();
  const box = els.list;
  const scroll = box.scrollTop;
  mount(box, rows.length
    ? rows.map(jobRow)
    : empty(state.kind === 'all' ? 'No jobs yet' : `No ${state.kind} jobs`,
        'Downloads, image builds, fine-tunes and Heretic runs all show up here.'));
  box.scrollTop = scroll;
  // A short page means the API has nothing older left to hand over.
  els.more.disabled = state.jobs.length < state.limit || state.limit >= MAX_LIMIT;
  renderKinds();
  ctx.setBadge('jobs', activeCount());
}

function renderKinds() {
  mount(els.kinds, KINDS.map(([id, label]) => {
    const n = id === 'all' ? state.jobs.length : state.jobs.filter((job) => job.kind === id).length;
    return h('button', {
      class: 'btn-sm',
      'aria-pressed': state.kind === id ? 'true' : 'false',
      onClick: () => { state.kind = id; renderList(); },
    }, label, n ? h('span', { class: 'tag' }, String(n)) : null);
  }));
}

/* --- detail ------------------------------------------------------------- */

const SKIP_FACTS = new Set([
  'percent', 'phase', 'step', 'total', 'total_steps', 'files_done', 'files_total',
]);

function factValue(key, value) {
  if (Array.isArray(value)) return `${value.length}`;
  if (typeof value !== 'number') return String(value);
  if (key === 'speed_bps') return `${bytes(value)}/s`;
  if (key === 'eta' || key === 'elapsed') return duration(value);
  if (key.endsWith('_bytes')) return bytes(value);
  if (Number.isInteger(value)) return String(value);
  return Math.abs(value) < 1e-3 ? value.toExponential(2) : value.toFixed(4);
}

const fact = (label, value) =>
  h('div', { class: 'job-fact' }, h('div', { class: 'k' }, label), h('div', { class: 'v' }, value));

function factsFor(job) {
  const progress = job.progress || {};
  const facts = [
    fact('created', when(job.created_at)),
    fact('elapsed', elapsedOf(job)),
    fact('container', job.container_name || '—'),
  ];
  if (job.exit_code !== null && job.exit_code !== undefined) {
    facts.push(fact('exit code', String(job.exit_code)));
  }
  const steps = stepLabel(progress);
  if (steps) facts.push(fact('progress', steps));
  for (const [key, value] of Object.entries(progress)) {
    if (SKIP_FACTS.has(key) || value === null || value === undefined) continue;
    if (typeof value === 'object' && !Array.isArray(value)) continue;
    facts.push(fact(key.replace(/_/g, ' '), factValue(key, value)));
  }
  return h('div', { class: 'job-facts' }, facts);
}

const isOom = (job) =>
  job.exit_code === 137 || /oom|out of memory/i.test(job.error || '');

function failureNotices(job) {
  if (job.status !== 'failed') return null;
  const parts = [
    notice('danger',
      h('strong', null, 'Job failed'),
      job.exit_code !== null && job.exit_code !== undefined
        ? h('span', { class: 'mono' }, ` (exit ${job.exit_code})`) : null,
      h('div', { class: 'mono small' }, job.error || 'no error was recorded')),
  ];
  if (isOom(job)) {
    parts.push(notice('danger',
      h('strong', null, 'This was a host-memory event, not a bug in the job. '),
      h('span', null,
        'Everything on this box — vLLM weights, KV cache, a training run — comes out of the same '),
      h('span', null,
        '121 GiB of unified memory, so a job dies when the servers already hold too much. '),
      h('span', null, 'Check what is committed before retrying.'),
      h('div', { class: 'row', style: { marginTop: '8px' } },
        h('button', { class: 'btn-sm', onClick: () => ctx.navigate('overview') },
          'Open the Overview memory panel'))));
  }
  return parts;
}

function specDetails(job) {
  const spec = job.spec || {};
  const meta = spec.meta || {};
  const scalars = Object.entries(meta).filter(([, v]) => typeof v !== 'object' || v === null);
  return h('details', { class: 'collapse' },
    h('summary', null, 'Container spec'),
    h('div', { class: 'stack' },
      h('div', { class: 'cmdbox' }, `${spec.image || '?'}\n${(spec.command || []).join(' ')}`),
      (spec.mounts || []).length
        ? h('div', { class: 'cmdbox' }, (spec.mounts || []).join('\n')) : null,
      scalars.length ? h('div', { class: 'job-facts' },
        scalars.map(([key, value]) => fact(key.replace(/_/g, ' '), String(value)))) : null));
}

async function copyLog(job_id, button) {
  button.disabled = true;
  try {
    const text = await getText(`/jobs/${job_id}/log`);
    try {
      await navigator.clipboard.writeText(text);
      const original = button.textContent;
      button.textContent = 'Copied';
      setTimeout(() => { button.textContent = original; }, 1300);
    } catch {
      // The dashboard is usually reached over plain http, where the clipboard
      // API is unavailable; hand the operator something selectable instead.
      modal(`Log · ${job_id}`,
        h('textarea', { readOnly: true, rows: 24, value: text }), { wide: true });
    }
  } catch (error) {
    toast(error.message, { level: 'danger' });
  } finally {
    button.disabled = false;
  }
}

async function cancel(job) {
  const ok = await confirmDialog('Cancel job',
    `Stop "${job.title}"? The container is killed and any partial output is left on disk.`,
    { confirmLabel: 'Cancel job' });
  if (!ok) return;
  try {
    const result = await post(`/jobs/${job.id}/cancel`);
    if (!result.cancelled) toast('The job had already finished.', { level: 'warn' });
  } catch (error) {
    toast(error.message, { level: 'danger' });
  }
}

/** Repaint everything about the selected job except the log, which streams. */
function paintDetail(job) {
  mount(els.detailBadge, badge(job.status, job.status));
  mount(els.detailActions,
    !TERMINAL.has(job.status)
      ? h('button', { class: 'btn-sm btn-danger', onClick: () => cancel(job) }, 'Cancel')
      : null,
    h('button', {
      class: 'btn-sm btn-ghost',
      onClick: (event) => copyLog(job.id, event.currentTarget),
    }, 'Copy full log'));
  mount(els.detailTop,
    failureNotices(job),
    progressBlock(job),
    factsFor(job),
    specDetails(job));
}

function select(job_id) {
  state.selected = job_id;
  const job = state.byId.get(job_id);
  if (!job) return;
  ctx.navigate('jobs', job_id);
  renderList();

  if (closeDetail) closeDetail();
  closeDetail = null;

  const log = logBox();
  els.log = log;
  els.lastShape = null;
  els.live = h('div', { class: 'job-live', hidden: true });
  els.detailBadge = h('span', null);
  els.detailActions = h('div', { class: 'row' });
  els.detailTop = h('div', { class: 'stack' });

  mount(els.detail, panel(job.title, {
    sub: `${job.kind} · ${job.id}`,
    actions: [els.detailBadge, els.detailActions],
    body: h('div', { class: 'stack' }, els.detailTop, els.live, log),
  }));
  paintDetail(job);
  followJob(job_id);
}

function followJob(job_id) {
  const close = stream(`/jobs/${job_id}/stream`, {
    log: (payload) => { if (state.selected === job_id) appendLog(payload.line); },
    // A redraw of one progress bar, not a new line: overwrite in place, or the
    // pane fills with thousands of copies of the same bar.
    'progress-line': (payload) => {
      if (state.selected !== job_id) return;
      els.live.hidden = false;
      els.live.textContent = plain(payload.line);
    },
    status: (payload) => {
      const job = state.byId.get(job_id);
      if (job && state.selected === job_id) paintDetail({ ...job, ...payload });
    },
    // The server closes the connection after `end`; without this the browser
    // reconnects and replays the entire backlog.
    end: () => {
      if (closeDetail === close) { closeDetail(); closeDetail = null; }
      if (state.selected === job_id) els.live.hidden = true;
    },
  });
  closeDetail = close;
}

/* --- view --------------------------------------------------------------- */

export async function render(container, context) {
  ctx = context;
  ensureStyles('jobs', CSS);
  state.kind = 'all';
  state.limit = PAGE;
  state.selected = null;

  els.kinds = h('div', { class: 'jobs-kinds' });
  els.list = h('div', { class: 'jobs-list' });
  els.more = h('button', {
    class: 'btn-sm btn-ghost',
    onClick: async () => {
      state.limit = Math.min(MAX_LIMIT, state.limit + PAGE);
      await refresh();
    },
  }, 'Load older');
  els.detail = h('div', null,
    panel('No job selected', {
      body: empty('Nothing selected', 'Pick a job on the left to follow its log live.'),
    }));

  mount(container,
    h('div', { class: 'page-head' },
      h('div', null,
        h('h1', null, 'Jobs'),
        h('p', null,
          'Downloads, image builds, fine-tunes and Heretic runs. Each one is a detached '
          + 'container, so closing this tab — or restarting the dashboard — does not stop it.')),
      h('div', { class: 'page-actions' },
        els.more,
        h('button', { class: 'btn-sm', onClick: () => refresh() }, 'Refresh'))),
    h('div', { class: 'jobs-layout' },
      panel('History', { sub: 'newest first', actions: els.kinds, body: els.list, flush: true }),
      els.detail));

  await refresh();

  // A deep link can name a job older than the page we just loaded.
  const wanted = ctx.routeDetail();
  if (wanted && !state.byId.has(wanted)) {
    try {
      upsert(await get(`/jobs/${wanted}`));
      renderList();
    } catch { /* a stale link is not worth a toast */ }
  }
  if (wanted) select(wanted);

  closeAll = stream('/jobs/stream/all', {
    job: (payload) => {
      if (!payload?.job) return;
      upsert(payload.job);
      renderList();
      if (payload.job.id === state.selected) paintDetail(payload.job);
    },
  });

  // Relative times and the elapsed counter go stale on their own.
  ticker = setInterval(() => {
    if (activeCount()) renderList();
  }, 5000);
}

async function refresh() {
  try {
    await load();
    renderList();
    if (state.selected && state.byId.has(state.selected)) paintDetail(state.byId.get(state.selected));
  } catch (error) {
    toast(error.message, { level: 'danger' });
  }
}

export function dispose() {
  if (closeAll) closeAll();
  if (closeDetail) closeDetail();
  closeAll = closeDetail = null;
  clearInterval(ticker);
  ticker = null;
}

const CSS = `
.jobs-layout { display: grid; grid-template-columns: minmax(0, 380px) minmax(0, 1fr); gap: var(--gap); align-items: start; }
@media (max-width: 1100px) { .jobs-layout { grid-template-columns: 1fr; } }

.jobs-kinds { display: flex; gap: 4px; flex-wrap: wrap; }
.jobs-kinds button { gap: 5px; }
.jobs-kinds button[aria-pressed="true"] { border-color: var(--accent); color: var(--accent); background: var(--accent-dim); }

.jobs-list { max-height: calc(100vh - 210px); overflow-y: auto; }
.job-row {
  display: flex; flex-direction: column; align-items: stretch; gap: 6px; width: 100%;
  padding: 10px 13px; border: 0; border-bottom: 1px solid var(--border); border-radius: 0;
  background: none; text-align: left;
}
.job-row:hover:not(:disabled) { background: var(--bg-raised); border-color: var(--border); }
.job-row[aria-current="true"] { background: var(--accent-dim); box-shadow: inset 2px 0 0 var(--accent); }
.job-row .j-title { font-family: var(--mono); font-size: 12.5px; }
.job-row .j-meta { display: flex; gap: 10px; font-size: 11px; color: var(--text-faint); font-family: var(--mono); }
.job-row .j-meta .l-err { color: var(--danger); }

.job-progress-label { display: flex; justify-content: space-between; gap: 10px; font-family: var(--mono); font-size: 11px; color: var(--text-dim); margin-bottom: 4px; }
.progress.failed > span { background: var(--danger); }
.progress.stalled > span { background: var(--border-strong); }
.progress.indeterminate > span { width: 32%; animation: job-sweep 1.6s ease-in-out infinite; }
@keyframes job-sweep { 0% { margin-left: 0; } 50% { margin-left: 68%; } 100% { margin-left: 0; } }

.job-facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px 18px; }
.job-fact .k { font-size: 10.5px; text-transform: uppercase; letter-spacing: .06em; color: var(--text-faint); }
.job-fact .v { font-family: var(--mono); font-size: 12.5px; word-break: break-all; }

.job-live {
  font-family: var(--mono); font-size: 11.5px; color: var(--text-dim);
  background: var(--bg-sunken); border: 1px solid var(--border); border-radius: var(--radius-s);
  padding: 6px 10px; white-space: pre-wrap; word-break: break-all;
}
`;
