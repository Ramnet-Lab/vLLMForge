/* The vLLM control surface: what is resident, what it costs in host memory, and
   every flag the image will accept.

   The parameter form is built from GET /api/servers/schema, which the backend
   generates from the image itself, so the form cannot drift from the binary
   that will run. The only hard-coded vLLM flags in this file are in PRESETS,
   and each of those carries the provenance of its numbers in the UI. */

import { ApiError, del, get, getText, patch, post } from '../api.js';
import {
  badge, bytes, confirmDialog, copyButton, count, debounce, empty, ensureStyles, field, h,
  logBox, modal, mount, notice, panel, pct, spinner, stat, toast,
} from '../ui.js';

const STYLES = `
.serve-actions { display: flex; flex-wrap: wrap; gap: 5px; justify-content: flex-end; }
.serve-row { cursor: pointer; }
.serve-row.sel > td { background: var(--accent-dim); }
.serve-row .s-name { font-weight: 600; }
.serve-row .s-model { font-family: var(--mono); font-size: 11.5px; color: var(--text-dim); }
.serve-tabs { display: flex; gap: 4px; margin: -14px -14px 14px; padding: 0 10px;
  border-bottom: 1px solid var(--border); background: var(--bg-raised); }
.serve-tabs button { border: 0; border-bottom: 2px solid transparent; border-radius: 0;
  background: none; color: var(--text-dim); padding: 9px 10px; font-size: 12.5px; }
.serve-tabs button:hover:not(:disabled) { background: none; color: var(--text); }
.serve-tabs button[aria-current="true"] { color: var(--text);
  border-bottom-color: var(--accent); }
.serve-presets { display: grid; gap: 10px;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); }
.serve-preset { display: block; text-align: left; white-space: normal;
  line-height: 1.45; padding: 10px 12px; }
.serve-preset b { display: block; font-size: 12.5px; margin-bottom: 2px; }
.serve-preset span { display: block; font-size: 11.5px; color: var(--text-faint); }
.serve-def { font-family: var(--mono); font-size: 11px; color: var(--text-faint); }
.serve-form .field .help { display: -webkit-box; -webkit-line-clamp: 3;
  -webkit-box-orient: vertical; overflow: hidden; }
.serve-form .field:hover .help, .serve-form .field:focus-within .help { -webkit-line-clamp: unset; }
.serve-foot { position: sticky; bottom: 0; z-index: 6; display: flex; flex-wrap: wrap;
  align-items: center; gap: 10px; padding: 12px 14px; background: var(--bg-raised);
  border: 1px solid var(--border-strong); border-radius: var(--radius); box-shadow: var(--shadow); }
.serve-foot .notice { margin: 0; flex: 1 1 340px; }
.serve-metrics { display: grid; gap: 14px;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); }
.serve-unmanaged td:first-child { border-left: 3px solid var(--info); }
.serve-tenants { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
.serve-pick { display: flex; flex-direction: column; gap: 6px; }
.serve-pick-row { display: flex; align-items: center; gap: 6px; }
.serve-pick-row select { flex: 1 1 auto; min-width: 0; }
`;

/* Two of these three are configurations that have actually run co-resident on
   this box; the third is derived from them. The UI repeats that distinction,
   because an operator who trusts a preset that was never measured pays for it
   in a frozen machine. */
const PRESETS = [
  {
    title: '27B NVFP4 chat · util 0.52',
    note: 'Measured: the vllm-qwen container on this host runs these values beside the embedder '
      + 'below, for a combined 0.68. max-num-seqs 8 is the conservative end of its range.',
    args: {
      gpu_memory_utilization: 0.52,
      max_model_len: 262144,
      kv_cache_dtype: 'fp8',
      max_num_seqs: 8,
      max_num_batched_tokens: 4096,
      enable_prefix_caching: true,
      enable_chunked_prefill: true,
    },
  },
  {
    title: 'Embedding server · util 0.16',
    note: 'Measured: the vllm-embed container on this host. A pooling runner serves '
      + '/v1/embeddings and no chat endpoint at all.',
    args: {
      gpu_memory_utilization: 0.16,
      runner: 'pooling',
      convert: 'embed',
      max_model_len: 8192,
      max_num_batched_tokens: 8192,
    },
  },
  {
    title: 'Small model on a busy box · util 0.10',
    note: 'Derived, not measured: room for a model under ~4B while both servers above stay up. '
      + 'Start here when you only want to see whether a model loads.',
    args: {
      gpu_memory_utilization: 0.10,
      max_model_len: 8192,
      max_num_seqs: 8,
    },
  },
];

const DETAIL_TABS = [['command', 'Command'], ['logs', 'Logs'], ['metrics', 'Metrics']];
const LIVE = ['running', 'loading', 'starting', 'unhealthy'];

const state = {
  ctx: null,
  main: null,
  schema: null,
  flagIndex: null,
  status: { servers: [], foreign: [], budget: null },
  mode: 'list',
  selected: null,
  detailTab: 'command',
  detailKey: '',
  verdict: null,
  editor: null,
  nodes: {},
  log: { lines: [], box: null },
  timers: [],
  stopped: false,
};

/* --- entry points -------------------------------------------------------- */

export async function render(container, ctx) {
  ensureStyles('serve', STYLES);
  state.ctx = ctx;
  state.stopped = false;
  state.main = h('div', { class: 'stack' });

  mount(container,
    h('div', { class: 'page-head' },
      h('div', null,
        h('h1', null, 'Serve'),
        h('p', null,
          'vLLM engines on this host. Utilisation is a fraction of the memory the CPU and GPU '
          + 'share, so every engine that is up spends the same pool.')),
      h('div', { class: 'page-actions' },
        h('button', { onClick: () => refreshStatus() }, 'Refresh'),
        h('button', { class: 'btn-primary', onClick: () => openEditor(null) }, 'New server'))),
    state.main);

  const [schema] = await Promise.all([get('/servers/schema'), refreshStatus({ render: false })]);
  if (state.stopped) return;
  state.schema = schema;
  state.flagIndex = new Map();
  for (const section of [...schema.featured, ...schema.advanced]) {
    for (const arg of section.flags) state.flagIndex.set(arg.dest, arg);
  }

  // Two tabs link here: Overview sends a server id, Models sends a repo id.
  const detail = routeDetail(ctx);
  if (/^\d+$/.test(detail)) {
    renderList();
    selectServer(Number(detail));
  } else if (detail === 'new') {
    openEditor(null);
  } else if (detail) {
    openEditor(null, { model: detail });
  } else {
    renderList();
  }

  state.timers.push(setInterval(() => { if (state.mode === 'list') refreshStatus(); }, 5000));
  state.timers.push(setInterval(tickDetail, 3000));
}

export function dispose() {
  state.stopped = true;
  for (const timer of state.timers) clearInterval(timer);
  state.timers = [];
  state.editor = null;
  state.selected = null;
  state.detailKey = '';
  state.nodes = {};
}

function routeDetail(ctx) {
  const raw = ctx.routeDetail();
  if (!raw) return '';
  try {
    return decodeURIComponent(raw);
  } catch {
    return raw;
  }
}

/* --- status -------------------------------------------------------------- */

async function refreshStatus({ render: doRender = true } = {}) {
  try {
    state.status = await get('/servers');
  } catch (error) {
    toast(error.message, { level: 'danger', title: 'Could not read server status' });
    return;
  }
  if (state.stopped) return;
  state.ctx.setBadge('serve', state.status.servers.filter((s) => s.status === 'running').length);
  if (doRender && state.mode === 'list') renderList();
}

const selectedServer = () => state.status.servers.find((s) => s.id === state.selected) || null;

/* --- list mode ----------------------------------------------------------- */

function ensureListShell() {
  const nodes = state.nodes;
  if (nodes.budget?.isConnected) return;
  nodes.budget = h('div');
  nodes.servers = h('div');
  nodes.foreign = h('div');
  nodes.foreignPanel = panel('Containers this dashboard did not start', {
    sub: 'read-only',
    flush: true,
    body: nodes.foreign,
  });
  nodes.detail = h('div');
  mount(state.main,
    nodes.budget,
    panel('Servers', { flush: true, body: nodes.servers }),
    nodes.foreignPanel,
    nodes.detail);
}

function renderList() {
  state.mode = 'list';
  if (state.selected !== null && !selectedServer()) state.selected = null;
  ensureListShell();
  mount(state.nodes.budget, budgetPanel());
  renderServerTable();
  renderForeignTable();
  renderDetail();
}

function budgetPanel() {
  const budget = state.status.budget;
  if (!budget) return null;
  const level = budget.free_util <= 0 ? 'crit' : budget.free_util < 0.1 ? 'warn' : '';
  const reserveWidth = Math.max(0,
    Math.min(1 - budget.occupied_util, budget.reserve_bytes / budget.total_bytes));
  return panel('Host memory budget', {
    sub: `${budget.tenants.length} engine(s) resident`,
    body: h('div', null,
      h('div', { class: 'grid cols-4' },
        stat('Committed', pct(budget.committed_util, 0),
          `${bytes(budget.committed_bytes)} claimed by vLLM flags`),
        stat('Measured', bytes(budget.measured_gpu_bytes), 'actually held by GPU processes'),
        stat('Free to commit', pct(budget.free_util, 0),
          `ceiling ${pct(budget.max_util, 0)} · ${bytes(budget.reserve_bytes)} kept for the OS`),
        stat('Available now', bytes(budget.available_bytes),
          `${bytes(budget.free_bytes)} genuinely free`)),
      h('div', { class: `bar ${level}`, style: { marginTop: '12px' } },
        h('span', { class: 'seg-used', style: { width: pct(Math.min(1, budget.occupied_util)) } }),
        h('span', { class: 'seg-reserve', style: { width: pct(reserveWidth) } })),
      h('div', { class: 'legend' },
        h('span', null, h('i', { style: { background: 'var(--accent)' } }), 'occupied'),
        h('span', null, h('i', { style: { background: 'var(--border-strong)' } }), 'reserve'),
        h('span', null, 'the rest is available to a new engine')),
      h('div', { class: 'serve-tenants' },
        budget.tenants.map((tenant) => h('span', { class: 'tag', title: tenant.note || '' },
          `${tenant.name} ${tenant.util}${tenant.implicit ? ' (implied)' : ''}`)),
        budget.tenants.length ? null : h('span', { class: 'faint small' }, 'nothing resident'))),
  });
}

function renderServerTable() {
  const rows = state.status.servers;
  if (!rows.length) {
    mount(state.nodes.servers, empty(
      'No servers defined',
      'A server here is a saved parameter set plus a container. Create one, or use a preset.',
      h('button', { class: 'btn-primary', onClick: () => openEditor(null) }, 'New server')));
    return;
  }
  mount(state.nodes.servers, h('div', { class: 'table-wrap' },
    h('table', null,
      h('thead', null, h('tr', null,
        h('th', null, 'Name'),
        h('th', { class: 'num' }, 'Port'),
        h('th', { class: 'num' }, 'Util'),
        h('th', null, 'Status'),
        h('th', null, 'Serving'),
        h('th', null, ''))),
      h('tbody', null, rows.map(serverRow)))));
}

function utilCell(util) {
  return util === null || util === undefined
    ? h('span', {
      class: 'faint',
      title: 'no --gpu-memory-utilization set, so vLLM applies its own default',
    }, 'default')
    : String(util);
}

function servingCell(entry) {
  const models = entry.health?.models || [];
  if (models.length) {
    return models.map((name) => h('span', { class: 'tag', style: { marginRight: '4px' } }, name));
  }
  // A starting engine binds no port at all, so an absent /v1/models is load
  // time rather than a fault.
  const label = entry.status === 'loading' || entry.status === 'starting' ? 'loading' : '—';
  return h('span', { class: 'faint small' }, label);
}

function serverRow(server) {
  const act = (label, fn, cls) => h('button', {
    class: `btn-sm ${cls || 'btn-ghost'}`,
    onClick: (event) => { event.stopPropagation(); fn(); },
  }, label);
  const live = LIVE.includes(server.status);

  return h('tr', {
    class: `serve-row${server.id === state.selected ? ' sel' : ''}`,
    onClick: () => selectServer(server.id),
  },
  h('td', null,
    h('div', { class: 's-name' }, server.name),
    h('div', { class: 's-model truncate', title: server.model }, server.model)),
  h('td', { class: 'num' }, String(server.port)),
  h('td', { class: 'num' }, utilCell(server.util)),
  h('td', null,
    badge(server.status),
    server.oom_killed ? h('div', { class: 'faint small' }, 'OOM-killed') : null,
    !server.oom_killed && server.exit_code
      ? h('div', { class: 'faint small' }, `exit ${server.exit_code}`)
      : null),
  h('td', null, servingCell(server)),
  h('td', null, h('div', { class: 'serve-actions' },
    live
      ? act('Stop', () => stopServer(server.id))
      : act('Start', () => startServer(server.id), 'btn-primary'),
    live ? act('Restart', () => restartServer(server.id)) : null,
    act('Edit', () => openEditor(server)),
    act('Logs', () => selectServer(server.id, 'logs')),
    act('Delete', () => deleteServer(server), 'btn-danger'))));
}

function renderForeignTable() {
  const rows = state.status.foreign;
  state.nodes.foreignPanel.hidden = rows.length === 0;
  if (!rows.length) return;
  mount(state.nodes.foreign,
    h('div', { style: { padding: '12px 14px 0' } },
      notice('info',
        h('strong', null, 'Not managed here. '),
        'These were launched outside the dashboard — by hand or by a script. Their utilisation '
        + 'still counts against the budget above, so stopping one is the quickest way to make '
        + 'room. Nothing else about them can be changed from here.')),
    h('div', { class: 'table-wrap' },
      h('table', null,
        h('thead', null, h('tr', null,
          h('th', null, 'Container'),
          h('th', { class: 'num' }, 'Port'),
          h('th', { class: 'num' }, 'Util'),
          h('th', null, 'Status'),
          h('th', null, 'Serving'),
          h('th', null, ''))),
        h('tbody', null, rows.map(foreignRow)))));
}

function foreignRow(item) {
  return h('tr', { class: 'serve-unmanaged' },
    h('td', null,
      h('div', { class: 's-name' }, item.name),
      h('div', { class: 's-model truncate', title: item.model }, item.model || item.image)),
    h('td', { class: 'num' }, item.port ? String(item.port) : '—'),
    h('td', { class: 'num' }, utilCell(item.util)),
    h('td', null, badge(item.status)),
    h('td', null, servingCell(item)),
    h('td', null, h('div', { class: 'serve-actions' },
      h('button', { class: 'btn-sm', onClick: () => showForeignCommand(item) }, 'Command'),
      h('button', { class: 'btn-sm btn-danger', onClick: () => stopForeign(item) }, 'Stop'))));
}

function showForeignCommand(item) {
  const text = (item.command || []).join(' ');
  modal(item.name,
    h('div', null,
      h('p', { class: 'muted small' },
        'Recorded container command. Copy it into a new server definition if you want the '
        + 'dashboard to own this configuration.'),
      h('div', { class: 'cmdbox' }, text || 'no recorded command')),
    { actions: [copyButton(text, 'Copy')], wide: true });
}

/* --- detail -------------------------------------------------------------- */

function selectServer(id, tab) {
  state.selected = id;
  state.detailTab = tab || state.detailTab || 'command';
  state.verdict = null;
  state.log = { lines: [], box: null };
  renderServerTable();
  renderDetail();
  loadDetailBody();
  refreshVerdict();
}

async function refreshVerdict() {
  const server = selectedServer();
  if (!server) return;
  const util = server.util;
  const query = util === null || util === undefined ? '' : `?util=${util}`;
  try {
    const verdict = await get(`/system/budget/check${query}`);
    if (state.selected !== server.id) return;
    state.verdict = verdict;
    renderDetail();
  } catch (error) {
    console.error('budget check failed', error);
  }
}

/** Rebuilds the detail panel only when something it shows actually changed —
 *  re-parenting the log box on every poll would throw away its scroll. */
function renderDetail() {
  const server = selectedServer();
  if (!server) {
    state.detailKey = '';
    return mount(state.nodes.detail);
  }
  const key = [server.id, server.status, state.detailTab, state.verdict?.level ?? ''].join('|');
  if (key === state.detailKey) return undefined;
  state.detailKey = key;

  const live = LIVE.includes(server.status);
  const blocked = state.verdict?.level === 'block';
  state.nodes.detailBody = state.nodes.detailBody || h('div');

  mount(state.nodes.detail, panel(server.name, {
    sub: `${server.model} · ${server.url}`,
    actions: [
      badge(server.status),
      live
        ? h('button', { class: 'btn-sm', onClick: () => stopServer(server.id) }, 'Stop')
        : h('button', {
          class: 'btn-sm btn-primary',
          disabled: blocked,
          title: blocked ? state.verdict.message : '',
          onClick: () => startServer(server.id),
        }, 'Start'),
      !live && blocked
        ? h('button', { class: 'btn-sm btn-danger', onClick: () => startAnyway(server.id) },
          'Start anyway')
        : null,
      h('button', { class: 'btn-sm', onClick: () => openEditor(server) }, 'Edit'),
    ],
    body: h('div', null,
      h('div', { class: 'serve-tabs' }, DETAIL_TABS.map(([id, label]) => h('button', {
        'aria-current': String(state.detailTab === id),
        onClick: () => {
          if (state.detailTab === id) return;
          state.detailTab = id;
          state.log = { lines: [], box: null };
          renderDetail();
          loadDetailBody();
        },
      }, label))),
      state.verdict ? verdictNotice(state.verdict) : null,
      state.nodes.detailBody),
  }));
  return undefined;
}

function verdictNotice(verdict) {
  const level = { ok: 'ok', warn: 'warn', block: 'danger' }[verdict.level] || 'info';
  const lead = { ok: 'Fits. ', warn: 'Tight. ', block: 'Blocked. ' }[verdict.level] || '';
  return notice(level,
    h('strong', null, lead),
    verdict.message,
    verdict.suggested_util
      ? h('div', { class: 'faint', style: { marginTop: '4px' } },
        `Largest --gpu-memory-utilization that fits right now: ${verdict.suggested_util}`)
      : null);
}

function loadDetailBody() {
  const server = selectedServer();
  const host = state.nodes.detailBody;
  if (!server || !host) return;
  if (state.detailTab === 'command') loadCommand(server, host);
  else if (state.detailTab === 'logs') loadLogs(server, host);
  else loadMetrics(server, host);
}

function tickDetail() {
  if (state.mode !== 'list' || !state.nodes.detailBody) return;
  const server = selectedServer();
  if (!server) return;
  if (state.detailTab === 'logs') loadLogs(server, state.nodes.detailBody);
  else if (state.detailTab === 'metrics') loadMetrics(server, state.nodes.detailBody);
}

async function loadCommand(server, host) {
  mount(host, h('div', { class: 'row' }, spinner()));
  let preview;
  try {
    preview = await get(`/servers/${server.id}/preview`);
  } catch (error) {
    mount(host, notice('danger', error.message));
    return;
  }
  if (state.selected !== server.id || state.detailTab !== 'command') return;
  const argv = preview.argv.join(' ');
  mount(host,
    h('p', { class: 'muted small', style: { marginTop: '0' } },
      'Exactly what Start runs. Nothing is hidden: paste it into a shell and you get the same '
      + 'container.'),
    commandBlock('docker run', preview.command),
    commandBlock('vllm serve', argv));
}

function commandBlock(title, text) {
  return h('div', { style: { marginBottom: '14px' } },
    h('div', { class: 'row', style: { marginBottom: '6px' } },
      h('strong', { class: 'small' }, title),
      h('span', { class: 'spacer' }),
      copyButton(text, 'Copy')),
    h('div', { class: 'cmdbox' }, text));
}

async function loadLogs(server, host) {
  let text;
  try {
    text = await getText(`/servers/${server.id}/logs?tail=400`);
  } catch (error) {
    mount(host, notice('danger', error.message));
    return;
  }
  if (state.selected !== server.id || state.detailTab !== 'logs') return;

  const lines = text.split('\n');
  const box = state.log.box;
  const appended = box && box.isConnected && lines.length >= state.log.lines.length
    && state.log.lines.every((line, index) => line === lines[index]);

  if (appended) {
    for (const line of lines.slice(state.log.lines.length)) box.append(line);
    state.log.lines = lines;
    return;
  }
  state.log = { lines, box: logBox(lines) };
  mount(host,
    h('div', { class: 'row', style: { marginBottom: '8px' } },
      h('span', { class: 'faint small' }, 'last 400 lines · refreshed every 3s'),
      h('span', { class: 'spacer' }),
      copyButton(() => state.log.lines.join('\n'), 'Copy all')),
    state.log.box);
}

async function loadMetrics(server, host) {
  if (!LIVE.includes(server.status)) {
    mount(host, empty('Not running', 'Start the server to collect metrics.'));
    return;
  }
  let data;
  try {
    data = await get(`/servers/${server.id}/metrics`);
  } catch (error) {
    mount(host, notice('danger', error.message));
    return;
  }
  if (state.selected !== server.id || state.detailTab !== 'metrics') return;

  const selected = data.selected || {};
  if (!Object.keys(selected).length) {
    mount(host, notice('warn', server.status === 'running'
      ? 'The server is up but published no vLLM metrics. --disable-log-stats turns them off.'
      : 'Loading. vLLM binds no port until the weights are in and CUDA graphs are captured, so '
        + 'there is nothing to scrape yet — this is not an unreachable server.'));
    return;
  }

  const value = (name) => selected[`vllm:${name}`];
  const queries = value('prefix_cache_queries_total');
  const hits = value('prefix_cache_hits_total');
  // 0.24 dropped gpu_prefix_cache_hit_rate on this build; the counters are the
  // ones that are actually there, so derive the rate when the gauge is absent.
  const rate = value('gpu_prefix_cache_hit_rate') ?? (queries ? hits / queries : null);
  const kv = value('kv_cache_usage_perc') ?? value('gpu_cache_usage_perc');

  mount(host, h('div', { class: 'serve-metrics' },
    stat('Running', count(value('num_requests_running') ?? 0), 'in the current batch'),
    stat('Waiting', count(value('num_requests_waiting') ?? 0), 'queued behind max-num-seqs'),
    stat('KV cache', kv === undefined || kv === null ? '—' : pct(kv, 1), 'of allocated blocks'),
    stat('Prefix hits', rate === null || rate === undefined ? '—' : pct(rate, 1),
      queries ? `${count(hits)} of ${count(queries)} queries` : 'no queries yet'),
    stat('Prompt tokens', count(value('prompt_tokens_total') ?? 0), 'cumulative'),
    stat('Generated', count(value('generation_tokens_total') ?? 0), 'cumulative'),
    stat('Completed', count(value('request_success_total') ?? 0), 'successful requests'),
    stat('Preemptions', count(value('num_preemptions_total') ?? 0), 'KV cache pressure')));
}

/* --- lifecycle ----------------------------------------------------------- */

async function startServer(id, { force = false } = {}) {
  try {
    const result = await post(`/servers/${id}/start${force ? '?force=true' : ''}`);
    toast(result.safety?.message || 'Starting.', {
      level: result.safety?.level === 'warn' ? 'warn' : 'ok',
      title: 'Starting',
    });
  } catch (error) {
    if (error instanceof ApiError && error.status === 409 && error.detail?.message) {
      const go = await confirmDialog('The memory guard refused this launch', error.detail.message,
        { confirmLabel: 'Start anyway' });
      if (go) await startServer(id, { force: true });
      return;
    }
    toast(error.message, { level: 'danger', title: 'Start failed' });
    return;
  }
  await refreshStatus();
  refreshVerdict();
}

async function startAnyway(id) {
  const go = await confirmDialog('Start past the memory guard?',
    state.verdict?.message || 'This launch does not fit the budget.',
    { confirmLabel: 'Start anyway' });
  if (go) await startServer(id, { force: true });
}

async function stopServer(id) {
  try {
    await post(`/servers/${id}/stop`);
  } catch (error) {
    toast(error.message, { level: 'danger', title: 'Stop failed' });
    return;
  }
  await refreshStatus();
  refreshVerdict();
}

async function restartServer(id) {
  try {
    const result = await post(`/servers/${id}/restart`);
    if (result.started === false) {
      const go = await confirmDialog('Stopped, but the relaunch does not fit',
        result.safety?.message || 'The memory guard refused the relaunch.',
        { confirmLabel: 'Start anyway' });
      if (go) await startServer(id, { force: true });
    }
  } catch (error) {
    toast(error.message, { level: 'danger', title: 'Restart failed' });
  }
  await refreshStatus();
  refreshVerdict();
}

async function deleteServer(server) {
  const go = await confirmDialog('Delete this server?',
    `"${server.name}" and its container will be removed. The model files are untouched.`,
    { confirmLabel: 'Delete' });
  if (!go) return;
  try {
    await del(`/servers/${server.id}`);
  } catch (error) {
    toast(error.message, { level: 'danger', title: 'Delete failed' });
    return;
  }
  if (state.selected === server.id) state.selected = null;
  await refreshStatus();
}

async function stopForeign(item) {
  const go = await confirmDialog('Stop an unmanaged container?',
    `"${item.name}" was not started by the dashboard and will not be brought back by it. `
    + `Stopping it releases ${item.util ? `${item.util} of` : 'its share of'} the budget.`,
    { confirmLabel: 'Stop' });
  if (!go) return;
  try {
    await post(`/servers/foreign/${encodeURIComponent(item.name)}/stop`);
  } catch (error) {
    toast(error.message, { level: 'danger', title: 'Stop failed' });
    return;
  }
  await refreshStatus();
}

/* --- disk pickers -------------------------------------------------------- */

/* The backend tags every flag whose value is something on disk with a
   `path_kind`, and GET /servers/paths lists what is actually there for each
   kind. A mistyped path is only discovered minutes later, when the engine has
   already pulled the weights in and then died, so the form offers the real
   thing instead of a text box wherever it can. */

// Sentinel for the escape option, matching the Heretic picker. Not a null byte:
// browsers may replace one in an attribute value with U+FFFD, and the equality
// check would then never fire.
const OTHER = '__llmd_other__';

const EMPTY_PATHS = { options: {}, counts: {}, cache_ok: true };

const pathOptions = (kind) => state.editor?.paths.options[kind] || [];

/** One scan feeds every picker in the form — sixteen path flags must not mean
 *  sixteen requests. */
async function loadPaths() {
  let payload;
  try {
    payload = await get('/servers/paths');
  } catch (error) {
    // A failed scan costs the operator nothing but the list: an empty kind
    // falls through to its text box, so the form stays usable.
    toast(error.message, { level: 'warn', title: 'Could not list what is on disk' });
    return;
  }
  if (!state.editor) return;
  state.editor.paths = payload;
  if (!payload.cache_ok) {
    toast('The Hub cache could not be read, so only local paths are listed.', { level: 'warn' });
  }
}

/** The list goes stale the moment a download finishes or a fine-tune exports,
 *  and the form outlives both. */
async function refreshPaths(button) {
  button.disabled = true;
  await loadPaths();
  button.disabled = false;
  if (!state.editor) return;
  for (const sync of state.editor.pathViews) sync();
  const counts = state.editor.paths.counts;
  toast(`${counts.model || 0} model(s), ${counts.adapter || 0} adapter(s), `
    + `${counts.template || 0} template(s) on disk.`);
}

function optionNodes(kind, { lead, current = '', custom = false }) {
  const options = pathOptions(kind);
  return [
    h('option', { value: '', selected: !custom && !current }, lead),
    options.map((option) => h('option', {
      value: option.value,
      selected: !custom && option.value === current,
      title: option.note || '',
    }, option.detail ? `${option.label} — ${option.detail}` : option.label)),
    // plugin and cert are empty on a box where nobody has written one. Saying so
    // beats a dropdown that opens onto nothing.
    options.length
      ? null
      : h('option', { value: OTHER, disabled: true }, 'nothing on this box'),
    h('option', { value: OTHER, selected: custom }, 'Other — type it'),
  ];
}

/** Returns [control element, setter], like every other widget here. */
function diskPicker(kind, { value = '', lead, placeholder = '', extra = null, onChange }) {
  let current = value === undefined || value === null ? '' : String(value);
  let custom = false;

  const text = h('input', {
    type: 'text',
    placeholder,
    onInput: (event) => {
      current = event.target.value.trim();
      onChange(current);
    },
  });

  const select = h('select', {
    onChange: (event) => {
      const picked = event.target.value;
      custom = picked === OTHER;
      current = custom ? text.value.trim() : picked;
      onChange(current);
      sync();
      if (custom) text.focus();
    },
  });

  function sync() {
    // A value the scan did not turn up — a Hub id nobody has pulled, a path that
    // a refresh has since removed — stays the value: it selects the escape and
    // prefills the box rather than silently becoming whatever is first in the
    // list. With nothing on disk at all there is nothing else it could be.
    const options = pathOptions(kind);
    const known = options.some((option) => option.value === current);
    custom = custom || !options.length || (Boolean(current) && !known);
    mount(select, optionNodes(kind, { lead, current, custom }));
    text.hidden = !custom;
    // Only the escape owns the box; a pick from the list leaves it empty, so
    // reaching for "Other" afterwards starts a fresh entry rather than
    // resurrecting the last thing typed.
    const typed = custom ? current : '';
    if (text.value !== typed) text.value = typed;
  }

  sync();
  state.editor.pathViews.push(sync);

  return [
    h('div', { class: 'serve-pick' },
      h('div', { class: 'serve-pick-row' }, select, extra),
      text),
    (next) => {
      current = next === undefined || next === null ? '' : String(next);
      custom = false;
      sync();
    },
  ];
}

/** A list flag takes several values — --lora-modules preloads more than one
 *  adapter — so a pick adds to what the box already holds instead of replacing
 *  it. The box stays the value the form stores, exactly as it was before. */
function pathListControl(arg, asText, put) {
  const box = h('input', {
    type: 'text',
    placeholder: `${formatDefault(arg)} — space or comma separated`,
    value: asText(state.editor.args[arg.dest]),
    onInput: (event) => {
      const text = event.target.value.trim();
      put(text === '' ? undefined : text);
    },
  });

  const select = h('select', {
    onChange: (event) => {
      const picked = event.target.value;
      // The box is the value; the select is only a way of filling it, so it
      // snaps back and stays ready for the next adapter.
      event.target.selectedIndex = 0;
      box.focus();
      if (picked === '' || picked === OTHER) return;
      const items = box.value.replace(/,/g, ' ').split(/\s+/).filter(Boolean);
      if (items.includes(picked)) return;
      items.push(picked);
      box.value = items.join(' ');
      put(box.value);
    },
  });

  const sync = () => mount(select, optionNodes(arg.path_kind, { lead: 'Add one from disk…' }));
  sync();
  state.editor.pathViews.push(sync);

  return [
    h('div', { class: 'serve-pick' }, select, box),
    (v) => { box.value = asText(v); },
  ];
}

/* --- parameter widgets --------------------------------------------------- */

/** Mirrors vllm_spec._is_default: these values are never rendered into argv. */
function isDefaultValue(arg, value) {
  if (value === undefined || value === null) return true;
  if (typeof value === 'string' && value.trim() === '') return true;
  return value === arg.default;
}

function formatDefault(arg) {
  if (arg.default === null || arg.default === undefined) return 'unset';
  if (typeof arg.default === 'object') return JSON.stringify(arg.default);
  return String(arg.default);
}

function setArg(arg, value) {
  const editor = state.editor;
  if (value === undefined) delete editor.args[arg.dest];
  else editor.args[arg.dest] = value;
  const el = editor.fields.get(arg.dest);
  if (el) el.classList.toggle('changed', !isDefaultValue(arg, editor.args[arg.dest]));
  if (arg.dest === 'gpu_memory_utilization') scheduleSafety();
}

/** Returns [control element, setter] — the setter is what presets drive. */
function controlFor(arg) {
  const value = state.editor.args[arg.dest];
  const put = (v) => setArg(arg, v);

  if (arg.path_kind && arg.widget !== 'list') {
    return diskPicker(arg.path_kind, {
      value,
      lead: `default (${formatDefault(arg)})`,
      placeholder: formatDefault(arg),
      onChange: (next) => put(next === '' ? undefined : next),
    });
  }

  if (arg.widget === 'bool' && arg.default === null) {
    // Tri-state in vLLM: unset lets the engine decide, which is not the same as
    // off — only --no-<flag> forces it off.
    const select = h('select', {
      onChange: (e) => put(e.target.value === '' ? undefined : e.target.value === 'true'),
    },
    h('option', { value: '' }, 'default (unset)'),
    h('option', { value: 'true' }, 'on'),
    h('option', { value: 'false' }, 'off'));
    select.value = value === undefined ? '' : String(value);
    return [select, (v) => { select.value = v === undefined ? '' : String(v); }];
  }

  if (arg.widget === 'bool') {
    const box = h('input', {
      type: 'checkbox',
      checked: value === undefined ? Boolean(arg.default) : Boolean(value),
      onChange: (e) => put(e.target.checked),
    });
    return [box, (v) => { box.checked = Boolean(v); }];
  }

  if (arg.widget === 'enum') {
    const select = h('select', {
      onChange: (e) => put(e.target.value === '' ? undefined : e.target.value),
    },
    h('option', { value: '' }, `default (${formatDefault(arg)})`),
    (arg.choices || []).map((choice) => h('option', { value: choice }, choice)));
    select.value = value === undefined ? '' : String(value);
    return [select, (v) => { select.value = v === undefined ? '' : String(v); }];
  }

  if (arg.widget === 'int' || arg.widget === 'float') {
    const box = h('input', {
      type: 'number',
      step: arg.widget === 'float' ? 'any' : '1',
      placeholder: formatDefault(arg),
      value: value === undefined ? '' : String(value),
      onInput: (e) => put(e.target.value === '' ? undefined : Number(e.target.value)),
    });
    if (arg.dest !== 'gpu_memory_utilization') {
      return [box, (v) => { box.value = v === undefined ? '' : String(v); }];
    }
    const fallback = String(arg.default ?? 0.9);
    const slider = h('input', {
      type: 'range',
      min: '0.05',
      max: '0.95',
      step: '0.01',
      value: value === undefined ? fallback : String(value),
      onInput: (e) => { box.value = e.target.value; put(Number(e.target.value)); },
    });
    box.addEventListener('input', () => { if (box.value !== '') slider.value = box.value; });
    return [
      h('div', { class: 'field-row' }, slider, box),
      (v) => {
        box.value = v === undefined ? '' : String(v);
        slider.value = v === undefined ? fallback : String(v);
      },
    ];
  }

  if (arg.widget === 'list') {
    const asText = (v) => (Array.isArray(v) ? v.join(' ') : (v === undefined ? '' : String(v)));
    if (arg.path_kind) return pathListControl(arg, asText, put);
    const box = h('input', {
      type: 'text',
      placeholder: `${formatDefault(arg)} — space or comma separated`,
      value: asText(value),
      onInput: (e) => put(e.target.value.trim() === '' ? undefined : e.target.value.trim()),
    });
    return [box, (v) => { box.value = asText(v); }];
  }

  // A str-typed flag whose default is a dict is really a JSON blob (hf-overrides
  // and friends), so give it room to type in.
  if (arg.widget === 'json' || (arg.default !== null && typeof arg.default === 'object')) {
    const asText = (v) => {
      if (v === undefined) return '';
      return typeof v === 'string' ? v : JSON.stringify(v);
    };
    const area = h('textarea', {
      rows: 3,
      placeholder: formatDefault(arg),
      value: asText(value),
      onInput: (e) => put(e.target.value.trim() === '' ? undefined : e.target.value.trim()),
    });
    return [area, (v) => { area.value = asText(v); }];
  }

  const box = h('input', {
    type: 'text',
    placeholder: formatDefault(arg),
    value: value === undefined ? '' : String(value),
    onInput: (e) => put(e.target.value === '' ? undefined : e.target.value),
  });
  return [box, (v) => { box.value = v === undefined ? '' : String(v); }];
}

function paramField(arg) {
  const [control, setter] = controlFor(arg);
  const el = field(arg.dest.replaceAll('_', ' '), control, {
    flag: arg.flag,
    help: arg.help || '',
    inline: arg.widget === 'bool' && arg.default !== null,
    changed: !isDefaultValue(arg, state.editor.args[arg.dest]),
  });
  el.append(h('span', { class: 'help serve-def' }, `default: ${formatDefault(arg)}`));
  el.dataset.search = `${arg.dest} ${arg.flag} ${arg.help || ''}`.toLowerCase();

  state.editor.fields.set(arg.dest, el);
  state.editor.setters.set(arg.dest, setter);
  state.editor.searchable.push(el);
  return el;
}

function paramSection(section, { collapsed = false } = {}) {
  const grid = h('div', { class: 'param-grid' }, section.flags.map(paramField));
  const el = collapsed
    ? h('details', { class: 'collapse' },
      h('summary', null, section.title, h('span', { class: 'faint' }, ` ${section.flags.length}`)),
      section.blurb ? h('p', { class: 'blurb' }, section.blurb) : null,
      grid)
    : h('div', { class: 'param-section' },
      h('h3', null, section.title),
      section.blurb ? h('p', { class: 'blurb' }, section.blurb) : null,
      grid);
  state.editor.sections.push({
    el,
    fields: section.flags.map((arg) => state.editor.fields.get(arg.dest)),
  });
  return el;
}

function applyFilter(query) {
  const needle = query.trim().toLowerCase();
  for (const el of state.editor.searchable) {
    el.hidden = Boolean(needle) && !el.dataset.search.includes(needle);
  }
  for (const section of state.editor.sections) {
    const visible = section.fields.some((el) => el && !el.hidden);
    section.el.hidden = !visible;
    if (needle && visible && section.el.tagName === 'DETAILS') section.el.open = true;
  }
}

/* --- editor -------------------------------------------------------------- */

async function openEditor(server, prefill = {}) {
  state.mode = 'edit';
  state.editor = {
    id: server?.id ?? null,
    args: { ...(server?.args || {}) },
    verdict: null,
    fields: new Map(),
    setters: new Map(),
    sections: [],
    searchable: [],
    paths: EMPTY_PATHS,
    pathViews: [],
    form: {
      name: server?.name || '',
      model: server?.model || prefill.model || '',
      port: server?.port || '',
      image: server?.image || '',
      served_name: server?.served_name || '',
      notes: server?.notes || '',
      autostart: Boolean(server?.autostart),
      env: Object.entries(server?.env || {}).map(([k, v]) => `${k}=${v}`).join('\n'),
    },
  };

  await Promise.all([loadPaths(), server ? null : suggest()]);
  if (state.mode !== 'edit' || state.stopped) return;
  renderEditor();
  scheduleSafety();
}

async function suggest() {
  try {
    const suggestion = await get('/servers/suggest');
    if (!state.editor) return;
    state.editor.form.port = state.editor.form.port || suggestion.port;
    state.editor.form.image = state.editor.form.image || suggestion.image;
  } catch (error) {
    console.error('port suggestion failed', error);
  }
}

function closeEditor() {
  state.editor = null;
  state.detailKey = '';
  renderList();
}

function renderEditor() {
  const editor = state.editor;
  const input = (key, props = {}) => h('input', {
    type: 'text',
    value: String(editor.form[key] ?? ''),
    onInput: (e) => { editor.form[key] = e.target.value; },
    ...props,
  });

  state.nodes.safety = h('div', { class: 'row', style: { flex: '1 1 340px' } },
    h('span', { class: 'faint small' }, 'checking the memory budget…'));
  state.nodes.footActions = h('div', { class: 'row' });

  const flagCount = [...state.schema.featured, ...state.schema.advanced]
    .reduce((total, section) => total + section.flags.length, 0);

  // The field that gets typed most, so the one that most wants a list. Refresh
  // lives here rather than on all seventeen pickers: one scan feeds them all.
  const [modelPicker] = diskPicker('model', {
    value: editor.form.model,
    lead: 'choose a model…',
    placeholder: 'org/repo, or a path the container can see',
    onChange: (next) => { editor.form.model = next; },
    extra: h('button', {
      class: 'btn-sm',
      title: 'Re-read the model cache and the outputs directory',
      onClick: (event) => refreshPaths(event.currentTarget),
    }, 'Refresh'),
  });

  const basics = h('div', { class: 'param-grid' },
    field('Name', input('name', { placeholder: 'qwen3-chat' }), {
      help: 'Names the container and identifies the server everywhere in this dashboard.',
    }),
    field('Model', modelPicker, {
      flag: 'positional',
      help: 'What is cached on this box, plus what the Fine-tune and Heretic tabs have written '
        + 'under /outputs. Paths are as the container sees them. A Hub id that has never been '
        + 'pulled goes under "Other" and is downloaded when the server first starts.',
    }),
    field('Port', input('port', { type: 'number', min: '1024', max: '65535' }), {
      flag: '--port',
      help: 'A host port: the container runs with --network host, so there is no mapping.',
    }),
    field('Image', input('image'), {
      help: 'Leave the suggested image unless you built your own.',
    }),
    field('Served name', input('served_name', { placeholder: 'optional' }), {
      flag: '--served-model-name',
      help: 'What clients put in the "model" field. Defaults to the model id.',
    }),
    field('Notes', input('notes', { placeholder: 'why this configuration exists' }), {
      help: 'Free text for whoever reads this in three months.',
    }),
    field('Environment', h('textarea', {
      rows: 3,
      placeholder: 'KEY=value, one per line',
      value: editor.form.env,
      onInput: (e) => { editor.form.env = e.target.value; },
    }), {
      help: 'Extra container environment. HF_HOME, HF_TOKEN and the NCCL variables are '
        + 'already set for you.',
    }),
    field('Autostart', h('input', {
      type: 'checkbox',
      checked: editor.form.autostart,
      onChange: (e) => { editor.form.autostart = e.target.checked; },
    }), { inline: true, help: 'Bring this server up when the dashboard starts.' }));

  const search = h('input', {
    type: 'search',
    placeholder: `Filter ${flagCount} flags by name or help text…`,
    onInput: debounce((event) => applyFilter(event.target.value), 160),
  });

  mount(state.main,
    panel(editor.id ? `Edit ${editor.form.name}` : 'New server', {
      sub: `${state.schema.image} · vLLM ${state.schema.vllm_version}`,
      actions: h('button', { onClick: closeEditor }, 'Cancel'),
      body: h('div', { class: 'serve-form' },
        basics,
        h('div', { class: 'param-section' },
          h('h3', null, 'Presets'),
          h('p', { class: 'blurb' },
            'Two of these are configurations that have actually run on this box; the third is '
            + 'derived from them. Applying one sets only the flags it names.'),
          h('div', { class: 'serve-presets' }, PRESETS.map((preset) => h('button', {
            class: 'serve-preset',
            onClick: () => applyPreset(preset),
          }, h('b', null, preset.title), h('span', null, preset.note))))),
        h('div', { class: 'param-search' }, search),
        state.schema.featured.map((section) => paramSection(section)),
        h('div', { class: 'param-section' },
          h('h3', null, 'All other parameters'),
          h('p', { class: 'blurb' },
            'Every remaining flag this image accepts, grouped the way vLLM groups them. '
            + `${state.schema.managed.join(', ')} are set by the dashboard and are not listed.`),
          state.schema.advanced.map((section) => paramSection(section, { collapsed: true })))),
    }),
    h('div', { class: 'serve-foot' }, state.nodes.safety, state.nodes.footActions));

  renderFootActions();
}

function applyPreset(preset) {
  const missing = [];
  for (const [dest, value] of Object.entries(preset.args)) {
    const arg = state.flagIndex.get(dest);
    if (!arg) {
      missing.push(dest);
      continue;
    }
    setArg(arg, value);
    state.editor.setters.get(dest)?.(value);
  }
  if (missing.length) {
    toast(`This vLLM build has no ${missing.join(', ')}; the rest of the preset was applied.`,
      { level: 'warn' });
  }
  scheduleSafety();
}

const scheduleSafety = debounce(() => checkSafety(), 300);

async function checkSafety() {
  if (state.mode !== 'edit' || !state.editor) return;
  const util = state.editor.args.gpu_memory_utilization;
  const query = util === undefined || util === null || util === '' ? '' : `?util=${util}`;
  let verdict;
  try {
    verdict = await get(`/system/budget/check${query}`);
  } catch (error) {
    console.error('budget check failed', error);
    return;
  }
  if (state.mode !== 'edit' || !state.editor) return;
  state.editor.verdict = verdict;
  mount(state.nodes.safety, verdictNotice(verdict));
  renderFootActions();
}

function renderFootActions() {
  const host = state.nodes.footActions;
  if (!host) return;
  const blocked = state.editor?.verdict?.level === 'block';
  mount(host,
    h('button', { onClick: closeEditor }, 'Cancel'),
    h('button', { onClick: () => save({ start: false }) }, 'Save'),
    h('button', {
      class: 'btn-primary',
      disabled: blocked,
      title: blocked ? state.editor.verdict.message : '',
      onClick: () => save({ start: true }),
    }, 'Save & start'),
    blocked
      ? h('button', { class: 'btn-danger', onClick: () => saveAndForce() }, 'Save & start anyway')
      : null);
}

async function saveAndForce() {
  const go = await confirmDialog('Start past the memory guard?',
    state.editor?.verdict?.message || 'This launch does not fit the budget.',
    { confirmLabel: 'Save & start anyway' });
  if (go) await save({ start: true, force: true });
}

function parseEnv(text) {
  const env = {};
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    const split = trimmed.indexOf('=');
    if (split <= 0) continue;
    env[trimmed.slice(0, split).trim()] = trimmed.slice(split + 1).trim();
  }
  return env;
}

async function save({ start = false, force = false } = {}) {
  const editor = state.editor;
  const form = editor.form;
  const port = Number(form.port);
  if (!form.name.trim() || !form.model.trim()) {
    toast('A name and a model are required.', { level: 'danger' });
    return;
  }
  if (!Number.isInteger(port) || port < 1024 || port > 65535) {
    toast('Port must be a whole number between 1024 and 65535.', { level: 'danger' });
    return;
  }

  const payload = {
    name: form.name.trim(),
    model: form.model.trim(),
    port,
    served_name: form.served_name.trim(),
    image: form.image.trim() || null,
    args: { ...editor.args },
    env: parseEnv(form.env),
    notes: form.notes,
    autostart: form.autostart,
  };

  let saved;
  try {
    saved = editor.id
      ? await patch(`/servers/${editor.id}`, payload)
      : await post('/servers', payload);
  } catch (error) {
    toast(error.message, { level: 'danger', title: 'Save failed' });
    return;
  }

  toast(`Saved "${saved.name}".`);
  state.editor = null;
  state.selected = saved.id;
  state.detailTab = start ? 'logs' : 'command';
  state.detailKey = '';
  state.log = { lines: [], box: null };
  state.verdict = null;
  await refreshStatus({ render: false });
  renderList();
  loadDetailBody();
  if (start) await startServer(saved.id, { force });
  refreshVerdict();
}
