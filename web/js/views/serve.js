/* The vLLM control surface: what is resident, what it costs in host memory, and
   every flag the image will accept.

   The parameter form is built from GET /api/servers/schema, which the backend
   generates from the image itself, so the form cannot drift from the binary
   that will run. The only hard-coded vLLM flags in this file are in PRESETS,
   and each of those carries the provenance of its numbers in the UI.

   A server also names the machine it runs on. Memory is the one thing that does
   not travel between them, so every figure in this view belongs to exactly one
   node and says which.

   The one exception is a pooled server: one engine split by layer across
   several machines, whose memory really is the sum of theirs. Nothing local can
   answer for it — the per-node budget check least of all — so everything a
   pooled definition shows comes from POST /api/servers/pool/plan, which asks
   every node in turn. */

import { ApiError, del, get, getText, patch, post, stream } from '../api.js';
import {
  badge, bytes, confirmDialog, copyButton, count, debounce, empty, ensureStyles, field, h,
  logBox, modal, mount, notice, panel, pct, spinner, stat, toast,
} from '../ui.js';

const STYLES = `
/* The recommendation. Directly under the profile, because it is the answer to
   what the profile just described. */
.serve-rec {
  border: 1px solid var(--border); border-left: 3px solid var(--ok);
  border-radius: var(--radius); background: var(--bg-raised);
  padding: 11px 13px; margin-bottom: var(--gap);
}
.serve-rec.lv-warn { border-left-color: var(--warn); }
.serve-rec.lv-danger { border-left-color: var(--danger); }
.sr-head { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.sr-list { display: flex; flex-direction: column; gap: 5px; margin-top: 9px; }
.sr-item { display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; }
.sr-item code {
  font-family: var(--mono); font-size: 12px; white-space: nowrap;
  background: var(--bg-sunken); border: 1px solid var(--border);
  border-radius: var(--radius-s); padding: 1px 6px; color: var(--text);
}
.sr-item.done code { opacity: .65; }
.sr-why { font-size: 11.5px; color: var(--text-dim); flex: 1 1 320px; min-width: 0; }
.serve-rec .notice { margin-top: 9px; }
.sr-left { margin-top: 10px; }
.sr-left summary { cursor: pointer; font-size: 11.5px; color: var(--text-faint); }
.sr-left summary:hover { color: var(--text-dim); }

.serve-pool-budgets { display: flex; flex-direction: column; gap: 6px; margin-top: 10px; }
.serve-pool-budgets .ov-key { display: inline-flex; align-items: center; gap: 6px; }

/* What the model is, read off disk. Sits directly under the basics so the
   answer is beside the question that provoked it. */
.serve-profile {
  border: 1px solid var(--border); border-left: 3px solid var(--accent);
  border-radius: var(--radius); background: var(--bg-raised);
  padding: 11px 13px; margin-bottom: var(--gap);
}
.serve-profile.bad { border-left-color: var(--danger); }
.sp-head { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; margin-bottom: 10px; }
.sp-head .truncate { max-width: 42ch; }
.sp-facts { display: grid; gap: 10px; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); }
.sp-fact { display: flex; flex-direction: column; gap: 1px; min-width: 0; }
.sp-label { font-size: 10.5px; letter-spacing: .05em; text-transform: uppercase;
  color: var(--text-faint); }
.sp-value { font-family: var(--mono); font-size: 14px; color: var(--text); }
.sp-hint { font-size: 11px; color: var(--text-dim); }

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
.serve-peer > td:first-child { border-left: 3px solid var(--accent); }
.serve-group > td { background: var(--bg-sunken); }
.serve-group .s-name { font-family: var(--mono); }
.serve-pick { display: flex; flex-direction: column; gap: 6px; }
.serve-pick-row { display: flex; align-items: center; gap: 6px; }
.serve-pick-row select { flex: 1 1 auto; min-width: 0; }
.serve-seg { display: inline-flex; border: 1px solid var(--border); border-radius: var(--radius);
  overflow: hidden; align-self: flex-start; }
.serve-seg button { border: 0; border-radius: 0; background: none; color: var(--text-dim);
  padding: 6px 12px; font-size: 12.5px; }
.serve-seg button[aria-pressed="true"] { background: var(--accent-dim); color: var(--text); }
.serve-pool { display: grid; gap: 10px; }
.serve-pool-list { display: flex; flex-direction: column; gap: 6px; }
.serve-pool-item { display: flex; align-items: center; gap: 9px; padding: 7px 10px;
  border: 1px solid var(--border); border-radius: var(--radius); background: var(--bg-sunken); }
.serve-pool-item .p-name { font-weight: 600; }
.serve-pool-item .p-wire { font-family: var(--mono); font-size: 11.5px; color: var(--text-dim); }
.serve-pool-add { display: flex; align-items: center; gap: 6px; }
.serve-pool-add select { flex: 0 1 340px; min-width: 0; }
.serve-pool-plan { display: grid; gap: 8px; }
.serve-sync { display: grid; gap: 6px; padding: 9px 10px; border: 1px solid var(--border);
  border-radius: var(--radius); background: var(--bg-sunken); }
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
const LOCAL = 'local';

const state = {
  ctx: null,
  main: null,
  schema: null,
  flagIndex: null,
  status: { servers: [], foreign: [], nodes: [], budgets: {}, budget: null },
  cluster: [],
  mode: 'list',
  selected: null,
  detailTab: 'command',
  detailKey: '',
  verdict: null,
  editor: null,
  nodes: {},
  log: { lines: [], box: null },
  // What the selected pooled server's machines are doing, and what they could
  // hold. Both are cluster-wide questions, so neither can come off /api/servers.
  pool: { forServer: null, status: null, plan: null },
  sync: null,
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
          'vLLM engines on the machines in this cluster. Utilisation is a fraction of the memory '
          + 'the CPU and GPU share on one node, so every engine on a node spends the same pool — '
          + 'and none of it is shared with the node next to it.')),
      h('div', { class: 'page-actions' },
        h('button', { onClick: () => { refreshStatus(); loadCluster(); } }, 'Refresh'),
        h('button', { class: 'btn-primary', onClick: () => openEditor(null) }, 'New server'))),
    state.main);

  const [schema] = await Promise.all([
    get('/servers/schema'), refreshStatus({ render: false }), loadCluster(),
  ]);
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
  // Pool status inspects a container on every peer over ssh, which is far too
  // slow to ride along with the three-second detail poll.
  state.timers.push(setInterval(refreshPoolStatus, 10000));
}

export function dispose() {
  state.stopped = true;
  for (const timer of state.timers) clearInterval(timer);
  state.timers = [];
  closeSync();
  state.editor = null;
  state.selected = null;
  state.detailKey = '';
  state.nodes = {};
  state.cluster = [];
  state.pool = { forServer: null, status: null, plan: null };
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

/* --- the cluster --------------------------------------------------------- */

/* GET /api/servers carries the registry, which is enough to place a server.
   GET /api/nodes additionally sshes into every peer to see whether it is up,
   which takes a second or so — worth it when the picker is about to offer a
   node, too expensive for the five-second list poll. */
async function loadCluster() {
  try {
    const payload = await get('/nodes');
    state.cluster = payload.nodes || [];
  } catch (error) {
    // The registry from /api/servers still names every node, so the picker
    // works; it just cannot say which of them are answering.
    toast(error.message, { level: 'warn', title: 'Could not read node status' });
  }
}

const nodeChoices = () => (state.cluster.length ? state.cluster : (state.status.nodes || []));

/** An unregistered name resolves to this machine on the backend (nodes.by_name),
 *  so only a name that matches a registered peer is remote. */
const isPeer = (name) => nodeChoices().some((node) => node.name === name && node.local === false);

/** The machines a server spans, head first. One name is not a pool: the backend
 *  only splits the engine when there are two or more. */
function poolOf(server) {
  const pool = server?.pool_nodes;
  return Array.isArray(pool) && pool.length > 1 ? pool.map(String) : [];
}

/* Head first, always, and joined rather than listed: the head is the node whose
   address clients hit, so the order is information and not presentation. */
const poolLabel = (pool) => pool.join(' + ');

const stagesPhrase = (pool) => `${pool.slice(1).join(', ')} `
  + `hold${pool.length === 2 ? 's' : ''} the later pipeline stages`;

const poolTitle = (pool) => `pooled engine · ${pool[0]} is the Ray head and serves the HTTP `
  + `frontend · ${stagesPhrase(pool)}`;

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
    sub: 'this machine · read-only',
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
  renderDetailSafety();
}

function budgetPanel() {
  const budget = state.status.budget;
  if (!budget) return null;
  const level = budget.free_util <= 0 ? 'crit' : budget.free_util < 0.1 ? 'warn' : '';
  const reserveWidth = Math.max(0,
    Math.min(1 - budget.occupied_util, budget.reserve_bytes / budget.total_bytes));
  return panel('Host memory budget', {
    sub: `this machine · ${budget.tenants.length} engine(s) resident`,
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
  // A single-machine install gets the table it always had: a header naming the
  // only node there is would be a row of noise.
  const clustered = (state.status.nodes || []).length > 1;
  mount(state.nodes.servers, h('div', { class: 'table-wrap' },
    h('table', null,
      h('thead', null, h('tr', null,
        h('th', null, 'Name'),
        h('th', { class: 'num' }, 'Port'),
        h('th', { class: 'num' }, 'Util'),
        h('th', null, 'Status'),
        h('th', null, 'Serving'),
        h('th', null, ''))),
      groupByNode(rows).map((group) => h('tbody', null,
        clustered ? nodeGroupRow(group) : null,
        group.servers.map(serverRow))))));
}

/** Servers grouped by where they run. A pooled server is not filed under its
 *  head: it does not belong to that machine any more than to the others, so its
 *  whole span is a group of its own. */
function groupByNode(rows) {
  const order = (state.status.nodes || []).map((node) => node.name);
  const groups = new Map();
  for (const server of rows) {
    const pool = poolOf(server);
    const head = pool.length ? pool[0] : (server.node || LOCAL);
    const key = pool.length ? `pool:${pool.join(',')}` : `node:${head}`;
    if (!groups.has(key)) groups.set(key, { head, pool, servers: [] });
    groups.get(key).servers.push(server);
  }
  // A server can still name a node that has since been deregistered. The
  // backend runs it here rather than losing it, so it goes last but stays.
  const rank = (group) => (order.indexOf(group.head) === -1 ? order.length
    : order.indexOf(group.head));
  return [...groups.values()].sort((a, b) => rank(a) - rank(b));
}

function nodeGroupRow(group) {
  if (group.pool.length) return poolGroupRow(group);
  const name = group.head;
  const node = (state.status.nodes || []).find((entry) => entry.name === name);
  const live = state.cluster.find((entry) => entry.name === name);
  const budget = (state.status.budgets || {})[name];
  const peer = Boolean(node) && node.local === false;

  return h('tr', { class: 'serve-group' },
    h('td', { colspan: '6' }, h('div', { class: 'row wrap' },
      h('span', { class: 's-name' }, name),
      badge(peer ? 'info' : 'absent', peer ? 'peer' : 'this machine'),
      node ? null : badge('failed', 'not registered'),
      live && live.reachable === false ? badge('failed', 'unreachable') : null,
      h('span', { class: 'faint small' }, `${group.servers.length} server(s)`),
      budget
        ? h('span', { class: 'faint small' },
          `${pct(budget.free_util, 0)} of a ${pct(budget.max_util, 0)} ceiling free · `
          + `${bytes(budget.available_bytes)} available of ${bytes(budget.total_bytes)}`)
        : null)));
}

function poolGroupRow(group) {
  const budgets = state.status.budgets || {};
  // The same sum cluster.plan() reports as pooled_bytes, from figures already in
  // hand — worth showing here rather than sshing to every node for the table.
  const known = group.pool.filter((name) => budgets[name]);
  const ceiling = known.reduce(
    (total, name) => total + budgets[name].max_util * budgets[name].total_bytes, 0);
  const down = group.pool.filter(
    (name) => state.cluster.some((entry) => entry.name === name && entry.reachable === false));

  return h('tr', { class: 'serve-group' },
    h('td', { colspan: '6' }, h('div', { class: 'row wrap' },
      h('span', { class: 's-name', title: poolTitle(group.pool) }, poolLabel(group.pool)),
      badge('info', 'pooled'),
      down.length ? badge('failed', `${down.join(', ')} unreachable`) : null,
      h('span', { class: 'faint small' }, `${group.servers.length} server(s)`),
      h('span', { class: 'faint small' },
        `${group.head} is the head and answers on its address`),
      known.length === group.pool.length
        ? h('span', { class: 'faint small' }, `${bytes(ceiling)} pooled ceiling`)
        : null)));
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
  const pool = poolOf(server);
  const peer = server.node_local === false;

  return h('tr', {
    class: `serve-row${peer || pool.length ? ' serve-peer' : ''}`
      + `${server.id === state.selected ? ' sel' : ''}`,
    onClick: () => selectServer(server.id),
  },
  h('td', null,
    h('div', { class: 's-name' }, server.name,
      pool.length
        ? h('span', { class: 'tag', style: { marginLeft: '6px' }, title: poolTitle(pool) },
          poolLabel(pool))
        : (peer ? h('span', { class: 'tag', style: { marginLeft: '6px' } }, server.node) : null)),
    h('div', { class: 's-model truncate', title: server.model }, server.model)),
  h('td', { class: 'num' }, String(server.port)),
  h('td', { class: 'num' }, utilCell(server.util)),
  h('td', null,
    statusBadge(server.status),
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
    h('td', null, statusBadge(item.status)),
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
  state.pool = { forServer: null, status: null, plan: null };
  state.log = { lines: [], box: null };
  renderServerTable();
  renderDetail();
  loadDetailBody();
  refreshVerdict();
  loadPoolDetail();
}

async function refreshVerdict() {
  const server = selectedServer();
  if (!server) return;
  // A pooled engine is sized against every node it spans; this machine's budget
  // is not the question. loadPoolDetail() answers the one that is.
  if (poolOf(server).length) {
    state.verdict = null;
    renderDetail();
    renderDetailSafety();
    return;
  }
  // /api/system/budget/check reads this machine's meminfo and this machine's
  // nvidia-smi. Running it for a server on a peer answers a question nobody
  // asked: it would refuse launches the peer has ample room for, and bless ones
  // it does not. The peer's own budget stands in its place.
  if (server.node_local === false) {
    state.verdict = null;
    renderDetail();
    renderDetailSafety();
    return;
  }
  const util = server.util;
  const query = util === null || util === undefined ? '' : `?util=${util}`;
  try {
    const verdict = await get(`/system/budget/check${query}`);
    if (state.selected !== server.id) return;
    state.verdict = verdict;
    renderDetail();
    renderDetailSafety();
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
  const pool = poolOf(server);
  const peer = server.node_local === false;
  state.nodes.detailBody = state.nodes.detailBody || h('div');
  // Held across rebuilds so the peer figures can be refreshed on the list poll
  // without re-parenting the log box below them.
  state.nodes.detailSafety = state.nodes.detailSafety || h('div');

  mount(state.nodes.detail, panel(server.name, {
    sub: pool.length
      ? `${server.model} · ${server.url} on ${pool[0]}, the head`
      : `${server.model} · ${server.url}`,
    actions: [
      pool.length
        ? badge('info', `pooled · ${poolLabel(pool)}`)
        : badge(peer ? 'info' : 'absent', server.node || LOCAL),
      statusBadge(server.status),
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
      state.nodes.detailSafety,
      state.nodes.detailBody),
  }));
  renderDetailSafety();
  return undefined;
}

function renderDetailSafety() {
  const host = state.nodes.detailSafety;
  const server = selectedServer();
  if (!host || !host.isConnected || !server) return;
  const pool = poolOf(server);
  if (pool.length) {
    mount(host, pooledDetailNotice(server, pool));
    return;
  }
  mount(host, server.node_local === false
    ? remoteLaunchNotice(server.node, server.util)
    : (state.verdict ? verdictNotice(state.verdict) : null));
}

/* What is true of a pooled engine and of nothing else here: it spans machines,
   its ceiling is their sum, and it dies whole. The per-node verdict the rest of
   the view shows would be an answer to a question nobody asked. */
function pooledDetailNotice(server, pool) {
  const status = state.pool.forServer === server.id ? state.pool.status : null;
  const plan = state.pool.forServer === server.id ? state.pool.plan : null;
  const workers = status?.workers || [];
  const joined = Number.isFinite(status?.nodes) ? status.nodes : null;
  const expected = status?.expected || pool.length;
  const live = LIVE.includes(server.status);
  // A pool that is short of nodes matters while the engine is meant to be up;
  // for a stopped definition it is simply the resting state.
  const short = live && status && status.running === false;

  return notice(short ? 'warn' : 'info',
    h('strong', null, `Pooled across ${poolLabel(pool)}. `),
    `${pool[0]} runs the Ray head and the HTTP frontend, so ${server.url} is the only address `
    + `clients use. ${stagesPhrase(pool)} — no client talks to them directly.`,
    h('div', { style: { marginTop: '4px' } },
      status
        ? (live
          ? `Ray: ${status.running ? 'up' : 'short of nodes'}`
            + `${joined === null ? '' : ` · ${joined} of ${expected} node(s) joined`}`
            + (workers.length
              ? ` · ${workers.map((w) => `${w.node} ${w.status}`).join(', ')}`
              : '')
          : `Not running: no Ray cluster for this engine yet. Starting it brings the head up on ${
            pool[0]} and a worker on ${pool.slice(1).join(', ')}.`)
        : 'Reading pool status…'),
    plan?.ok === false && plan.reason
      ? h('div', { class: 'faint', style: { marginTop: '4px' } },
        `A relaunch would be refused: ${plan.reason}`)
      : null,
    plan?.ok
      ? h('div', { class: 'faint', style: { marginTop: '4px' } },
        `Ceiling ${bytes(plan.pooled_bytes)} across ${plan.pipeline_parallel_size} machines, `
        + `against ${bytes(plan.single_node_bytes)} on ${pool[0]} alone · `
        + `--pipeline-parallel-size ${plan.pipeline_parallel_size}`)
      : null,
    fixedWorldSizeLine());
}

/* The single thing a pooled engine does that surprises people. vLLM builds the
   executor for exactly this world size; a node that goes away takes the whole
   engine with it, and without this sentence that reads as the dashboard
   breaking. */
const fixedWorldSizeLine = () => h('div', { class: 'faint', style: { marginTop: '4px' } },
  'Fixed world size: the engine is built for exactly these machines. If one of them leaves — a '
  + 'reboot, a dropped link, a stopped worker container — vLLM aborts the engine and it has to '
  + 'be started again. That is how vLLM\'s executor works, not a fault here.');

async function loadPoolDetail() {
  const server = selectedServer();
  if (!server) return;
  const pool = poolOf(server);
  if (!pool.length) {
    state.pool = { forServer: null, status: null, plan: null };
    return;
  }
  state.pool = { forServer: server.id, status: null, plan: null };
  renderDetailSafety();
  const query = `nodes=${pool.map(encodeURIComponent).join(',')}&server_id=${server.id}`;
  const [status, plan] = await Promise.all([
    get(`/servers/pool/status?${query}`).catch((error) => ({ error: error.message })),
    // The plan is what this pool could hold and whether it could start again;
    // it walks every node, so it is read once per selection, not on the poll.
    post('/servers/pool/plan', { nodes: pool, model: server.model, args: server.args || {} })
      .catch((error) => ({ ok: false, reason: error.message })),
  ]);
  if (state.pool.forServer !== server.id) return;
  state.pool.status = status;
  state.pool.plan = plan;
  renderDetailSafety();
}

async function refreshPoolStatus() {
  const server = selectedServer();
  if (state.mode !== 'list' || !server || state.pool.forServer !== server.id) return;
  const pool = poolOf(server);
  if (!pool.length) return;
  const query = `nodes=${pool.map(encodeURIComponent).join(',')}&server_id=${server.id}`;
  let status;
  try {
    status = await get(`/servers/pool/status?${query}`);
  } catch (error) {
    console.error('pool status failed', error);
    return;
  }
  if (state.pool.forServer !== server.id) return;
  state.pool.status = status;
  renderDetailSafety();
}

/* What there is to say about a launch aimed at another machine. The verdict the
   rest of this view shows is computed here, against this host; for a peer the
   honest substitute is that peer's own headroom plus where the decision is
   actually taken. Showing the local verdict instead would read as an answer
   about the wrong machine. */
function remoteLaunchNotice(name, util) {
  const budget = (state.status.budgets || {})[name];
  const asked = util === null || util === undefined
    ? 'No --gpu-memory-utilization is set, so vLLM applies its own default there.'
    : `You are asking for ${util} of that node.`;

  if (!budget) {
    return notice('info',
      h('strong', null, `Checked on ${name}, not here. `),
      `GET /api/servers reported no budget for ${name}, so there is nothing to show from it. `
      + `${asked} The memory guard runs on ${name} when the server starts.`);
  }

  const resident = budget.tenants.length
    ? budget.tenants.map((tenant) => `${tenant.name} ${tenant.util}`).join(', ')
    : 'nothing resident';

  return notice('info',
    h('strong', null, `Checked on ${name}, not here. `),
    `${name} reports ${pct(budget.free_util, 0)} of its ${pct(budget.max_util, 0)} ceiling free: `
    + `${bytes(budget.available_bytes)} available of ${bytes(budget.total_bytes)}, `
    + `${pct(budget.committed_util, 0)} already committed (${resident}). ${asked}`,
    h('div', { class: 'faint', style: { marginTop: '4px' } },
      'This is that node\'s headroom, not a verdict — the memory guard runs on '
      + `${name} at start time and it is the one that can refuse. Per-process GPU allocation is `
      + 'only readable on the machine running this dashboard, so a peer is measured by the '
      + 'utilisation fractions its containers declare.'));
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
  if (arg.dest === 'gpu_memory_utilization') {
    scheduleSafety();
    // A pooled definition has no single-node verdict; its answer comes from
    // the plan, so the plan is what has to be re-asked.
    if (state.editor?.form.pooled) schedulePlan();
  }
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

  // A size flag takes 32k, 1M or auto as readily as a plain integer, and a
  // number input can hold none of those — it silently blanks whatever it
  // cannot parse. So it stays text, and the value stays whatever was typed.
  if (arg.widget === 'size') {
    const box = h('input', {
      type: 'text',
      inputMode: 'numeric',
      placeholder: formatDefault(arg) || 'auto, 32k, or a number',
      value: value === undefined ? '' : String(value),
      onInput: (e) => put(e.target.value.trim() === '' ? undefined : e.target.value.trim()),
    });
    return [box, (v) => { box.value = v === undefined ? '' : String(v); }];
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
  // The debounce outlives the form it filters: saving or cancelling within the
  // keystroke window tears the editor down before this runs.
  if (!state.editor) return;
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

/** Docker's word for a state, in the operator's words.
 *  "absent" is accurate — no container exists — and reads like a fault to
 *  someone who has simply not pressed start yet. */
const STATUS_LABEL = {
  absent: 'not started',
  exited: 'stopped',
  failed: 'crashed',
  'oom-killed': 'out of memory',
};

const statusBadge = (status) => badge(status, STATUS_LABEL[status] || status);

/* --- naming ---------------------------------------------------------------

   A server's name and its served name are both derivable from the model, and
   typing them is the first of the several steps between "I want this model" and
   "it is serving". They fill themselves in — but only while they still hold
   what a previous model put there, so an edit is never overwritten. */

/** The tail of a model reference, as a container-safe slug.
 *  `Qwen/Qwen3-Embedding-8B` and `/outputs/heretic/Qwen3-Embedding-8B/` both
 *  become `qwen3-embedding-8b`. */
function slugFromModel(model) {
  const trimmed = String(model || '').trim().replace(/\/+$/, '');
  if (!trimmed) return '';
  const tail = trimmed.split('/').filter(Boolean).pop() || '';
  const slug = tail.toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 48)
    .replace(/-+$/, '');
  // Docker will not take a name that starts with a digit-free rule of its own,
  // but the dashboard's container name is llmd-vllm-<id>; this is the display
  // name, so the only real requirement is that it is not empty.
  return slug;
}

/** A name nothing else is already using. Two servers may legitimately point at
 *  one model — a long-context one and a fast one — so the second gets a -2. */
function uniqueName(base, selfId) {
  if (!base) return '';
  const taken = new Set((state.status.servers || [])
    .filter((server) => server.id !== selfId)
    .map((server) => String(server.name || '').toLowerCase()));
  if (!taken.has(base)) return base;
  for (let n = 2; n < 100; n += 1) {
    if (!taken.has(`${base}-${n}`)) return `${base}-${n}`;
  }
  return base;
}

/** Fill name and served name from the model, without ever clobbering an edit.
 *  A field is considered "still ours" while it is empty or still equal to what
 *  the previously selected model derived. */
function autoName(model) {
  const editor = state.editor;
  if (!editor) return;
  const slug = slugFromModel(model);
  const previous = editor.derived || { name: '', served: '' };

  const ours = (value, was) => !String(value || '').trim() || value === was;

  if (ours(editor.form.name, previous.name)) {
    editor.form.name = uniqueName(slug, editor.id);
    if (editor.refs?.name) editor.refs.name.value = editor.form.name;
  }
  if (ours(editor.form.served_name, previous.served)) {
    editor.form.served_name = slug;
    if (editor.refs?.served_name) editor.refs.served_name.value = slug;
  }
  editor.derived = { name: editor.form.name, served: editor.form.served_name };
}

/* --- what the model is ----------------------------------------------------

   Everything under here is read from the files a pull already fetched. It is
   reported, not acted on: the operator is the one deciding, and the reason
   this page was a guessing game is that the answers were never shown. */

const scheduleProfile = debounce(() => loadProfile(), 250);

async function loadProfile() {
  const editor = state.editor;
  if (state.mode !== 'edit' || !editor) return;
  const model = String(editor.form.model || '').trim();
  // A pooled engine reads the model from its head, which is the first node.
  const node = editor.form.pooled ? (editor.form.pool[0] || LOCAL) : editor.form.node;
  const key = `${node}\u0000${model}`;
  if (key === editor.profileFor) return;

  editor.profileFor = key;
  if (!model) {
    editor.profile = null;
    renderProfile();
    return;
  }
  renderProfile({ loading: true });
  renderRecommendation({ loading: true });
  try {
    // The recommendation carries the profile it was built from, so one call
    // answers both "what is this" and "what should it be set to".
    const rec = await post('/servers/recommend', { model, node, args: { ...editor.args } });
    if (state.mode !== 'edit' || !state.editor || state.editor.profileFor !== key) return;
    state.editor.profile = rec.profile;
    state.editor.rec = rec;
  } catch (error) {
    if (state.mode !== 'edit' || !state.editor || state.editor.profileFor !== key) return;
    state.editor.profile = { found: false, error: error.message };
    state.editor.rec = null;
  }
  renderProfile();
  renderRecommendation();
}

/* --- the recommendation ---------------------------------------------------

   Two flags and a reason for each, plus the reasons for everything it left
   alone. The button is the whole point of this page: the operator should not
   have to know which of ~190 flags matter before a model will load. */

/** Apply a set of {dest: value} through the form's own machinery.
 *  Going through state.flagIndex is not optional: the backend refuses a dest
 *  this image does not have, so a flag that would 422 on save is skipped here
 *  with a word about it instead. */
function applyArgs(args) {
  const missing = [];
  let applied = 0;
  for (const [dest, value] of Object.entries(args || {})) {
    const arg = state.flagIndex.get(dest);
    if (!arg) {
      missing.push(dest);
      continue;
    }
    setArg(arg, value);
    state.editor.setters.get(dest)?.(value);
    applied += 1;
  }
  return { applied, missing };
}

function applyRecommendation() {
  const rec = state.editor?.rec;
  if (!rec) return;
  const { applied, missing } = applyArgs(rec.args);
  if (missing.length) {
    toast(`This vLLM build has no ${missing.join(', ')}; the rest was applied.`,
      { level: 'warn' });
  }
  toast(applied ? `Applied ${applied} flag${applied === 1 ? '' : 's'}.` : 'Nothing to change.',
    { level: 'ok' });
  scheduleSafety();
  if (state.editor.form.pooled) schedulePlan();
  // The recommendation is relative to what is set, so re-ask it.
  state.editor.profileFor = '';
  loadProfile();
}

function recLevelClass(level) {
  return level === 'block' ? 'danger' : level === 'warn' ? 'warn' : 'ok';
}

function renderRecommendation(options = {}) {
  const host = state.nodes.recBox;
  if (!host) return;
  const editor = state.editor;
  if (!editor || !String(editor.form.model || '').trim()) return mount(host);

  if (options.loading) {
    return mount(host, h('div', { class: 'serve-rec' },
      h('div', { class: 'row' }, spinner(),
        h('span', { class: 'faint small' }, 'working out what this needs…'))));
  }

  const rec = editor.rec;
  if (!rec) return mount(host);

  const pending = Object.entries(rec.args || {})
    .filter(([dest, value]) => editor.args[dest] !== value);

  mount(host, h('div', { class: `serve-rec lv-${recLevelClass(rec.level)}` },
    h('div', { class: 'sr-head' },
      h('strong', null, rec.headline),
      h('span', { class: 'spacer' }),
      rec.ok && pending.length
        ? h('button', { class: 'btn-primary btn-sm', onClick: applyRecommendation },
          `Apply ${pending.length} flag${pending.length === 1 ? '' : 's'}`)
        : null),
    (rec.suggestions || []).length
      ? h('div', { class: 'sr-list' }, rec.suggestions.map((s) => {
        const already = editor.args[s.dest] === s.value;
        return h('div', { class: `sr-item${already ? ' done' : ''}` },
          h('code', null, `${flagOf(s.dest)} ${formatValue(s.value)}`),
          already ? badge('succeeded', 'set') : null,
          h('span', { class: 'sr-why' }, s.why));
      }))
      : null,
    (rec.findings || []).map((f) => notice(
      f.level === 'block' ? 'danger' : f.level === 'warn' ? 'warn' : 'info', f.text)),
    (rec.left_alone || []).length
      ? h('details', { class: 'sr-left' },
        h('summary', null, `${rec.left_alone.length} flags deliberately left alone`),
        h('div', { class: 'sr-list' }, rec.left_alone.map((entry) =>
          h('div', { class: 'sr-item' },
            h('code', null, flagOf(entry.dest)),
            h('span', { class: 'sr-why' }, entry.why)))))
      : null));
}

const flagOf = (dest) => state.flagIndex.get(dest)?.flag || `--${dest.replace(/_/g, '-')}`;

const formatValue = (value) => (value === true ? '' : String(value));

const CTX_FMT = (tokens) => (tokens >= 1024 ? `${Math.round(tokens / 1024)}k` : String(tokens));

function profileFacts(profile) {
  const facts = [];
  const arch = (profile.architectures || [])[0] || profile.model_type;
  if (arch) facts.push(['Architecture', arch, profile.model_type || '']);
  if (profile.max_position_embeddings) {
    // The KV cost of that context is the number that decides whether the
    // default max-model-len is a plan or a wish, so it is shown beside it.
    const kv = profile.kv_bytes_full;
    facts.push(['Context', `${CTX_FMT(profile.max_position_embeddings)} tokens`,
      kv ? `${bytes(kv)} of KV cache at full length` : 'full length']);
  }
  const scale = profile.parameters
    ? `${(profile.parameters / 1e9).toFixed(1)}B params`
    : (profile.quant_method ? `${profile.quant_method} quantised` : (profile.dtype || ''));
  facts.push(['Weights', profile.weight_bytes ? bytes(profile.weight_bytes) : '—', scale]);
  facts.push(['Chat template',
    profile.chat_template ? 'present' : 'none',
    profile.chat_template ? profile.chat_template_source : '/v1/chat/completions will refuse']);
  return facts;
}

function profileFlags(profile) {
  const flags = [];
  if (profile.supported === false) flags.push(badge('failed', 'architecture not in this image'));
  if (profile.runner === 'pooling') flags.push(badge('starting', 'embeddings, not chat'));
  if (profile.is_multimodal) flags.push(badge('info', 'multimodal'));
  if (profile.requires_remote_code) flags.push(badge('starting', 'needs trust-remote-code'));
  if (profile.is_adapter) flags.push(badge('failed', 'LoRA adapter, not a model'));
  if (profile.has_gguf && !profile.has_safetensors) flags.push(badge('failed', 'GGUF only'));
  if (profile.num_experts) flags.push(badge('plain', `${profile.num_experts} experts`));
  // rope_kind is worked out on the server, where the two spellings and the
  // nested-by-layer-type shape are already untangled.
  if (profile.rope_kind) flags.push(badge('plain', `rope ${profile.rope_kind}`));
  if (profile.source === 'peer') flags.push(badge('plain', 'read over ssh'));
  return flags;
}

function renderProfile(options = {}) {
  const host = state.nodes.profileBox;
  if (!host) return;
  const editor = state.editor;
  const model = String(editor?.form.model || '').trim();

  if (!model) return mount(host);
  if (options.loading) {
    return mount(host, h('div', { class: 'serve-profile' },
      h('div', { class: 'row' }, spinner(), h('span', { class: 'faint small' },
        'reading the files beside the weights…'))));
  }

  const profile = editor.profile;
  if (!profile) return mount(host);

  if (!profile.found) {
    // Not an error. A model that has never been pulled is a normal thing to
    // define a server for; the launch fetches it first.
    const where = editor.form.pooled ? (editor.form.pool[0] || LOCAL) : editor.form.node;
    return mount(host, notice('warn',
      h('strong', null, 'Not cached on '), h('strong', null, where), h('span', null, '. '),
      h('span', null, profile.error
        || 'Nothing is known about it until it is pulled — the first start downloads it, '
           + 'and the flags below cannot be checked against it until then.')));
  }

  const adapter = profile.is_adapter;
  const pooling = profile.runner === 'pooling';
  const unsupported = profile.supported === false;
  mount(host, h('div', { class: `serve-profile${adapter || unsupported ? ' bad' : ''}` },
    h('div', { class: 'sp-head' },
      h('strong', null, 'What this model is'),
      h('span', { class: 'row wrap' }, profileFlags(profile)),
      h('span', { class: 'spacer' }),
      h('span', { class: 'faint small mono truncate', title: profile.path }, profile.path)),
    h('div', { class: 'sp-facts' }, profileFacts(profile).map(([label, value, hint]) =>
      h('div', { class: 'sp-fact' },
        h('span', { class: 'sp-label' }, label),
        h('span', { class: 'sp-value' }, value),
        hint ? h('span', { class: 'sp-hint' }, hint) : null))),
    adapter
      ? h('p', { class: 'ov-note' },
        `This is a LoRA adapter. Serve ${profile.base_model || 'its base model'} with `
        + '--enable-lora and attach it there.')
      : null,
    // Not a flag anyone can override — it is resolved from the repo, and it
    // decides which endpoints exist. Better said here than discovered by a 400
    // four minutes after the engine started loading.
    pooling
      ? notice('warn',
        h('strong', null, 'This serves embeddings, not chat. '),
        h('span', null, `vLLM will run it as a pooling model: ${profile.runner_reason}. `
          + '/v1/embeddings works; /v1/chat/completions is never registered, so the route '
          + 'does not exist and the Playground cannot talk to it.'))
      : null,
    (profile.notes || []).length
      ? h('p', { class: 'ov-note' }, profile.notes.join(' · '))
      : null));
}

/* --- editor -------------------------------------------------------------- */

async function openEditor(server, prefill = {}) {
  state.mode = 'edit';
  closeSync();
  const pool = poolOf(server);
  state.editor = {
    id: server?.id ?? null,
    args: { ...(server?.args || {}) },
    verdict: null,
    plan: null,
    planning: false,
    planKey: '',
    syncChecks: new Map(),
    fields: new Map(),
    setters: new Map(),
    // The basics are plain inputs rather than generated flags, so autofill needs
    // its own handles on them.
    refs: {},
    // What the currently selected model derived. A field still holding this is
    // fair game to refill; anything else is the user's and is left alone.
    derived: null,
    // What the files beside the chosen model's weights say it is, and what
    // that implies for the handful of flags that decide whether it starts.
    profile: null,
    profileFor: '',
    rec: null,
    recFor: '',
    sections: [],
    searchable: [],
    paths: EMPTY_PATHS,
    pathViews: [],
    budgetFor: null,
    form: {
      name: server?.name || '',
      node: server?.node || LOCAL,
      pooled: pool.length > 0,
      pool,
      model: server?.model || prefill.model || '',
      port: server?.port || '',
      image: server?.image || '',
      served_name: server?.served_name || '',
      notes: server?.notes || '',
      autostart: Boolean(server?.autostart),
      env: Object.entries(server?.env || {}).map(([k, v]) => `${k}=${v}`).join('\n'),
    },
  };

  await Promise.all([loadPaths(), loadCluster(), server ? null : suggest()]);
  if (state.mode !== 'edit' || state.stopped) return;
  // A new definition arriving with a model already chosen — the Models page's
  // Serve button, or a preset — gets its names before it is ever shown. An
  // existing server keeps whatever it was saved with.
  if (!server && state.editor.form.model) autoName(state.editor.form.model);
  renderEditor();
  scheduleSafety();
  if (state.editor.form.pooled) runPlan();
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
  closeSync();
  state.editor = null;
  state.detailKey = '';
  renderList();
}

function renderEditor() {
  const editor = state.editor;
  const input = (key, props = {}) => {
    const el = h('input', {
      type: 'text',
      value: String(editor.form[key] ?? ''),
      onInput: (e) => { editor.form[key] = e.target.value; },
      ...props,
    });
    editor.refs[key] = el;
    return el;
  };

  state.nodes.safety = h('div', { class: 'row', style: { flex: '1 1 340px' } },
    h('span', { class: 'faint small' }, 'checking the memory budget…'));
  state.nodes.footActions = h('div', { class: 'row' });
  // Held across re-plans: the plan report is rebuilt whenever the answer
  // changes, but a running sync must keep its stream and its progress bar.
  state.nodes.planBox = h('div', { class: 'serve-pool-plan' });
  state.nodes.syncBox = h('div');
  state.nodes.poolBox = h('div', { class: 'param-section' });
  state.nodes.profileBox = h('div');
  state.nodes.recBox = h('div');

  const flagCount = [...state.schema.featured, ...state.schema.advanced]
    .reduce((total, section) => total + section.flags.length, 0);

  // The field that gets typed most, so the one that most wants a list. Refresh
  // lives here rather than on all seventeen pickers: one scan feeds them all.
  const [modelPicker] = diskPicker('model', {
    value: editor.form.model,
    lead: 'choose a model…',
    placeholder: 'org/repo, or a path the container can see',
    onChange: (next) => {
      editor.form.model = next;
      autoName(next);
      scheduleProfile();
      schedulePlan();
    },
    extra: h('button', {
      class: 'btn-sm',
      title: 'Re-read the model cache and the outputs directory',
      onClick: (event) => refreshPaths(event.currentTarget),
    }, 'Refresh'),
  });

  state.nodes.nodeField = field('Node', nodePicker(), {
    help: 'Which machine runs the container. Memory, ports and the model cache are that '
      + "machine's alone, so a model the list below offers is only cached here — a peer "
      + 'downloads it the first time that server starts.',
  });

  const basics = h('div', { class: 'param-grid' },
    field('Name', input('name', { placeholder: 'qwen3-chat' }), {
      help: 'Names the container and identifies the server everywhere in this dashboard.',
    }),
    field('Placement', placementControl(), {
      help: 'One engine on one machine, or one engine split by layer across several so their '
        + 'memory adds up. Pooling costs a network hop per token per stage boundary and a fixed '
        + 'world size; a model that fits on one box should stay on one box.',
    }),
    state.nodes.nodeField,
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
        state.nodes.profileBox,
        state.nodes.recBox,
        state.nodes.poolBox,
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

  renderPoolBox();
  syncPlacement();
  renderFootActions();
  renderProfile();
  renderRecommendation();
  loadProfile();
}

/** A select over the registry. Unreachable peers stay selectable: a node that
 *  is down now is still where this server belongs. */
function nodePicker() {
  const editor = state.editor;
  const choices = nodeChoices();
  const label = (node) => {
    const parts = [node.local ? 'this machine' : (node.address || 'peer')];
    if (node.reachable === false) parts.push('unreachable');
    // The local node's registry note is "this machine", which parts[0] just said.
    if (node.note && node.note !== parts[0]) parts.push(node.note);
    return `${node.name} — ${parts.join(' · ')}`;
  };

  const select = h('select', {
    onChange: (event) => {
      editor.form.node = event.target.value;
      // A model cached here and a model cached on a peer are different
      // questions, and it is the peer that is about to load it.
      scheduleProfile();
      scheduleSafety();
    },
  },
  choices.map((node) => h('option', {
    value: node.name,
    selected: node.name === editor.form.node,
  }, label(node))),
  // A server saved against a node that has since been deregistered keeps its
  // name here rather than silently becoming one of the others.
  choices.some((node) => node.name === editor.form.node)
    ? null
    : h('option', { value: editor.form.node, selected: true },
      `${editor.form.node} — not registered, runs here`));

  return select;
}

/* --- pooling ------------------------------------------------------------- */

/* A pooled definition answers to nothing local: which machines, in which order,
   and whether the cluster can actually take the launch all come from
   POST /api/servers/pool/plan. The form re-asks it on every change to the node
   set or the model, because both are things the plan is about. */

function placementControl() {
  const seg = h('div', { class: 'serve-seg' });
  const single = nodeChoices().length < 2;
  const button = (pooled, label) => h('button', {
    type: 'button',
    'aria-pressed': String(state.editor.form.pooled === pooled),
    disabled: pooled && single,
    title: pooled && single
      ? 'Only one machine is registered. Add a peer on the Nodes tab first.' : '',
    onClick: () => setPooled(pooled),
  }, label);
  state.nodes.segButtons = [button(false, 'One machine'), button(true, 'Pooled across machines')];
  mount(seg, state.nodes.segButtons);
  return seg;
}

function setPooled(pooled) {
  const editor = state.editor;
  if (editor.form.pooled === pooled) return;
  editor.form.pooled = pooled;
  if (pooled && editor.form.pool.length < 2) editor.form.pool = defaultPool();
  if (!pooled) editor.plan = null;
  syncPlacement();
  renderPoolBox();
  renderFootActions();
  if (pooled) runPlan();
  else scheduleSafety();
}

/** The pool the operator most likely wants: whatever the single-node picker was
 *  already pointing at, plus the first other machine. */
function defaultPool() {
  const names = nodeChoices().map((node) => node.name);
  const head = names.includes(state.editor.form.node) ? state.editor.form.node : (names[0] || LOCAL);
  const next = names.find((name) => name !== head);
  return next ? [head, next] : [head];
}

function syncPlacement() {
  const pooled = state.editor.form.pooled;
  for (const [index, button] of (state.nodes.segButtons || []).entries()) {
    button.setAttribute('aria-pressed', String(pooled === (index === 1)));
  }
  // The single-node picker is not merely irrelevant when pooling: the backend
  // ignores `node` entirely for a pooled server, so leaving it on screen would
  // invite a choice that has no effect.
  if (state.nodes.nodeField) state.nodes.nodeField.hidden = pooled;
  if (state.nodes.poolBox) state.nodes.poolBox.hidden = !pooled;
}

function movePoolNode(index, delta) {
  const pool = state.editor.form.pool;
  const target = index + delta;
  if (target < 0 || target >= pool.length) return;
  [pool[index], pool[target]] = [pool[target], pool[index]];
  renderPoolBox();
  runPlan();
}

function removePoolNode(index) {
  state.editor.form.pool.splice(index, 1);
  renderPoolBox();
  runPlan();
}

function addPoolNode(name) {
  if (!name || state.editor.form.pool.includes(name)) return;
  state.editor.form.pool.push(name);
  renderPoolBox();
  runPlan();
}

function renderPoolBox() {
  const host = state.nodes.poolBox;
  if (!host || !state.editor) return;
  const pool = state.editor.form.pool;
  const spare = nodeChoices().filter((node) => !pool.includes(node.name));
  const picker = h('select', null,
    h('option', { value: '' },
      spare.length ? 'add a machine…' : 'every registered machine is already in this pool'),
    spare.map((node) => h('option', { value: node.name },
      `${node.name} — ${node.local ? 'this machine' : (node.address || 'peer')}`)));

  mount(host,
    h('h3', null, 'Pooled across machines'),
    h('p', { class: 'blurb' },
      'One engine, split by layer — pipeline parallel, not tensor parallel, because each Spark '
      + 'has a single GPU. The order below is the pipeline order: the first machine runs the Ray '
      + 'head and the HTTP frontend, so its address is the one clients use, and the rest hold '
      + 'later stages and serve nothing directly.'),
    h('div', { class: 'serve-pool' },
      h('div', { class: 'serve-pool-list' }, pool.map(poolItem)),
      h('div', { class: 'serve-pool-add' },
        picker,
        h('button', {
          class: 'btn-sm',
          disabled: !spare.length,
          onClick: () => addPoolNode(picker.value),
        }, 'Add')),
      state.nodes.planBox,
      state.nodes.syncBox));
  renderPlan();
}

function poolItem(name, index) {
  const pool = state.editor.form.pool;
  const node = nodeChoices().find((entry) => entry.name === name);
  const wire = (state.editor.plan?.nodes || []).find((entry) => entry.name === name);
  const last = pool.length <= 2;

  return h('div', { class: 'serve-pool-item' },
    badge('info', index === 0 ? 'head' : `stage ${index + 1}`),
    h('span', { class: 'p-name' }, name),
    wire && wire.address
      ? h('span', { class: 'p-wire', title: 'the interface on this node carrying the cluster '
        + 'subnet — NCCL is pointed at this one by name, and the name differs per machine' },
        `${wire.interface} ${wire.address}`)
      : h('span', { class: 'faint small' },
        node?.local ? 'this machine' : (node?.address || 'not registered')),
    wire
      ? h('span', { class: 'faint small' },
        `${pct(wire.free_util, 0)} free to commit of ${bytes(wire.total_bytes)}`)
      : null,
    node && node.reachable === false ? badge('failed', 'unreachable') : null,
    h('span', { class: 'spacer' }),
    h('button', {
      class: 'btn-sm', disabled: index === 0, title: 'earlier in the pipeline',
      onClick: () => movePoolNode(index, -1),
    }, '↑'),
    h('button', {
      class: 'btn-sm', disabled: index === pool.length - 1, title: 'later in the pipeline',
      onClick: () => movePoolNode(index, 1),
    }, '↓'),
    h('button', {
      class: 'btn-sm btn-ghost',
      disabled: last,
      title: last ? 'a pool needs at least two machines' : `drop ${name} from the pool`,
      onClick: () => removePoolNode(index),
    }, 'Remove'));
}

const schedulePlan = debounce(() => runPlan(), 400);

async function runPlan() {
  const editor = state.editor;
  if (!editor || !editor.form.pooled) return;
  const pool = [...editor.form.pool];
  const model = editor.form.model.trim();
  // The arguments are part of the question now: the plan runs the memory guard
  // on every node in the pool, so a change to the utilisation fraction is a
  // different question and has to re-ask it.
  const args = { ...editor.args };
  const key = `${pool.join(',')}|${model}|${JSON.stringify(args)}`;
  editor.planKey = key;

  if (pool.length < 2) {
    editor.planning = false;
    editor.plan = { ok: false, reason: 'pooling needs at least two machines' };
    renderPlan();
    renderFootActions();
    mountPoolSafety();
    return;
  }

  editor.planning = true;
  renderPlan();
  mountPoolSafety();
  let plan;
  try {
    plan = await post('/servers/pool/plan', { nodes: pool, model, args });
  } catch (error) {
    plan = { ok: false, reason: error.message };
  }
  // A slower answer to an older question would overwrite a newer one.
  if (state.editor !== editor || editor.planKey !== key) return;
  editor.planning = false;
  editor.plan = plan;
  editor.syncChecks = new Map();
  renderPoolBox();
  renderFootActions();
  mountPoolSafety();
}

function renderPlan() {
  const host = state.nodes.planBox;
  const editor = state.editor;
  if (!host || !editor) return;
  const plan = editor.plan;

  if (editor.planning) {
    mount(host, h('div', { class: 'row' }, spinner(),
      h('span', { class: 'faint small' },
        'asking every machine for its interface, its image and its cache…')));
    return;
  }
  if (!plan) {
    mount(host);
    return;
  }

  mount(host,
    plan.ok
      ? notice('ok',
        h('strong', null, 'These machines can hold it. '),
        `${bytes(plan.pooled_bytes)} across ${plan.pipeline_parallel_size} machines, against `
        + `${bytes(plan.single_node_bytes)} on ${plan.head} alone — `
        + `${(plan.pooled_bytes / Math.max(plan.single_node_bytes, 1)).toFixed(2)}× the room, at `
        + `--pipeline-parallel-size ${plan.pipeline_parallel_size}.`)
      : notice('danger',
        h('strong', null, 'Cannot launch. '),
        plan.reason || 'the cluster cannot take this pool'),
    poolBudgets(plan),
    (plan.missing_model_on || []).map((name) => missingModelRow(name)),
    (plan.missing_image || []).length ? missingImageNotice(plan.missing_image) : null);
}

/** What each machine in the pool can give, and what this configuration asks of
 *  it. A pooled engine declares the same utilisation fraction on every node it
 *  spans, so the tightest one decides — and until now nothing in this form
 *  showed the fraction against any node at all. */
function poolBudgets(plan) {
  const rows = (plan.nodes || []).filter((node) => node.verdict);
  if (!rows.length) return null;
  const asked = rows[0].verdict.requested_util;
  return h('div', { class: 'serve-pool-budgets' },
    h('div', { class: 'faint small' },
      `Asking ${asked ? asked.toFixed(2) : '—'} of each machine; the tightest can give `
      + `${(plan.free_util ?? 0).toFixed(2)}.`),
    h('div', { class: 'row wrap' }, rows.map((node) => h('span', {
      class: 'ov-key',
      title: node.verdict.message,
    },
    badge(node.verdict.level === 'ok' ? 'running'
      : node.verdict.level === 'warn' ? 'starting' : 'failed', node.name),
    h('span', { class: 'faint small' },
      `${node.free_util.toFixed(2)} free of ${bytes(node.total_bytes)}`)))));
}

/* A blocker the operator can clear from here: the bytes are already on this
   machine and the cluster link is faster than the internet, so the fix is a
   copy, not a second download. */
function missingModelRow(name) {
  const model = state.editor.form.model.trim();
  // undefined: never asked. null: the check is in flight.
  const check = state.editor.syncChecks.get(name);
  const button = h('button', {
    class: 'btn-sm btn-primary',
    onClick: () => syncModelTo(name, button),
  }, `Sync to ${name}`);

  if (check === undefined) checkSync(name);

  return notice('warn',
    h('strong', null, `${model} is not cached on ${name}. `),
    'Every node loads its own shard from its own disk, so a launch would have each machine '
    + 'fetch the weights on its own, minutes in.',
    h('div', { class: 'row wrap', style: { marginTop: '6px' } },
      !check
        ? h('span', { class: 'faint small' }, 'measuring what would have to be copied…')
        : h('span', { class: 'faint small' },
          check.ok
            ? `${bytes(check.to_copy_bytes)} to copy over the cluster link`
              + (check.already_there_bytes
                ? ` · ${bytes(check.already_there_bytes)} already there` : '')
            : check.reason || 'that copy cannot be started'),
      check && check.ok === false ? null : button));
}

async function checkSync(name) {
  const editor = state.editor;
  const model = editor.form.model.trim();
  if (!model) return;
  editor.syncChecks.set(name, null);
  let check;
  try {
    check = await get(`/nodes/${encodeURIComponent(name)}/sync/check`
      + `?repo_id=${encodeURIComponent(model)}`);
  } catch (error) {
    check = { ok: false, reason: error.message };
  }
  if (state.editor !== editor || editor.form.model.trim() !== model) return;
  editor.syncChecks.set(name, check);
  renderPlan();
}

/* Nothing here can fix this one: there is no build endpoint for the ray image,
   so the honest answer is the command that builds it. */
function missingImageNotice(names) {
  return notice('danger',
    h('strong', null, `The Ray image is missing on ${names.join(', ')}. `),
    'The NGC vLLM image has no ray in it at all, so a pooled launch needs the derived image '
    + 'built from docker/vllm-ray.Dockerfile on every machine in the pool. There is no build '
    + 'endpoint for it — build it on each of those machines and re-plan:',
    h('div', { class: 'cmdbox', style: { marginTop: '6px' } },
      'docker build -f docker/vllm-ray.Dockerfile -t llmd-vllm-ray .'));
}

function syncModelTo(name, button) {
  const model = state.editor.form.model.trim();
  button.disabled = true;
  post(`/nodes/${encodeURIComponent(name)}/sync`, { repo_id: model })
    .then(({ job_id: jobId }) => followSync(jobId, name, model))
    .catch((error) => {
      button.disabled = false;
      toast(error.message, { level: 'danger', title: `Sync to ${name} failed` });
    });
}

function followSync(jobId, name, model) {
  closeSync();
  const bar = h('span', { style: { width: '0%' } });
  const nums = h('div', { class: 'faint small' }, 'starting…');
  const line = h('div', { class: 'faint small mono truncate' });
  const caveat = h('div');
  const status = badge('running', 'copying');
  const cancel = h('button', { class: 'btn-sm btn-danger' }, 'Cancel');
  cancel.addEventListener('click', () => {
    cancel.disabled = true;
    post(`/jobs/${jobId}/cancel`).catch((error) => {
      cancel.disabled = false;
      toast(error.message, { level: 'danger' });
    });
  });

  mount(state.nodes.syncBox, h('div', { class: 'serve-sync' },
    h('div', { class: 'row wrap' },
      h('strong', { class: 'small' }, `Copying ${model} → ${name}`),
      status,
      h('span', { class: 'spacer' }),
      h('span', { class: 'faint small mono' }, jobId),
      cancel),
    h('div', { class: 'progress' }, bar),
    nums,
    line,
    caveat));

  const apply = (progress) => {
    const percent = Number(progress.percent);
    if (Number.isFinite(percent)) bar.style.width = `${Math.max(0, Math.min(100, percent))}%`;
    mount(nums,
      h('span', null, Number.isFinite(percent) ? `${percent.toFixed(0)}%` : '—'),
      h('span', null, ` · ${bytes(progress.transferred_bytes || 0)} copied`),
      progress.speed_bps ? h('span', null, ` · ${bytes(progress.speed_bps)}/s`) : null,
      progress.files_total
        ? h('span', null, ` · ${progress.files_done ?? 0}/${progress.files_total} files`)
        : null,
      progress.elapsed ? h('span', null, ` · ${progress.elapsed}`) : null);
  };

  state.sync = {
    jobId,
    close: stream(`/jobs/${jobId}/stream`, {
      progress: (payload) => apply(payload?.progress || {}),
      status: (payload) => {
        mount(status, payload?.status || 'running');
        status.className = `badge ${payload?.status || 'running'}`;
        if (payload?.progress && Object.keys(payload.progress).length) apply(payload.progress);
      },
      // rsync rewrites one line with \r; the manager forwards it as transient.
      'progress-line': (payload) => { line.textContent = payload?.line ?? ''; },
      end: (payload) => {
        const finished = payload?.status || 'succeeded';
        mount(status, finished);
        status.className = `badge ${finished}`;
        cancel.remove();
        closeSync();
        // The plan's cache check is a directory test on the peer, so a copy
        // that stopped half way makes the blocker disappear without the model
        // actually being loadable. Nothing but this can tell the operator.
        if (finished !== 'succeeded') {
          mount(caveat, notice('warn',
            h('strong', null, `Copy ${finished} part way. `),
            `${name} now holds an incomplete copy. The plan below only checks that the model's `
            + 'directory exists there, so it will stop reporting this as a blocker — run the sync '
            + `again to finish it, or delete ${model} from ${name} before launching.`));
        }
        toast(`${model} → ${name}: ${finished}`,
          { level: finished === 'succeeded' ? 'ok' : 'warn' });
        // The blocker this cleared is the plan's, so the plan is what has to
        // agree that it is gone.
        runPlan();
      },
    }),
  };
}

function closeSync() {
  state.sync?.close?.();
  state.sync = null;
}

/* What stands beside the launch buttons for a pooled server. The per-node
   verdict has nothing to say about an engine spread over several nodes, so the
   plan's ceiling takes its place. */
function poolSafetyNotice() {
  const editor = state.editor;
  const plan = editor.plan;
  if (editor.planning || !plan) {
    return notice('info',
      h('strong', null, 'Pooled. '),
      'Working out what these machines can hold…',
      fixedWorldSizeLine());
  }
  if (!plan.ok) {
    return notice('danger',
      h('strong', null, 'Cannot launch. '),
      plan.reason || 'the cluster cannot take this pool',
      fixedWorldSizeLine());
  }
  return notice('ok',
    h('strong', null, `Pooled ceiling ${bytes(plan.pooled_bytes)}. `),
    `Across ${poolLabel(editor.form.pool)}, head first — ${plan.head} runs the HTTP frontend. `
    + `One machine alone would top out at ${bytes(plan.single_node_bytes)}. The per-machine `
    + 'memory guard does not apply: nothing on one node can judge this launch.',
    fixedWorldSizeLine());
}

function mountPoolSafety() {
  if (state.nodes.safety && state.editor?.form.pooled) {
    mount(state.nodes.safety, poolSafetyNotice());
  }
}

function applyPreset(preset) {
  const { missing } = applyArgs(preset.args);
  if (missing.length) {
    toast(`This vLLM build has no ${missing.join(', ')}; the rest of the preset was applied.`,
      { level: 'warn' });
  }
  scheduleSafety();
  if (state.editor.form.pooled) schedulePlan();
  // A preset can undo what the recommendation asked for, so re-ask it.
  state.editor.profileFor = '';
  loadProfile();
}

const scheduleSafety = debounce(() => checkSafety(), 300);

async function checkSafety() {
  if (state.mode !== 'edit' || !state.editor) return;
  // /api/system/budget/check answers for this machine alone, which is the wrong
  // question for an engine spread over several. The plan answers it instead.
  if (state.editor.form.pooled) {
    state.editor.verdict = null;
    mountPoolSafety();
    renderFootActions();
    return;
  }
  const node = state.editor.form.node;
  if (isPeer(node)) {
    // Budgets ride along with GET /api/servers, so re-read them once per node
    // the operator picks — not on every keystroke that moves the util slider,
    // which is what actually drives this call.
    if (state.editor.budgetFor !== node) {
      state.editor.budgetFor = node;
      await refreshStatus({ render: false });
      if (state.mode !== 'edit' || !state.editor) return;
    }
    state.editor.verdict = null;
    mount(state.nodes.safety,
      remoteLaunchNotice(node, state.editor.args.gpu_memory_utilization));
    renderFootActions();
    return;
  }
  state.editor.budgetFor = null;
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
  const editor = state.editor;
  const pooled = Boolean(editor?.form.pooled);
  // Forcing past the memory guard is a judgement call an operator is allowed to
  // make. A plan that says no is not: the ray image or the weights are missing,
  // and starting anyway just fails slowly on another machine.
  const planBlocked = pooled && (editor.planning || editor.plan?.ok !== true);
  const blocked = pooled ? planBlocked : editor?.verdict?.level === 'block';
  const why = pooled
    ? (editor.planning ? 'still planning across the pool' : (editor.plan?.reason || ''))
    : (editor?.verdict?.message || '');

  mount(host,
    h('button', { onClick: closeEditor }, 'Cancel'),
    h('button', { onClick: () => save({ start: false }) }, 'Save'),
    h('button', {
      class: 'btn-primary',
      disabled: blocked,
      title: blocked ? why : '',
      onClick: () => save({ start: true }),
    }, 'Save & start'),
    blocked && !pooled
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
  if (form.pooled && form.pool.length < 2) {
    toast('A pooled server needs at least two machines, in pipeline order.', { level: 'danger' });
    return;
  }

  const payload = {
    name: form.name.trim(),
    node: form.node || LOCAL,
    // More than one name is what makes it pooled, and the first is the head.
    // The backend ignores `node` for placement once this is set.
    pool_nodes: form.pooled ? [...form.pool] : [],
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
  closeSync();
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
  loadPoolDetail();
}
