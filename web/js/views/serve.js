/* The engine control surface: what is resident, what it costs in the memory its
   node actually spends, and every flag the image will accept.

   Two engines live here. A server names one — vLLM or llama.cpp — and that
   choice reaches almost everything below it: the flag form is generated from
   that engine's own binary, the presets are measurements of that engine, the
   memory verdict is phrased in the units that engine has (a fraction for vLLM,
   bytes for llama.cpp, which declares no fraction at all), and pooling is
   offered only for the engine that can do it. The editor therefore keeps a
   schema and an argument set PER ENGINE and swaps between them, rather than one
   of each: switching to llama.cpp and back must not throw away what was typed.

   The dropdown picks itself when the answer is obvious. A model reference with
   'gguf' in it can only be served by llama.cpp — vLLM cannot read the format —
   so choosing one selects the engine, and keeps selecting it only for as long
   as the operator has not overridden the choice by hand.

   "The memory its node spends" is the framebuffer on a discrete GPU and host
   RAM on a unified part, and the backend decides which before any figure gets
   here (app/accel.py). This view never assumes one of them.

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
const PRESETS_BY_ENGINE = {
  vllm: [
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
  ],

  /* llama.cpp's presets are shapes rather than measurements, and they say so.
     The reason is structural: a vLLM preset is a fraction of THIS box and is
     the same fraction whatever the model, so it can be measured once. A
     llama.cpp configuration is a layer count and a context length, and both
     depend on the .gguf — 40 layers is a whole model or a third of one. What
     transfers between models is the intent, so that is what these carry. */
  llamacpp: [
    {
      title: 'Everything on the GPU',
      note: 'The fast case, for a model that fits. -ngl all offloads every layer; if it '
        + 'does not fit, llama.cpp will not quietly do less — it fails. Check the memory '
        + 'verdict below before starting.',
      args: {
        n_gpu_layers: 'all',
        ctx_size: 8192,
        flash_attn: 'on',
      },
    },
    {
      title: 'Long context on a busy box',
      note: 'Quantising the KV cache to q8_0 roughly halves it against f16 for very little '
        + 'quality, which is what buys the context. Flash attention is required for a '
        + 'quantised V cache and also keeps the compute buffer from growing with length.',
      args: {
        n_gpu_layers: 'all',
        ctx_size: 32768,
        cache_type_k: 'q8_0',
        cache_type_v: 'q8_0',
        flash_attn: 'on',
        ubatch_size: 256,
      },
    },
    {
      title: 'Split with the CPU',
      note: 'What llama.cpp can do and vLLM cannot: run a model larger than the '
        + 'accelerator by keeping some layers on the CPU. It is slower per token in '
        + 'proportion to how much stayed behind. Start at half and raise it until the '
        + 'verdict says no.',
      args: {
        n_gpu_layers: 20,
        ctx_size: 4096,
        flash_attn: 'on',
      },
    },
  ],
};

const presetsFor = (engine) => PRESETS_BY_ENGINE[engine] || [];

const DETAIL_TABS = [['command', 'Command'], ['logs', 'Logs'], ['metrics', 'Metrics']];
const LIVE = ['running', 'loading', 'starting', 'unhealthy'];
const LOCAL = 'local';

const DEFAULT_ENGINE = 'vllm';

/* A model reference only llama.cpp can serve. Matched against the WHOLE trimmed
   reference rather than its tail, because both spellings occur: a Hub repo says
   it at the end (`bartowski/Qwen3-8B-GGUF`) and a fine-tune export says it in
   the middle (`/outputs/finetune/run7/gguf/model-q4_k_m.gguf`). */
const GGUF_HINT = /gguf/i;

const state = {
  ctx: null,
  main: null,
  // One schema per engine, fetched lazily. The default engine's is fetched at
  // render; a second engine's is fetched the first time it is selected, and its
  // failure is contained to the dropdown rather than to the whole view — see
  // schemaFor().
  schemas: {},
  engines: [],
  flagIndexes: {},
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
          'vLLM and llama.cpp engines on the machines in this cluster. Every engine on a node '
          + 'spends the same pool — the GPU\'s own framebuffer where it has one, the memory the '
          + 'CPU and GPU share where they share it — and none of it is shared with the node next '
          + 'to it. vLLM claims its share as a fraction of that pool; llama.cpp claims no share '
          + 'at all, so what it takes is worked out from its weights file and its context.')),
      h('div', { class: 'page-actions' },
        h('button', { onClick: () => { refreshStatus(); loadCluster(); } }, 'Refresh'),
        h('button', { class: 'btn-primary', onClick: () => openEditor(null) }, 'New server'))),
    state.main);

  // Only the default engine's schema is awaited here. Everything on this page —
  // the list, the timers, the detail panel — is downstream of this await, so a
  // second engine's schema failing to load would have taken the whole Serve tab
  // down with it. The others are fetched when they are first selected.
  const [schema, engineList] = await Promise.all([
    get('/servers/schema'), get('/servers/engines').catch(() => null),
    refreshStatus({ render: false }), loadCluster(),
  ]);
  if (state.stopped) return;
  rememberSchema(DEFAULT_ENGINE, schema);
  state.engines = engineList?.engines || [
    { name: DEFAULT_ENGINE, label: schema.label || 'vLLM' },
  ];

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

/* --- per-engine schemas --------------------------------------------------

   `state.schema` used to be one object fetched once. It is now one per engine,
   because the flag index built from it is the gate on everything that applies a
   flag programmatically: applyArgs silently drops any dest the index does not
   hold, which is what makes a preset degrade instead of exploding — and a stale
   index would let a vLLM preset write gpu_memory_utilization into a llama.cpp
   payload that then 422s on save. */

function rememberSchema(engine, schema) {
  state.schemas[engine] = schema;
  const index = new Map();
  for (const section of [...(schema.featured || []), ...(schema.advanced || [])]) {
    for (const arg of section.flags) index.set(arg.dest, arg);
  }
  state.flagIndexes[engine] = index;
  return schema;
}

const currentEngine = () => state.editor?.form.engine || DEFAULT_ENGINE;
const schemaOf = (engine) => state.schemas[engine || currentEngine()] || null;
const indexOf = (engine) => state.flagIndexes[engine || currentEngine()] || new Map();
const engineLabel = (name) => state.engines.find((e) => e.name === name)?.label
  || (name === 'llamacpp' ? 'llama.cpp' : 'vLLM');
const engineInfo = (name) => state.engines.find((e) => e.name === name) || {};

/** Fetch an engine's schema once, keeping the failure local.
 *  Returns null rather than throwing: an engine whose schema will not load is a
 *  dropdown entry that cannot be selected, not a broken page. */
async function schemaFor(engine) {
  if (state.schemas[engine]) return state.schemas[engine];
  try {
    return rememberSchema(engine, await get(`/servers/schema?engine=${encodeURIComponent(engine)}`));
  } catch (error) {
    toast(`Could not load the ${engineLabel(engine)} parameter list: ${error.message}`,
      { level: 'danger', title: 'Engine unavailable' });
    return null;
  }
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

const poolTitle = (pool) => `pooled engine · ${pool[0]} is rank 0 and serves the HTTP `
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
  // The title is the reading. Calling a framebuffer budget "host memory" sends
  // an operator to check free -g, which on a discrete box will look fine while
  // this panel is refusing launches.
  const poolLabel = budget.pool_kind === 'discrete' ? 'GPU memory budget' : 'Host memory budget';
  return panel(poolLabel, {
    sub: `this machine · ${budget.tenants.length} engine(s) resident`,
    body: h('div', null,
      h('div', { class: 'grid cols-4' },
        stat('Committed', pct(budget.committed_util, 0),
          `${bytes(budget.committed_bytes)} the resident engines are estimated to hold`),
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
        // `label` is the backend's rendering of a tenant, and it is the only
        // one that works for an engine with no fraction: it prints bytes there
        // and the fraction where there is one, so the strip never shows a blank
        // number beside a container holding 40 GiB.
        budget.tenants.map((tenant) => h('span', { class: 'tag', title: tenant.note || '' },
          tenant.label
          || `${tenant.name} ${tenant.util}${tenant.implicit ? ' (implied)' : ''}`)),
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

/* The Util column, which only one engine has.
   For vLLM an empty value means "the default applies", which is a real
   reservation of over 100 GiB on a unified box and is worth saying. For an
   engine with no fraction at all it means nothing of the kind, and the old
   tooltip named a flag that engine has never had. A byte figure is shown
   instead where the backend could work one out. */
function utilCell(util, engine = DEFAULT_ENGINE, footprintBytes = 0) {
  if (util !== null && util !== undefined) return String(util);
  if (engine !== 'vllm') {
    return footprintBytes
      ? h('span', { class: 'faint', title: `${engineLabel(engine)} declares no memory `
        + 'fraction; this is what its configuration is estimated to take.' },
      bytes(footprintBytes))
      : h('span', { class: 'faint', title: `${engineLabel(engine)} declares no memory `
        + 'fraction, and its footprint could not be estimated from here.' }, '—');
  }
  return h('span', {
    class: 'faint',
    title: 'no --gpu-memory-utilization set, so vLLM applies its own default',
  }, 'default');
}

/* Which engine a row runs, beside the pooled and peer tags it already carries.
   A badge in the name cell rather than a column: the table has six already, and
   this belongs with the other things that say what a row IS. Shown only for the
   non-default engine, so a single-engine box looks exactly as it did. */
const engineTag = (engine) => (engine && engine !== DEFAULT_ENGINE
  ? h('span', { class: 'tag', style: { marginLeft: '6px' }, title: 'inference engine' },
    engineLabel(engine))
  : null);

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
      engineTag(server.engine),
      pool.length
        ? h('span', { class: 'tag', style: { marginLeft: '6px' }, title: poolTitle(pool) },
          poolLabel(pool))
        : (peer ? h('span', { class: 'tag', style: { marginLeft: '6px' } }, server.node) : null)),
    h('div', { class: 's-model truncate', title: server.model }, server.model)),
  h('td', { class: 'num' }, String(server.port)),
  h('td', { class: 'num' }, utilCell(server.util, server.engine, server.footprint_bytes)),
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
        'These were launched outside the dashboard — by hand or by a script. What they hold '
        + 'counts against the budget above, so stopping one is the quickest way to make room. '
        + 'A vLLM container declares a fraction the guard can read; a llama.cpp one declares '
        + 'nothing, so its column is an estimate from its own weights file. Nothing else about '
        + 'them can be changed from here.')),
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
      h('div', { class: 's-name' }, item.name, engineTag(item.engine)),
      h('div', { class: 's-model truncate', title: item.model }, item.model || item.image)),
    h('td', { class: 'num' }, item.port ? String(item.port) : '—'),
    h('td', { class: 'num' }, utilCell(item.util, item.engine, item.footprint_bytes)),
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
  try {
    // The saved definition's own arguments, asked of its own engine — the same
    // question the editor asks, so a selected server and the form that produced
    // it cannot disagree about whether it fits.
    const verdict = await post('/system/budget/check', {
      engine: server.engine || DEFAULT_ENGINE,
      args: server.args || {},
      model: server.model,
      node: server.node,
      server_id: server.id,
    });
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
  const key = [server.id, server.status, server.engine, state.detailTab,
    state.verdict?.level ?? ''].join('|');
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
      server.engine && server.engine !== DEFAULT_ENGINE
        ? badge('info', engineLabel(server.engine)) : null,
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
    ? remoteLaunchNotice(server.node, server.util, server.engine)
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
    `${pool[0]} runs rank 0 and the HTTP frontend, so ${server.url} is the only address `
    + `clients use. ${stagesPhrase(pool)} — no client talks to them directly.`,
    h('div', { style: { marginTop: '4px' } },
      status
        ? (live
          ? `Ranks: ${status.running ? 'all up' : 'incomplete — the engine is down'}`
            + `${joined === null ? '' : ` · ${joined} of ${expected} running`}`
            + (workers.length
              ? ` · ${workers.map((w) => `${w.node} ${w.status}`).join(', ')}`
              : '')
          : `Not running. Starting it brings rank 0 up on ${pool[0]} and a headless rank on ${
            pool.slice(1).join(', ')}; they meet over the cluster link.`)
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
    post('/servers/pool/plan', { nodes: pool, model: server.model, args: server.args || {},
      server_id: server.id })
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
function remoteLaunchNotice(name, util, engine = DEFAULT_ENGINE) {
  const budget = (state.status.budgets || {})[name];
  let asked;
  if (engine !== 'vllm') {
    asked = `${engineLabel(engine)} declares no memory fraction, so how much of that node this `
      + 'takes is decided by the .gguf, --n-gpu-layers and --ctx-size — and it is worked out '
      + 'there, where the file is.';
  } else if (util === null || util === undefined) {
    asked = 'No --gpu-memory-utilization is set, so vLLM applies its own default there.';
  } else {
    asked = `You are asking for ${util} of that node.`;
  }

  if (!budget) {
    return notice('info',
      h('strong', null, `Checked on ${name}, not here. `),
      `GET /api/servers reported no budget for ${name}, so there is nothing to show from it. `
      + `${asked} The memory guard runs on ${name} when the server starts.`);
  }

  const resident = budget.tenants.length
    // `label` is the backend's own rendering of a tenant, and it is the only one
    // that works for both engines: a fraction where there is one, bytes where
    // there is not.
    ? budget.tenants.map((tenant) => tenant.label
      || `${tenant.name} ${tenant.util}`).join(', ')
    : 'nothing resident';

  return notice('info',
    h('strong', null, `Checked on ${name}, not here. `),
    `${name} reports ${pct(budget.free_util, 0)} of its ${pct(budget.max_util, 0)} ceiling free: `
    + `${bytes(budget.available_bytes)} available of ${bytes(budget.total_bytes)}, `
    + `${pct(budget.committed_util, 0)} already committed (${resident}). ${asked}`,
    h('div', { class: 'faint', style: { marginTop: '4px' } },
      'This is that node\'s headroom, not a verdict — the memory guard runs on '
      + `${name} at start time and it is the one that can refuse. Per-process GPU allocation is `
      + 'only readable on the machine running this dashboard, so a peer is measured by what '
      + 'its containers declare or, for an engine that declares nothing, by what its weights '
      + 'and context imply.'));
}

function verdictNotice(verdict) {
  const level = { ok: 'ok', warn: 'warn', block: 'danger' }[verdict.level] || 'info';
  const lead = { ok: 'Fits. ', warn: 'Tight. ', block: 'Blocked. ' }[verdict.level] || '';
  // Two different advices, because the two engines act on different things. A
  // vLLM operator retypes a fraction; a llama.cpp operator has no fraction to
  // retype and needs to know how many bytes are going spare. Rendering
  // suggested_util unguarded printed the word "null" into the advice as soon as
  // a bytes-native verdict came back.
  let advice = null;
  if (verdict.suggested_util !== null && verdict.suggested_util !== undefined) {
    advice = `Largest --gpu-memory-utilization that fits right now: ${verdict.suggested_util}`;
  } else if (verdict.suggested_bytes) {
    advice = `${bytes(verdict.suggested_bytes)} is free for this engine right now — spend it `
      + 'on layers with --n-gpu-layers or on context with --ctx-size.';
  }
  return notice(level,
    h('strong', null, lead),
    verdict.message,
    advice ? h('div', { class: 'faint', style: { marginTop: '4px' } }, advice) : null);
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
    commandBlock(server.engine === 'llamacpp' ? 'llama-server' : 'vllm serve', argv));
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

  const engine = server.engine || DEFAULT_ENGINE;
  const selected = data.selected || {};
  if (!Object.keys(selected).length) {
    mount(host, notice('warn', emptyMetricsReason(engine, server.status)));
    return;
  }
  mount(host, h('div', { class: 'serve-metrics' },
    ...(engine === 'llamacpp' ? llamacppMetrics(selected) : vllmMetrics(selected))));
}

function emptyMetricsReason(engine, status) {
  if (engine === 'llamacpp') {
    return status === 'running'
      ? 'The server is up but published no metrics. The dashboard always launches '
        + 'llama-server with --metrics, so a container started by hand without it will '
        + 'never fill this panel.'
      : 'Loading. llama-server answers /metrics only once the model is in, so there is '
        + 'nothing to scrape yet — this is not an unreachable server.';
  }
  return status === 'running'
    ? 'The server is up but published no vLLM metrics. --disable-log-stats turns them off.'
    : 'Loading. vLLM binds no port until the weights are in and CUDA graphs are captured, so '
      + 'there is nothing to scrape yet — this is not an unreachable server.';
}

function vllmMetrics(selected) {
  const value = (name) => selected[`vllm:${name}`];
  const queries = value('prefix_cache_queries_total');
  const hits = value('prefix_cache_hits_total');
  // 0.24 dropped gpu_prefix_cache_hit_rate on this build; the counters are the
  // ones that are actually there, so derive the rate when the gauge is absent.
  const rate = value('gpu_prefix_cache_hit_rate') ?? (queries ? hits / queries : null);
  const kv = value('kv_cache_usage_perc') ?? value('gpu_cache_usage_perc');

  return [
    stat('Running', count(value('num_requests_running') ?? 0), 'in the current batch'),
    stat('Waiting', count(value('num_requests_waiting') ?? 0), 'queued behind max-num-seqs'),
    stat('KV cache', kv === undefined || kv === null ? '—' : pct(kv, 1), 'of allocated blocks'),
    stat('Prefix hits', rate === null || rate === undefined ? '—' : pct(rate, 1),
      queries ? `${count(hits)} of ${count(queries)} queries` : 'no queries yet'),
    stat('Prompt tokens', count(value('prompt_tokens_total') ?? 0), 'cumulative'),
    stat('Generated', count(value('generation_tokens_total') ?? 0), 'cumulative'),
    stat('Completed', count(value('request_success_total') ?? 0), 'successful requests'),
    stat('Preemptions', count(value('num_preemptions_total') ?? 0), 'KV cache pressure'),
  ];
}

/* Not a rename of the panel above: llama.cpp measures different things.
   There is no KV-usage gauge and no prefix-cache hit rate, and there are two
   throughput gauges vLLM does not publish. Rendering vendor metric names into
   the same eight tiles would have produced four permanent dashes and thrown
   away the two figures that are actually there. */
function llamacppMetrics(selected) {
  const value = (name) => selected[`llamacpp:${name}`];
  const drafted = value('spec_decode_num_draft_tokens_total');
  const accepted = value('spec_decode_num_accepted_tokens_total');

  return [
    stat('Processing', count(value('requests_processing') ?? 0), 'in a slot right now'),
    stat('Deferred', count(value('requests_deferred') ?? 0), 'waiting for a free slot'),
    stat('Busy slots', (value('n_busy_slots_per_decode') ?? 0).toFixed(2),
      'average per decode — near --parallel means saturated'),
    stat('Prompt', `${count(Math.round(value('prompt_tokens_seconds') ?? 0))}/s`,
      `${count(value('prompt_tokens_total') ?? 0)} tokens read`),
    stat('Generation', `${count(Math.round(value('predicted_tokens_seconds') ?? 0))}/s`,
      `${count(value('tokens_predicted_total') ?? 0)} tokens written`),
    stat('Decodes', count(value('n_decode_total') ?? 0), 'llama_decode calls'),
    stat('Longest context', count(value('n_tokens_max') ?? 0), 'tokens seen in one request'),
    stat('Draft accepted', drafted ? pct(accepted / drafted, 1) : '—',
      drafted ? `${count(accepted)} of ${count(drafted)} drafted`
        : 'speculative decoding is off'),
  ];
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
    // Both refusals answer 409 — a 502 for the environment one was rewritten by
    // Cloudflare into its own error page — so `kind` is what tells them apart.
    // Forcing past a missing image or a bound port would only fail again.
    if (error instanceof ApiError && error.status === 409 && error.detail?.message
        && error.detail?.kind !== 'environment') {
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
      // Restart answers 200 with what happened rather than raising, because the
      // stop is real either way and the caller has to know it. But there are
      // two ways the relaunch fails and only one is worth forcing: `error` is
      // the environment refusing it — an unbuilt image, a bound port — which no
      // amount of "start anyway" fixes, and which the memory verdict would
      // otherwise caption with a sentence about a launch that never happened.
      if (result.error) {
        toast(result.error, { level: 'danger', title: 'Stopped, and it did not come back' });
      } else {
        const go = await confirmDialog('Stopped, but the relaunch does not fit',
          result.safety?.message || 'The memory guard refused the relaunch.',
          { confirmLabel: 'Start anyway' });
        if (go) await startServer(id, { force: true });
      }
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
  // Parameterised by engine, and for one kind that is the whole point: vLLM is
  // pointed at a repo and resolves the weights itself, while llama.cpp is
  // pointed at ONE .gguf file — and a GGUF release ships six of them. Fetching
  // the vLLM list for a llama.cpp form would offer repo ids to a field that
  // takes a filename.
  const engine = state.editor?.form.engine || DEFAULT_ENGINE;
  try {
    payload = await get(`/servers/paths?engine=${encodeURIComponent(engine)}`);
  } catch (error) {
    // A failed scan costs the operator nothing but the list: an empty kind
    // falls through to its text box, so the form stays usable.
    toast(error.message, { level: 'warn', title: 'Could not list what is on disk' });
    return;
  }
  // The engine may have changed while this was in flight; a list of the other
  // engine's models is worse than none.
  if (!state.editor || state.editor.form.engine !== engine) return;
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
  if (MEMORY_DESTS[editor.form.engine]?.has(arg.dest)) {
    scheduleSafety();
    // A pooled definition has no single-node verdict; its answer comes from
    // the plan, so the plan is what has to be re-asked.
    if (state.editor?.form.pooled) schedulePlan();
  }
}

/* The flags that change what a launch will cost, and therefore the ones whose
   editing has to re-ask the memory guard. vLLM has exactly one — the fraction
   IS the reservation. llama.cpp has no fraction at all, so its footprint moves
   with the layer count, the context, the cache dtypes and the batch. Leaving
   this keyed on one vLLM dest would mean nothing on screen changed as a
   llama.cpp operator moved the two numbers that actually decide it. */
const MEMORY_DESTS = {
  vllm: new Set(['gpu_memory_utilization', 'kv_cache_memory_bytes', 'cpu_offload_gb']),
  llamacpp: new Set(['n_gpu_layers', 'ctx_size', 'cache_type_k', 'cache_type_v',
    'ubatch_size', 'flash_attn', 'parallel', 'cpu_moe', 'n_cpu_moe']),
};

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

function applyFilter(query, generation) {
  // The debounce outlives the form it filters: saving or cancelling within the
  // keystroke window tears the editor down before this runs, and an engine
  // switch rebuilds every registry it walks — so a stale callback would hide
  // and show nodes belonging to a form that is no longer on screen, against a
  // query the (new, empty) search box does not show.
  if (!state.editor) return;
  if (generation !== undefined && generation !== state.editor.generation) return;
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
 *  become `qwen3-embedding-8b`.
 *
 *  Once a model could be a FILE rather than a repo, the tail became a filename —
 *  so `.../Qwen3-8B-Q4_K_M.gguf` would derive `qwen3-8b-q4-k-m-gguf`. That is
 *  not cosmetic: the served name becomes the engine's --alias, which is the
 *  exact string every OpenAI client has to put in its `model` field. The suffix
 *  and the quantisation tag are stripped for that reason. */
function slugFromModel(model) {
  const trimmed = String(model || '').trim().replace(/\/+$/, '');
  if (!trimmed) return '';
  let tail = trimmed.split('/').filter(Boolean).pop() || '';
  tail = tail.replace(/\.gguf$/i, '')
    // A shard suffix, then a quantisation tag. Both describe the file rather
    // than the model, and a client should not have to type either.
    .replace(/-0{0,4}\d+-of-0{0,4}\d+$/i, '')
    .replace(/[.-](?:[IQ]Q?\d(?:_[A-Za-z0-9]+)*|[QF]\d+(?:_[A-Za-z0-9]+)*|BF16|F16|F32|MXFP4)$/i, '');
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
  // The engine key is carried through rather than rebuilt. It records whether
  // the engine is still ours to derive, and it is written by maybeAdoptEngine
  // and by applyEngine — assigning a fresh object here would forget a manual
  // choice on the very next model change.
  editor.derived = {
    ...(previous.engine === undefined ? {} : { engine: previous.engine }),
    name: editor.form.name,
    served: editor.form.served_name,
  };
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
  // The engine is part of the key, not just the body: without it the
  // short-circuit below would serve the previous engine's recommendation after
  // a switch, and suppress the re-fetch that would have corrected it.
  const key = `${editor.form.engine}\u0000${node}\u0000${model}\u0000`
    + `${editor.form.pooled ? editor.form.pool.join(',') : ''}`;
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
    const rec = await post('/servers/recommend', {
      model, node, engine: editor.form.engine, args: { ...editor.args },
      pool: editor.form.pooled ? editor.form.pool : [],
      // So an existing server's own container is not counted against itself.
      server_id: editor.id,
      // Some of what decides whether an engine starts is an environment
      // variable rather than a flag, so what is already in that box is part of
      // the question too.
      env: parseEnv(editor.form.env || ''),
    });
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
 *  Going through the ACTIVE engine's flag index is not optional: the backend refuses a dest
 *  this image does not have, so a flag that would 422 on save is skipped here
 *  with a word about it instead. */
function applyArgs(args) {
  const missing = [];
  let applied = 0;
  for (const [dest, value] of Object.entries(args || {})) {
    const arg = indexOf().get(dest);
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

/** Merge {NAME: value} into the Environment box, leaving everything else alone.
 *  Rewriting the whole textarea would drop the operator's own variables and any
 *  comments in it, so each name is replaced in place and only a genuinely new
 *  one is appended. */
function applyEnv(vars) {
  const entries = Object.entries(vars || {});
  if (!entries.length) return 0;
  const lines = String(state.editor.form.env || '').split('\n');
  let applied = 0;
  for (const [name, value] of entries) {
    const line = `${name}=${value}`;
    const at = lines.findIndex((text) => text.trim().split('=')[0].trim() === name);
    if (at >= 0) {
      if (lines[at] === line) continue;
      lines[at] = line;
    } else {
      lines.push(line);
    }
    applied += 1;
  }
  state.editor.form.env = lines.filter((text) => text.trim()).join('\n');
  if (state.editor.refs.env) state.editor.refs.env.value = state.editor.form.env;
  return applied;
}

function applyRecommendation() {
  const rec = state.editor?.rec;
  if (!rec) return;
  const { applied, missing } = applyArgs(rec.args);
  const envApplied = applyEnv(rec.env);
  if (missing.length) {
    toast(`This ${engineLabel(currentEngine())} build has no ${missing.join(', ')}; `
      + 'the rest was applied.', { level: 'warn' });
  }
  const parts = [];
  if (applied) parts.push(`${applied} flag${applied === 1 ? '' : 's'}`);
  if (envApplied) parts.push(`${envApplied} environment variable${envApplied === 1 ? '' : 's'}`);
  toast(parts.length ? `Applied ${parts.join(' and ')}.` : 'Nothing to change.', { level: 'ok' });
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
  const currentEnv = parseEnv(editor.form.env || '');
  const pendingEnv = Object.entries(rec.env || {})
    .filter(([name, value]) => currentEnv[name] !== value);
  const outstanding = pending.length + pendingEnv.length;

  // A refusal that names the engine which WOULD serve this model is one click
  // from a working configuration, and leaving the operator to find the dropdown
  // themselves is the difference between a dead end and a next step. It is
  // offered only where the backend was certain — GGUF weights and nothing else,
  // or safetensors and nothing else — never inferred from a name.
  // Offered only when the backend was certain, and `ok` is how it says so: the
  // llama.cpp advisor sets engine_hint='vllm' whenever it cannot open the file,
  // which is the ordinary state of a repo that has not been pulled yet — and a
  // primary "Switch to vLLM" button on a model that is still downloading is
  // advice pointing the wrong way.
  const hint = !rec.ok && rec.engine_hint && rec.engine_hint !== editor.form.engine
    ? rec.engine_hint : '';

  mount(host, h('div', { class: `serve-rec lv-${recLevelClass(rec.level)}` },
    h('div', { class: 'sr-head' },
      h('strong', null, rec.headline),
      h('span', { class: 'spacer' }),
      hint
        ? h('button', { class: 'btn-primary btn-sm', onClick: () => applyEngine(hint) },
          `Switch to ${engineLabel(hint)}`)
        : null,
      rec.ok && outstanding
        ? h('button', { class: 'btn-primary btn-sm', onClick: applyRecommendation },
          `Apply ${outstanding} setting${outstanding === 1 ? '' : 's'}`)
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
    // Environment variables read as what you type into the box below, not as
    // flags, because that is where they have to go — they are not in the schema
    // the flag form is generated from.
    (rec.env_suggestions || []).length
      ? h('div', { class: 'sr-list' }, rec.env_suggestions.map((e) => {
        const already = currentEnv[e.name] === e.value;
        return h('div', { class: `sr-item${already ? ' done' : ''}` },
          h('code', null, `${e.name}=${e.value}`),
          already ? badge('succeeded', 'set') : badge('info', 'env'),
          h('span', { class: 'sr-why' }, e.why));
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

const flagOf = (dest) => indexOf().get(dest)?.flag || `--${dest.replace(/_/g, '-')}`;

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
  const llamacpp = currentEngine() === 'llamacpp';
  // Every one of these except the GGUF badge is a question about what THIS
  // vLLM image can construct, and none of them is the question a llama.cpp
  // server is asking. `supported` in particular reads the vLLM architecture
  // registry, so under llama.cpp it would report a red failure about a build
  // that is not the one running the model.
  if (!llamacpp && profile.supported === false) {
    flags.push(badge('failed', 'architecture not in this image'));
  }
  if (!llamacpp && profile.custom_sampler) flags.push(badge('starting', 'one machine only'));
  if (!llamacpp && profile.runner === 'pooling') {
    flags.push(badge('starting', 'embeddings, not chat'));
  }
  if (profile.is_multimodal) flags.push(badge('info', 'multimodal'));
  if (!llamacpp && profile.requires_remote_code) {
    flags.push(badge('starting', 'needs trust-remote-code'));
  }
  if (profile.is_adapter) flags.push(badge('failed', 'LoRA adapter, not a model'));
  // The single most inverted line on this page before there were two engines.
  // GGUF-only is a hard failure for vLLM and the success condition for
  // llama.cpp — the same fact, and opposite verdicts.
  if (profile.has_gguf && !profile.has_safetensors) {
    flags.push(llamacpp ? badge('running', 'GGUF') : badge('failed', 'GGUF only'));
  }
  if (llamacpp && profile.has_safetensors && !profile.has_gguf) {
    flags.push(badge('failed', 'safetensors — llama.cpp needs GGUF'));
  }
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
  // Both of these describe what THIS vLLM image will do with the repo — one
  // reads its architecture registry, the other its runner resolution — and
  // llama.cpp decides neither the same way. Under it they are simply not
  // findings, rather than findings with different wording.
  const vllm = currentEngine() === 'vllm';
  const pooling = vllm && profile.runner === 'pooling';
  const unsupported = vllm && profile.supported === false;
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
  // An existing definition keeps the engine it was saved with, always. Its model
  // reference may say 'gguf' or may not, and either way the operator already
  // made this decision once — re-deciding it on open would flip a working server
  // under them.
  const engine = server?.engine || DEFAULT_ENGINE;
  state.editor = {
    id: server?.id ?? null,
    // A LIVE POINTER into argsByEngine, not a copy. Every reader either takes it
    // at control-build time or goes through state.editor.args, so swapping the
    // pointer and rebuilding the form is all an engine change needs.
    args: { ...(server?.args || {}) },
    // What was typed for the engines that are not selected. Seeded from the
    // row's own stash so a definition edited, switched, saved and reopened comes
    // back with both sets intact.
    argsByEngine: {
      ...(server?.args_by_engine || {}),
      [engine]: { ...(server?.args || {}) },
    },
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
    // What the currently selected model derived — name, served name, and now the
    // engine. A field still holding this is fair game to refill; anything else
    // is the user's and is left alone.
    derived: null,
    // The machines a pooled definition named, held while an engine that cannot
    // pool is selected so switching back restores them rather than silently
    // converting a three-machine server into a single-node one.
    stashedPool: pool.length > 1 ? [...pool] : [],
    // Which render the form on screen belongs to; see applyFilter.
    generation: 0,
    // The question the outstanding budget check was asked, so a slower answer
    // for a previous engine cannot land after a newer one.
    safetyFor: '',
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
      engine,
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

  // A new definition arriving with a model already chosen — the Models page's
  // Serve button, or a deep link — settles its engine BEFORE anything is
  // fetched or drawn. Doing it after would paint the form from one engine's
  // schema and then rebuild it from another's while the operator watched.
  if (!server && state.editor.form.model) {
    const wanted = engineForModel(state.editor.form.model);
    if (wanted && wanted !== state.editor.form.engine) state.editor.form.engine = wanted;
  }
  if (!state.schemas[state.editor.form.engine]) {
    const loaded = await schemaFor(state.editor.form.engine);
    if (state.mode !== 'edit' || !state.editor) return;
    if (!loaded) {
      // A saved server keeps the engine it was saved with, always. Falling back
      // here would turn a transient failure on GET /servers/schema into a
      // llama.cpp definition silently rewritten to vLLM on the next Save —
      // with its flags dropped as unknown. Only a NEW definition, whose engine
      // was merely inferred from a model name, falls back.
      if (server) {
        closeEditor();
        toast(`${engineLabel(state.editor?.form.engine || '')} is not available right now, `
          + 'so this server cannot be edited without changing what it is.',
        { level: 'danger', title: 'Engine unavailable' });
        return;
      }
      state.editor.form.engine = DEFAULT_ENGINE;
    }
  }
  state.editor.args = argsFor(state.editor.form.engine);

  await Promise.all([loadPaths(), loadCluster(), server ? null : suggest()]);
  if (state.mode !== 'edit' || state.stopped) return;
  // The names derive from the model, and are only ever refilled while they still
  // hold what a previous model put there.
  if (!server && state.editor.form.model) autoName(state.editor.form.model);
  syncEnginePlacement();
  renderEditor();
  scheduleSafety();
  if (state.editor.form.pooled) runPlan();
}

/** The engine a model reference can only be served by, or '' for "no opinion".
 *
 *  GGUF is the one unambiguous signal available before anything is read off
 *  disk: vLLM cannot load the format at all, so a reference naming it has
 *  exactly one answer. The reverse is deliberately NOT inferred — a repo id
 *  with no 'gguf' in it may still be a GGUF repo, and guessing vLLM for it
 *  would fight an operator who has just chosen llama.cpp on purpose. */
function engineForModel(model) {
  const text = String(model || '').trim();
  if (!text) return '';
  return GGUF_HINT.test(text) ? 'llamacpp' : '';
}

/** The argument set belonging to one engine, created on first use.
 *  Returns the stored object itself so `editor.args` stays a live pointer. */
function argsFor(engine) {
  const editor = state.editor;
  if (!editor.argsByEngine[engine]) editor.argsByEngine[engine] = {};
  return editor.argsByEngine[engine];
}

async function suggest() {
  try {
    const engine = state.editor?.form.engine || DEFAULT_ENGINE;
    const suggestion = await get(`/servers/suggest?engine=${encodeURIComponent(engine)}`);
    if (!state.editor) return;
    state.editor.form.port = state.editor.form.port || suggestion.port;
    state.editor.form.image = state.editor.form.image || suggestion.image;
  } catch (error) {
    console.error('port suggestion failed', error);
  }
}

/** Switch the editor to another engine.
 *
 *  Deliberately mirrors applyPreset's tail — invalidate the profile, re-check
 *  the memory verdict, re-plan if pooled — and adds the four things only an
 *  engine change needs: the schema, the argument pointer, the registries the
 *  form appends to, and a full redraw. */
async function applyEngine(next, { manual = true } = {}) {
  const editor = state.editor;
  if (!editor || editor.form.engine === next) return;
  if (!await schemaFor(next)) {
    // Put the select back where it was: the engine could not be loaded, so the
    // form must not claim to be showing it.
    renderEditor();
    return;
  }
  if (state.mode !== 'edit' || !state.editor) return;

  editor.form.engine = next;
  editor.args = argsFor(next);
  // The suggested image belongs to the engine, and suggest() only fills an empty
  // box — so an image left at another engine's default has to be cleared for the
  // new suggestion to land. An image the operator typed is theirs and stays.
  const previousDefault = state.engines.find((e) => e.image === editor.form.image);
  if (!editor.form.image || previousDefault) {
    editor.form.image = engineInfo(next).image || '';
  }
  syncEnginePlacement();
  renderEditor();
  if (manual) {
    // Once chosen by hand, the choice stops being derivable — see autoName's
    // `ours` test, which this marks as no longer ours.
    editor.derived = { ...(editor.derived || {}), engine: null };
  }
  editor.profileFor = '';
  loadProfile();
  scheduleSafety();
  if (editor.form.pooled) schedulePlan();

  // What is on disk is a different list for each engine — a repo id for vLLM,
  // a .gguf file for llama.cpp — so the pickers are refilled, then re-synced.
  // Deliberately last and unawaited: the form is already correct without it,
  // and a slow cache scan must not hold the redraw.
  loadPaths().then(() => {
    if (state.mode !== 'edit' || !state.editor || state.editor.form.engine !== next) return;
    for (const sync of state.editor.pathViews) sync();
  });
}

/** Pooling is one engine's feature, so selecting the other has to actually turn
 *  it off — not merely hide the control. `save()` writes pool_nodes from
 *  form.pooled and renderFootActions gates Save & start on a pool plan, so a
 *  hidden-but-still-true flag would post a pooled llama.cpp definition and
 *  disable the button that could fix it. */
function syncEnginePlacement() {
  const editor = state.editor;
  if (!editor) return;
  if (enginePools(editor.form.engine)) {
    // Coming back to an engine that pools: restore the machines the definition
    // named. Without this a vLLM -> llama.cpp -> vLLM round trip silently
    // converted a three-machine pooled server into a single-node one, and the
    // only sign was that Placement had reset itself.
    if (!editor.form.pooled && editor.stashedPool?.length > 1) {
      editor.form.pool = [...editor.stashedPool];
      editor.form.pooled = true;
    }
    return;
  }
  if (editor.form.pooled) {
    editor.stashedPool = [...editor.form.pool];
    editor.form.pooled = false;
    editor.form.pool = [];
    editor.plan = null;
  }
}

/* Whether this engine can be split across machines.
   Defaults to FALSE for a name the registry does not carry, which is the safe
   direction: GET /servers/engines is fetched with a .catch, so a failed
   request leaves only vLLM in the list — and treating the unknown as capable
   would offer Placement for a llama.cpp server and let save() post pool_nodes
   the API then rejects. vLLM is named explicitly because it is the one engine
   that is always present. */
const enginePools = (name) => (name === DEFAULT_ENGINE
  ? true : engineInfo(name).supports_pooling === true);

function closeEditor() {
  closeSync();
  state.editor = null;
  state.detailKey = '';
  renderList();
}

function renderEditor() {
  const editor = state.editor;
  const schema = schemaOf(editor.form.engine);
  if (!schema) return;

  // Cleared HERE rather than in openEditor, and that is the point: these five
  // registries are append-only — paramField, diskPicker and pathListControl all
  // push into them — so a second render without a reset leaves setArg toggling
  // the changed-highlight on detached elements and applyFilter walking two
  // engines' worth of sections, hiding nodes that are no longer in the document.
  // Making a clean registry a precondition of every render means no future
  // caller has to remember. For the single-engine path this is a no-op:
  // openEditor only ever called renderEditor once.
  editor.fields = new Map();
  editor.setters = new Map();
  editor.sections = [];
  editor.searchable = [];
  editor.pathViews = [];
  // Bumped on every render, so a debounced callback queued against the previous
  // form can tell that it is no longer the form on screen.
  editor.generation = (editor.generation || 0) + 1;

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

  const flagCount = [...schema.featured, ...schema.advanced]
    .reduce((total, section) => total + section.flags.length, 0);

  // The field that gets typed most, so the one that most wants a list. Refresh
  // lives here rather than on all seventeen pickers: one scan feeds them all.
  const [modelPicker] = diskPicker('model', {
    value: editor.form.model,
    lead: 'choose a model…',
    placeholder: 'org/repo, or a path the container can see',
    onChange: (next) => {
      editor.form.model = next;
      // Before autoName, so the derived name is computed with the engine already
      // settled — and only for a new definition, because an existing server's
      // engine is a decision its operator already made.
      maybeAdoptEngine(next);
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

  // Second in the grid, and everything after it is downstream of it: what
  // Placement may offer, which image is suggested, which flags the form below
  // is built from, and which presets are measurements of anything.
  state.nodes.engineField = field('Engine', enginePicker(), {
    help: engineHelp(editor.form.engine),
  });
  state.nodes.placementField = field('Placement', placementControl(), {
    help: 'One engine on one machine, or one engine split by layer across several so their '
      + 'memory adds up. Pooling costs a network hop per token per stage boundary and a fixed '
      + 'world size; a model that fits on one box should stay on one box.',
  });

  const basics = h('div', { class: 'param-grid' },
    field('Name', input('name', { placeholder: 'qwen3-chat' }), {
      help: 'Names the container and identifies the server everywhere in this dashboard.',
    }),
    state.nodes.engineField,
    state.nodes.placementField,
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
    // Kept in refs so the recommendation can write into it: some of what
    // decides whether an engine starts is read from the environment and is not
    // a flag at all, so Apply has to be able to reach this box.
    field('Environment', (editor.refs.env = h('textarea', {
      rows: 3,
      placeholder: 'KEY=value, one per line',
      value: editor.form.env,
      onInput: (e) => { editor.form.env = e.target.value; },
    })), {
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
    onInput: (() => {
      // The generation is captured now, when the box is built, not read when
      // the debounce fires.
      const generation = editor.generation;
      return debounce((event) => applyFilter(event.target.value, generation), 160);
    })(),
  });

  mount(state.main,
    panel(editor.id ? `Edit ${editor.form.name}` : 'New server', {
      sub: `${schema.image} · ${schema.label || engineLabel(editor.form.engine)} `
        + `${schema.version || schema.vllm_version || ''}`.trim(),
      actions: h('button', { onClick: closeEditor }, 'Cancel'),
      body: h('div', { class: 'serve-form' },
        basics,
        state.nodes.profileBox,
        state.nodes.recBox,
        state.nodes.poolBox,
        presetSection(editor.form.engine),
        h('div', { class: 'param-search' }, search),
        schema.featured.map((section) => paramSection(section)),
        h('div', { class: 'param-section' },
          h('h3', null, 'All other parameters'),
          h('p', { class: 'blurb' },
            'Every remaining flag this image accepts, grouped the way vLLM groups them. '
            + `${schema.managed.join(', ')} are set by the dashboard and are not listed.`),
          schema.advanced.map((section) => paramSection(section, { collapsed: true })))),
    }),
    h('div', { class: 'serve-foot' }, state.nodes.safety, state.nodes.footActions));

  renderPoolBox();
  syncPlacement();
  renderFootActions();
  renderProfile();
  renderRecommendation();
  loadProfile();
}

/* --- the engine ---------------------------------------------------------- */

/** Which inference engine runs this definition.
 *
 *  A plain select rather than a segmented control, because the list comes from
 *  the backend and is not fixed at two: an engine this build does not have
 *  simply is not in it. A saved server whose engine has since been removed keeps
 *  its own name here rather than silently becoming something else, the same way
 *  the node picker treats a de-registered peer. */
function enginePicker() {
  const editor = state.editor;
  const choices = state.engines.length
    ? state.engines
    : [{ name: DEFAULT_ENGINE, label: engineLabel(DEFAULT_ENGINE) }];

  const select = h('select', {
    onChange: (event) => applyEngine(event.target.value),
  },
  choices.map((engine) => h('option', {
    value: engine.name,
    selected: engine.name === editor.form.engine,
  }, engine.version && engine.version !== 'unknown'
    ? `${engine.label} ${engine.version}`
    : engine.label)),
  choices.some((engine) => engine.name === editor.form.engine)
    ? null
    : h('option', { value: editor.form.engine, selected: true },
      `${editor.form.engine} — not available in this build`));

  return select;
}

function engineHelp(engine) {
  if (engine === 'llamacpp') {
    return 'llama.cpp serves GGUF, one file at a time, and can run a model larger than the '
      + 'accelerator by leaving some layers on the CPU. It declares no memory fraction — '
      + 'what it takes is worked out from the file and --ctx-size. Every flag below comes '
      + "from llama-server's own help, and pooling across machines is vLLM's alone.";
  }
  return 'vLLM serves safetensors and cannot read GGUF. It reserves a fraction of the '
    + 'node\'s memory up front — that fraction is the memory decision — and it is the '
    + 'engine that can be pooled across machines. Choosing a .gguf model switches this '
    + 'to llama.cpp for you.';
}

function presetSection(engine) {
  const presets = presetsFor(engine);
  if (!presets.length) return null;
  const blurb = engine === 'vllm'
    ? 'Two of these are configurations that have actually run on this box; the third is '
      + 'derived from them. Applying one sets only the flags it names.'
    : 'Starting points rather than measurements: what a llama.cpp launch costs depends on '
      + 'the .gguf, so these carry the intent and the memory verdict below carries the '
      + 'number. Applying one sets only the flags it names.';
  return h('div', { class: 'param-section' },
    h('h3', null, 'Presets'),
    h('p', { class: 'blurb' }, blurb),
    h('div', { class: 'serve-presets' }, presets.map((preset) => h('button', {
      class: 'serve-preset',
      onClick: () => applyPreset(preset),
    }, h('b', null, preset.title), h('span', null, preset.note)))));
}

/** Adopt the engine a newly chosen model implies — while the choice is still
 *  ours to make.
 *
 *  The test is the one autoName already uses for the derived name: a field is
 *  still ours while it holds what the PREVIOUS model put there. That forgives,
 *  where a plain "the user touched it once" flag would not — set it back by
 *  hand and the form starts deriving again. An existing server never enters
 *  here at all. */
function maybeAdoptEngine(model) {
  const editor = state.editor;
  if (!editor || editor.id !== null) return;
  const wanted = engineForModel(model);
  if (!wanted || wanted === editor.form.engine) return;

  const previous = editor.derived || {};
  // `engine: null` is what applyEngine writes when the operator chooses by
  // hand, and it can never equal a real engine name — so a manual pick stays.
  const ours = !('engine' in previous) || previous.engine === editor.form.engine;
  if (!ours) return;

  applyEngine(wanted, { manual: false }).then(() => {
    if (state.mode !== 'edit' || !state.editor) return;
    // The first switch of a session awaits a real schema fetch, and the model
    // field is editable throughout. If it has moved on to something this engine
    // is not the answer for, the adoption is stale and saying so would be wrong.
    if (engineForModel(state.editor.form.model) !== wanted) return;
    state.editor.derived = { ...(state.editor.derived || {}), engine: wanted };
    toast(`${model.split('/').pop()} is GGUF, so this is a llama.cpp server. `
      + 'Change the Engine field if that is not what you want.', { level: 'info' });
  });
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
  // Nothing may turn pooling on for an engine that has none — not a stale
  // handler, not a queued debounce. The control is hidden, but the guard is
  // what makes `form.pooled` trustworthy to save() and renderFootActions.
  if (pooled && !enginePools(editor.form.engine)) return;
  if (editor.form.pooled === pooled) return;
  editor.form.pooled = pooled;
  if (pooled && editor.form.pool.length < 2) editor.form.pool = defaultPool();
  if (!pooled) editor.plan = null;
  syncPlacement();
  renderPoolBox();
  renderFootActions();
  // Placement is part of what the recommendation answers — whether spreading
  // this model across machines buys anything depends on how many there are.
  scheduleProfile();
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
  const pools = enginePools(state.editor.form.engine);
  const pooled = pools && state.editor.form.pooled;
  for (const [index, button] of (state.nodes.segButtons || []).entries()) {
    button.setAttribute('aria-pressed', String(pooled === (index === 1)));
  }
  // Hidden rather than disabled when the engine cannot pool at all: a greyed
  // two-way control with one reachable side is a question that was never a
  // question. The Node picker stays — a llama.cpp server still runs somewhere.
  if (state.nodes.placementField) state.nodes.placementField.hidden = !pools;
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
  scheduleProfile();
  runPlan();
}

function addPoolNode(name) {
  if (!name || state.editor.form.pool.includes(name)) return;
  state.editor.form.pool.push(name);
  renderPoolBox();
  scheduleProfile();
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
      + 'has a single GPU. The order below is the pipeline order: the first machine runs rank 0 '
      + 'and the HTTP frontend, so its address is the one clients use, and the rest run headless, '
      + 'holding later stages and serving nothing directly.'),
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
    // server_id so an existing pooled engine's own ranks come off the budget
    // rather than being counted against its own restart on every node.
    plan = await post('/servers/pool/plan', { nodes: pool, model, args,
      server_id: state.editor.id });
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

/* Nothing here can fix this one: there is no pull endpoint for a peer's daemon,
   so the honest answer is the command that pulls it. */
function missingImageNotice(names) {
  return notice('danger',
    h('strong', null, `The vLLM image is missing on ${names.join(', ')}. `),
    'Every rank loads the model itself, so each machine in the pool needs the image locally. '
    + 'Pull it on each of those machines and re-plan:',
    h('div', { class: 'cmdbox', style: { marginTop: '6px' } },
      `docker pull ${schemaOf()?.image || 'the engine image'}`));
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
    toast(`This ${engineLabel(currentEngine())} build has no ${missing.join(', ')}; `
      + 'the rest of the preset was applied.', { level: 'warn' });
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
      remoteLaunchNotice(node, state.editor.args.gpu_memory_utilization,
        state.editor.form.engine));
    renderFootActions();
    return;
  }
  state.editor.budgetFor = null;
  // Keyed against the question it asked, the way loadProfile keys on profileFor
  // and runPlan on planKey. Without it a slower verdict for the previous engine
  // or node lands after a newer one and drives the Save & start gate.
  const asked = `${state.editor.form.engine}\u0000${state.editor.form.node}`
    + `\u0000${state.editor.form.model}\u0000${JSON.stringify(state.editor.args)}`;
  state.editor.safetyFor = asked;
  // The whole argument set, not one number: llama.cpp has no fraction to send,
  // and even vLLM's fraction understates a config that sets --kv-cache-memory.
  let verdict;
  try {
    verdict = await post('/system/budget/check', {
      engine: state.editor.form.engine,
      args: { ...state.editor.args },
      // Separately, because it is a managed flag and never lives in args — and
      // an engine priced from its weights file cannot be sized without it.
      model: state.editor.form.model,
      node: state.editor.form.node,
      // So an existing server's own container is not counted against itself.
      server_id: state.editor.id,
    });
  } catch (error) {
    console.error('budget check failed', error);
    return;
  }
  if (state.mode !== 'edit' || !state.editor || state.editor.safetyFor !== asked) return;
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
  // make. A plan that says no is not: the image or the weights are missing on
  // a node, and starting anyway just fails slowly on another machine.
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
    engine: form.engine,
    node: form.node || LOCAL,
    // More than one name is what makes it pooled, and the first is the head.
    // The backend ignores `node` for placement once this is set.
    pool_nodes: form.pooled ? [...form.pool] : [],
    model: form.model.trim(),
    port,
    served_name: form.served_name.trim(),
    image: form.image.trim() || null,
    args: { ...editor.args },
    // What the other engines hold, so switching back after a save finds it
    // still there. `args` above stays the authoritative set for the engine
    // actually selected; this is only the editor's memory of the rest.
    args_by_engine: { ...editor.argsByEngine, [form.engine]: { ...editor.args } },
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
