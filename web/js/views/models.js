/* Models: what is already in the shared HuggingFace cache, and pulling more.

   The cache is root-owned, so every write happens inside a container the
   backend drives as a job. That is why a pull here is a job id plus a live
   stream rather than a request that returns once the bytes have landed, and
   why several pulls can be in flight at once. */

import {
  ago, badge, bytes, confirmDialog, copyButton, count, debounce, empty, ensureStyles, h,
  logBox, modal, mount, notice, panel, spinner, stat, toast, when,
} from '../ui.js';
import { ApiError, del, get, post, put, stream } from '../api.js';

const CSS = `
/* One tab per node. A tab is a filter over which machine every action on this
   page talks to, so it sits at the top of the panel it governs. */
.node-tabs { display: flex; gap: 6px; margin-bottom: 14px; flex-wrap: wrap; }
.node-tab {
  font-family: var(--mono); font-size: 12px; padding: 5px 12px;
  border: 1px solid var(--border); border-radius: 999px;
  background: var(--bg-sunken); color: var(--text-dim); cursor: pointer;
}
.node-tab:hover { border-color: var(--border-strong); color: var(--text); }
.node-tab.active {
  background: var(--accent-dim); border-color: var(--accent); color: var(--text);
}
.node-tab.down { opacity: .6; }
.node-tab .dot-down { color: var(--danger); margin-left: 5px; font-weight: 700; }

.models-filters { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.models-filters input[type="search"] { flex: 1; min-width: 200px; }
.models-filters select { width: auto; }

.models-dl { border-bottom: 1px solid var(--border); padding: 11px 13px; }
.models-dl:last-child { border-bottom: 0; }
.models-dl .dl-head { display: flex; align-items: center; gap: 9px; margin-bottom: 7px; }
.models-dl .dl-id { font-family: var(--mono); font-size: 13px; font-weight: 560; }
.models-dl .dl-nums {
  display: flex; flex-wrap: wrap; gap: 12px; margin-top: 6px;
  font-family: var(--mono); font-size: 11px; color: var(--text-dim);
}
.models-dl .dl-nums b { color: var(--text); font-weight: 600; }
.models-dl .dl-transient {
  font-family: var(--mono); font-size: 10.5px; color: var(--text-faint);
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; margin-top: 5px;
}
.models-dl .logbox { max-height: 220px; margin-top: 8px; }

.models-kv { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 5px 16px; font-size: 12.5px; }
.models-kv dt { color: var(--text-faint); }
.models-kv dd { margin: 0; font-family: var(--mono); font-size: 12px; overflow-wrap: anywhere; }
.models-files { max-height: 260px; overflow: auto; border: 1px solid var(--border); border-radius: var(--radius-s); }
.models-tags { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 6px; }
`;

// The Hub accepts far more, but these are the tasks this box actually serves.
const PIPELINES = [
  ['', 'Any task'],
  ['text-generation', 'text-generation'],
  ['image-text-to-text', 'image-text-to-text'],
  ['text2text-generation', 'text2text-generation'],
  ['feature-extraction', 'feature-extraction'],
  ['sentence-similarity', 'sentence-similarity'],
  ['automatic-speech-recognition', 'automatic-speech-recognition'],
];
const SORTS = [
  ['downloads', 'Most downloaded'],
  ['likes', 'Most liked'],
  ['lastModified', 'Recently updated'],
  ['trending', 'Trending'],
];
const GATED = [['', 'Gated or not'], ['false', 'Ungated only'], ['true', 'Gated only']];

const MARKERS = ['@@RESULT@@', '@@PROGRESS@@'];
const LEVEL_CLASS = { ok: 'ok', warn: 'warn', block: 'danger' };

const state = {
  ctx: null,
  local: null,
  token: { configured: false, source: '' },
  query: { q: '', pipeline_tag: '', sort: 'downloads', gated: '' },
  results: [],
  downloads: new Map(),
  nodes: {},
  searchSeq: 0,
  /* Which machine's cache this page is managing. '' is this one; every cache
     call carries it, so switching the tab switches what a delete deletes and
     what a pull lands on. Search results are Hub-side and never node-specific,
     but "cached" is — so the results list restates itself per node too. */
  node: '',
  registry: [],
};

const nodeLabel = () => (state.node || 'this machine');

/* A repo id is a path segment pair; the routes take it as `{repo_id:path}`, so
   the slash has to survive encoding. */
const repoPath = (repoId) => repoId.split('/').map(encodeURIComponent).join('/');

const speed = (value) => (Number.isFinite(value) && value > 0 ? `${bytes(value)}/s` : '—');

function markerPayload(line) {
  for (const marker of MARKERS) {
    const at = line.indexOf(marker);
    if (at === -1) continue;
    try {
      const payload = JSON.parse(line.slice(at + marker.length));
      return { payload, final: marker === MARKERS[0] };
    } catch {
      return null;
    }
  }
  return null;
}

/** Disk on whichever node this page is managing. A peer reports its own with
 *  the cache scan; this machine's comes from the telemetry stream. */
function diskOf() {
  if (state.node) return state.local?.disk || {};
  return state.ctx?.telemetry()?.disk || {};
}

function freeDisk() {
  return diskOf().free_bytes ?? null;
}

/* --- local cache -------------------------------------------------------- */

async function loadLocal() {
  const wanted = state.node;
  try {
    const query = wanted ? `?node=${encodeURIComponent(wanted)}` : '';
    const payload = await get(`/hub/local${query}`);
    if (wanted !== state.node) return; // the tab moved while this was in flight
    state.local = payload;
  } catch (error) {
    if (wanted !== state.node) return;
    state.local = { ok: false, error: error.message, repos: [], cache_dir: '' };
  }
  renderLocal();
  renderResults();
}

async function loadRegistry() {
  try {
    const payload = await get('/nodes');
    state.registry = payload.nodes || [];
  } catch {
    state.registry = [];
  }
  renderNodeTabs();
}

function selectNode(name) {
  if (state.node === name) return;
  state.node = name;
  state.local = null;
  renderNodeTabs();
  renderLocal();
  loadLocal();
}

/** One tab per registered node. A node the dashboard cannot reach is still
 *  listed and still clickable — the error belongs in the panel, where it can
 *  say what failed, not in a tab that silently disappears. */
function renderNodeTabs() {
  const host = state.nodes.tabs;
  if (!host) return;
  if (state.registry.length < 2) return mount(host);
  mount(host, h('div', { class: 'node-tabs', role: 'tablist' },
    state.registry.map((node) => {
      const name = node.local ? '' : node.name;
      const active = name === state.node;
      return h('button', {
        class: `node-tab${active ? ' active' : ''}${node.reachable ? '' : ' down'}`,
        role: 'tab',
        'aria-selected': active ? 'true' : 'false',
        title: node.reachable ? node.address : (node.error || 'unreachable'),
        onClick: () => selectNode(name),
      }, node.name, node.reachable ? null : h('span', { class: 'dot-down' }, '·'));
    })));
}

function localIds() {
  return new Set((state.local?.repos || []).map((repo) => repo.repo_id));
}

async function deleteRepo(repo, button) {
  const ok = await confirmDialog(
    `Delete ${repo.repo_id}?`,
    `This removes the whole repo from ${state.local.cache_dir} on ${nodeLabel()}, freeing `
    + `${bytes(repo.size_on_disk)}. Any server still pointing at it will fail to load, and `
    + 'getting it back means downloading it again.',
    { confirmLabel: 'Delete from cache' },
  );
  if (!ok) return;
  button.disabled = true;
  button.textContent = 'Deleting…';
  try {
    const query = state.node ? `?node=${encodeURIComponent(state.node)}` : '';
    const result = await del(`/hub/local/${repoPath(repo.repo_id)}${query}`);
    toast(`Deleted ${repo.repo_id} — ${bytes(result.freed_bytes)} freed`, { level: 'ok' });
    await loadLocal();
  } catch (error) {
    toast(error.message, { level: 'danger', title: 'Delete failed' });
    button.disabled = false;
    button.textContent = 'Delete';
  }
}

function localRow(repo) {
  const servable = repo.kind === 'model';
  const deleteButton = h('button', { class: 'btn-sm btn-danger' }, 'Delete');
  deleteButton.addEventListener('click', () => deleteRepo(repo, deleteButton));

  const revisions = repo.revisions || [];
  const refs = Object.keys(repo.refs || {}).join(', ');
  return h('tr', null,
    h('td', null,
      h('div', { class: 'mono' }, repo.repo_id),
      // A peer's scan has no refs to name and its path is already the tile
      // above, so the line is simply absent rather than restating either.
      h('div', { class: 'faint small' }, refs ? `refs: ${refs}` : (repo.node ? '' : repo.path))),
    h('td', null, badge(servable ? 'succeeded' : 'info', repo.kind)),
    h('td', { class: 'num' }, bytes(repo.size_on_disk)),
    h('td', { class: 'num' }, String(repo.nb_files)),
    h('td', { class: 'num' },
      revisions.length
        ? h('span', { title: revisions.map((rev) => rev.sha).join('\n') },
          String(revisions.length))
        : h('span', { class: 'faint', title: 'a peer\'s cache is scanned by size, not by ref' },
          '—')),
    h('td', { class: 'nowrap', title: when(repo.last_modified) }, ago(repo.last_modified)),
    h('td', { class: 'right nowrap' },
      h('div', { class: 'row', style: { justifyContent: 'flex-end' } },
        servable
          ? h('button', {
            class: 'btn-sm',
            title: 'Open the Serve tab with this model filled in',
            onClick: () => state.ctx.navigate('serve', repo.repo_id),
          }, 'Serve')
          : null,
        h('button', {
          class: 'btn-sm',
          onClick: () => openDetail(repo.repo_id),
        }, 'Details'),
        deleteButton)));
}

function renderLocal() {
  const host = state.nodes.local;
  if (!host) return;
  const local = state.local;
  if (!local) {
    return mount(host, h('div', { class: 'row' }, spinner(),
      `Scanning the cache on ${nodeLabel()}…`));
  }

  const repos = local.repos || [];
  const free = freeDisk();
  const disk = diskOf();
  const models = repos.filter((repo) => repo.kind === 'model').length;
  const adapters = repos.filter((repo) => repo.kind === 'adapter').length;

  mount(host,
    local.ok
      ? (local.error ? notice('warn', local.error) : null)
      : notice('danger', h('strong', null, `Cache unreadable on ${nodeLabel()}. `), local.error),
    h('div', { class: 'grid cols-4', style: { marginBottom: '14px' } },
      stat('Cache on disk', bytes(local.size_on_disk || 0), local.cache_dir),
      stat('Repos', String(repos.length), state.node
        ? `${models} model${models === 1 ? '' : 's'} · scanned over ssh`
        : `${models} servable · ${adapters} adapters`),
      stat('Disk free', free === null ? '—' : bytes(free),
        disk.total_bytes ? `of ${bytes(disk.total_bytes)}` : ''),
      state.node
        ? stat('Node', state.node, 'deletes and pulls here land on this machine')
        : stat('HF token', state.token.configured ? 'Configured' : 'Not set',
          state.token.source ? `from ${state.token.source}` : 'anonymous rate limits')),
    local.incomplete_bytes
      ? notice('warn', `${bytes(local.incomplete_bytes)} of partial downloads `
        + `(${local.incomplete_files} files) are sitting in the cache from interrupted pulls.`)
      : null,
    repos.length
      ? h('div', { class: 'table-wrap' },
        h('table', null,
          h('thead', null, h('tr', null,
            h('th', null, 'Repo'),
            h('th', null, 'Kind'),
            h('th', { class: 'num' }, 'Size'),
            h('th', { class: 'num' }, 'Files'),
            h('th', { class: 'num' }, 'Revs'),
            h('th', null, 'Modified'),
            h('th', { class: 'right' }, 'Actions'))),
          h('tbody', null, repos.map(localRow))))
      : empty(`Nothing cached on ${nodeLabel()}`,
        'Search the Hub below and pull a model to get started.'),
    repos.length
      ? h('p', { class: 'faint small', style: { marginBottom: 0 } },
        `Deleting every repo here would reclaim ${bytes(local.size_on_disk || 0)}.`)
      : null);
}

/* --- Hub search --------------------------------------------------------- */

async function runSearch() {
  const seq = ++state.searchSeq;
  const { q, pipeline_tag: pipeline, sort, gated } = state.query;
  const params = new URLSearchParams({ q, sort, limit: '40' });
  if (pipeline) params.set('pipeline_tag', pipeline);
  if (gated) params.set('gated', gated);

  mount(state.nodes.results, h('div', { class: 'row', style: { padding: '18px' } },
    spinner(), 'Searching the Hub…'));
  try {
    const payload = await get(`/hub/search?${params}`);
    if (seq !== state.searchSeq) return; // a newer keystroke already won
    state.results = payload.models || [];
    renderResults();
  } catch (error) {
    if (seq !== state.searchSeq) return;
    mount(state.nodes.results, notice('danger', error.message));
  }
}

function resultRow(model, cached) {
  const gatedLabel = model.gated === 'manual' ? 'gated (manual)'
    : model.gated ? 'gated (auto)' : '';
  return h('div', { class: 'result' },
    h('div', { class: 'r-main' },
      h('div', { class: 'row wrap' },
        h('span', { class: 'r-id' }, model.id),
        cached ? badge('succeeded', state.node ? `on ${state.node}` : 'in cache') : null,
        gatedLabel ? badge('starting', gatedLabel) : null,
        model.private ? badge('info', 'private') : null),
      h('div', { class: 'r-meta' },
        h('span', null, model.author || '—'),
        h('span', null, `${count(model.downloads)} downloads`),
        h('span', null, `${count(model.likes)} likes`),
        h('span', { title: when(model.updated) }, `updated ${ago(model.updated)}`),
        model.pipeline_tag ? h('span', null, model.pipeline_tag) : null,
        model.library ? h('span', null, model.library) : null),
      model.tags.length
        ? h('div', { class: 'models-tags' },
          model.tags.filter((tag) => !tag.includes(':')).slice(0, 8)
            .map((tag) => h('span', { class: 'tag' }, tag)))
        : null),
    h('div', { class: 'r-actions' },
      h('button', { class: 'btn-sm', onClick: () => openDetail(model.id) }, 'Details'),
      h('button', {
        class: 'btn-sm btn-primary',
        title: `Download onto ${nodeLabel()}`,
        onClick: () => pull(model.id),
      }, state.node ? `Pull to ${state.node}` : 'Pull')));
}

function renderResults() {
  const host = state.nodes.results;
  // A search that lands after the tab was left has nothing to mount into.
  if (!host) return;
  const cached = localIds();
  mount(state.nodes.results, state.results.length
    ? state.results.map((model) => resultRow(model, cached.has(model.id)))
    : empty('No models matched', 'Try a shorter query, or clear the task and gated filters.'));
}

/* --- detail ------------------------------------------------------------- */

function detailBody(detail) {
  const config = detail.config || {};
  const estimate = detail.estimate || {};
  const compatibility = detail.compatibility || {};
  const quant = config.quantization_config;
  const parameters = Object.entries(detail.parameters || {})
    .map(([dtype, n]) => `${count(n)} ${dtype}`).join(', ');

  const rows = [
    ['Architecture', (config.architectures || []).join(', ') || '—'],
    ['Model type', config.model_type || '—'],
    ['Max context', config.max_position_embeddings
      ? `${config.max_position_embeddings.toLocaleString()} tokens` : 'unknown'],
    ['dtype', config.torch_dtype || '—'],
    ['Quantization', quant
      ? (quant.quant_method || quant.format || JSON.stringify(quant).slice(0, 120))
      : 'none'],
    ['Parameters', parameters || 'unknown'],
    // Which engine reads this, rather than what it is missing. "GGUF only"
    // read as a limitation for as long as there was one engine; it is now the
    // native case for the other one, and the operator's next decision.
    ['Weights', detail.has_safetensors && detail.has_gguf ? 'safetensors and GGUF — either engine'
      : detail.has_safetensors ? 'safetensors — serve with vLLM'
        : detail.has_gguf ? 'GGUF — serve with llama.cpp'
          : 'neither safetensors nor GGUF'],
    ['Kind', detail.is_adapter
      ? `LoRA adapter (${detail.peft_type || 'peft'}) on ${detail.base_model || 'unknown base'}`
      : 'full model'],
    ['Revision', `${detail.revision} @ ${(detail.sha || '').slice(0, 12) || '—'}`],
    ['Updated', when(detail.updated)],
    ['Popularity', `${count(detail.downloads)} downloads · ${count(detail.likes)} likes`],
  ];

  return [
    detail.gated && !state.token.configured
      ? notice('danger',
        h('strong', null, 'Gated repo, no token configured. '),
        'The pull will fail with a 401 until an HF token with approved access to this repo is '
        + 'saved — use the HF token button on this page.')
      : null,
    compatibility.note
      ? notice(LEVEL_CLASS[compatibility.level] || 'info', compatibility.note)
      : null,
    h('div', { class: 'grid cols-3', style: { margin: '12px 0' } },
      stat('Download size', bytes(estimate.total_bytes || 0),
        `${estimate.files_total || 0} files`),
      stat('Already cached', bytes(estimate.cached_bytes || 0),
        `${estimate.files_cached || 0} of ${estimate.files_total || 0} files`),
      stat('To fetch', bytes(estimate.fetch_bytes || 0),
        estimate.fetch_bytes ? 'over the network' : 'nothing to do')),
    h('dl', { class: 'models-kv' },
      rows.map(([label, value]) => [h('dt', null, label), h('dd', null, value)])),
    detail.tags?.length
      ? h('div', { class: 'models-tags' },
        detail.tags.map((tag) => h('span', { class: 'tag' }, tag)))
      : null,
    h('h3', { style: { margin: '16px 0 6px', fontSize: '13px' } },
      `Files (${(detail.files || []).length})`),
    h('div', { class: 'models-files' },
      h('table', null,
        h('tbody', null, (detail.files || []).map((file) => h('tr', null,
          h('td', { class: 'mono' }, file.path),
          h('td', { class: 'num' }, bytes(file.size))))))),
  ];
}

async function openDetail(repoId) {
  const body = h('div', null, h('div', { class: 'row' }, spinner(), 'Reading the repo…'));
  const actions = h('div', { class: 'row' });
  const control = modal(repoId, body, { wide: true, actions: [actions] });

  let detail = null;
  try {
    detail = await get(`/hub/models/${repoPath(repoId)}`);
  } catch (error) {
    mount(body, notice('danger', error.message));
    return;
  }
  mount(body, detailBody(detail));

  const cached = (detail.estimate?.fetch_bytes ?? 1) === 0;
  mount(actions,
    copyButton(repoId, 'Copy id'),
    h('a', {
      class: 'btn btn-sm', href: `https://huggingface.co/${repoId}`,
      target: '_blank', rel: 'noreferrer noopener',
    }, 'Open on HF'),
    cached && !detail.is_adapter
      ? h('button', {
        class: 'btn-sm',
        onClick: () => { control.close(); state.ctx.navigate('serve', repoId); },
      }, 'Serve')
      : null,
    h('button', {
      class: 'btn-primary',
      onClick: () => { control.close(); pull(repoId, detail); },
    }, cached ? 'Re-check and pull' : 'Pull'));
}

/* --- pulling ------------------------------------------------------------ */

async function pull(repoId, known = null) {
  let detail = known;
  if (!detail) {
    try {
      detail = await get(`/hub/models/${repoPath(repoId)}`);
    } catch (error) {
      toast(error.message, { level: 'danger', title: 'Could not price the pull' });
      return;
    }
  }

  const estimate = detail.estimate || {};
  const fetchBytes = estimate.fetch_bytes || 0;
  const free = freeDisk();
  const tooBig = free !== null && fetchBytes > free;
  const lines = [
    `${bytes(fetchBytes)} to fetch of ${bytes(estimate.total_bytes || 0)} `
    + `(${bytes(estimate.cached_bytes || 0)} already cached).`,
  ];
  if (free !== null) lines.push(`Disk free on ${nodeLabel()}: ${bytes(free)}.`);
  if (tooBig) {
    lines.push('That does not fit — the download will fail part-way and leave partial blobs '
      + 'behind. Delete something from the cache first.');
  }
  if (detail.gated && !state.token.configured) {
    lines.push('This repo is gated and no HF token is configured, so the pull will 401.');
  }
  if (detail.is_adapter) {
    lines.push('This is a LoRA adapter — it needs its base model to be served with --enable-lora.');
  }

  const ok = await confirmDialog(`Pull ${repoId} onto ${nodeLabel()}?`, lines.join(' '), {
    danger: tooBig,
    confirmLabel: tooBig ? 'Pull anyway' : 'Pull',
  });
  if (!ok) return;

  try {
    const { job_id: jobId } = await post('/hub/download',
      { repo_id: repoId, node: state.node });
    attachDownload(jobId, repoId, estimate.fetch_bytes || 0);
    toast(`Pulling ${repoId} onto ${nodeLabel()} — job ${jobId}`, { level: 'ok' });
  } catch (error) {
    const level = error instanceof ApiError && error.status === 409 ? 'warn' : 'danger';
    toast(error.message, { level, title: 'Pull not started' });
  }
}

function downloadCard(jobId, repoId) {
  const bar = h('span', { style: { width: '0%' } });
  const nums = h('div', { class: 'dl-nums' });
  const transient = h('div', { class: 'dl-transient' });
  const status = badge('pending', 'pending');
  const log = logBox([]);
  const cancel = h('button', { class: 'btn-sm btn-danger' }, 'Cancel');
  cancel.addEventListener('click', async () => {
    cancel.disabled = true;
    try {
      await post(`/jobs/${jobId}/cancel`);
    } catch (error) {
      toast(error.message, { level: 'danger' });
      cancel.disabled = false;
    }
  });

  const node = h('div', { class: 'models-dl' },
    h('div', { class: 'dl-head' },
      h('span', { class: 'dl-id' }, repoId),
      status,
      h('span', { class: 'spacer' }),
      h('span', { class: 'faint small mono' }, jobId),
      cancel),
    h('div', { class: 'progress' }, bar),
    nums,
    transient,
    h('details', { class: 'collapse' },
      h('summary', null, 'Log'),
      log));

  return { node, bar, nums, transient, status, log, cancel };
}

function applyProgress(card, progress) {
  const percent = Number(progress.percent);
  if (Number.isFinite(percent)) card.bar.style.width = `${Math.max(0, Math.min(100, percent))}%`;
  mount(card.nums,
    h('span', null, 'phase ', h('b', null, progress.phase || '—')),
    h('span', null, h('b', null, Number.isFinite(percent) ? `${percent.toFixed(1)}%` : '—')),
    h('span', null, h('b', null, bytes(progress.downloaded_bytes || 0)),
      ` / ${bytes(progress.total_bytes || 0)}`),
    h('span', null, h('b', null, `${progress.files_done ?? 0}/${progress.files_total ?? 0}`),
      ' files'),
    h('span', null, speed(progress.speed_bps)),
    progress.eta ? h('span', null, `ETA ${Math.round(progress.eta)}s`) : null);
}

function attachDownload(jobId, repoId, plannedBytes = 0) {
  if (state.downloads.has(jobId)) return;
  const card = downloadCard(jobId, repoId);
  const entry = { card, close: null };
  state.downloads.set(jobId, entry);
  renderDownloads();
  if (plannedBytes) applyProgress(card, { phase: 'starting', total_bytes: plannedBytes });

  const finish = (jobStatus) => {
    card.status.className = `badge ${jobStatus}`;
    mount(card.status, jobStatus);
    card.cancel.remove();
    entry.close?.();
    entry.close = null;
    loadLocal();
    renderResults();
    toast(`${repoId}: download ${jobStatus}`,
      { level: jobStatus === 'succeeded' ? 'ok' : 'warn' });
  };

  // The per-job stream only carries structured progress in its opening `status`
  // and closing `end` frames; in between, the worker's own marker lines are the
  // live source, so they are parsed here rather than shown as log noise.
  entry.close = stream(`/jobs/${jobId}/stream`, {
    log: (payload) => {
      const line = payload?.line ?? '';
      const marked = markerPayload(line);
      if (!marked) return card.log.append(line);
      applyProgress(card, marked.final ? { ...marked.payload, phase: 'done', percent: 100 }
        : marked.payload);
      card.status.className = 'badge running';
      mount(card.status, marked.payload.phase || 'running');
    },
    // The parsed dict from the job's own parser. Marker lines are kept out of
    // the log broadcast, so this — not the raw text — is what moves the bar.
    progress: (payload) => {
      const progress = payload?.progress;
      if (progress && Object.keys(progress).length) applyProgress(card, progress);
    },
    'progress-line': (payload) => { card.transient.textContent = payload?.line ?? ''; },
    status: (payload) => {
      card.status.className = `badge ${payload?.status || 'pending'}`;
      mount(card.status, payload?.status || 'pending');
      if (payload?.progress && Object.keys(payload.progress).length) {
        applyProgress(card, payload.progress);
      }
    },
    end: (payload) => {
      const job = payload?.job;
      if (job?.progress && Object.keys(job.progress).length) applyProgress(card, job.progress);
      finish(payload?.status || 'succeeded');
    },
  });
}

function renderDownloads() {
  const host = state.nodes.downloads;
  if (!host) return;
  const cards = [...state.downloads.values()].map((entry) => entry.card.node);
  mount(host, cards.length
    ? panel('Downloads', {
      sub: `${cards.length} in this session`,
      flush: true,
      actions: h('button', {
        class: 'btn-sm btn-ghost',
        onClick: () => {
          for (const [jobId, entry] of [...state.downloads]) {
            if (!entry.close) state.downloads.delete(jobId); // finished ones only
          }
          renderDownloads();
        },
      }, 'Clear finished'),
      body: cards,
    })
    : null);
}

async function adoptRunningDownloads() {
  try {
    const { jobs } = await get('/jobs?kind=download&limit=20');
    for (const job of jobs) {
      if (['succeeded', 'failed', 'cancelled'].includes(job.status)) continue;
      attachDownload(job.id, job.spec?.meta?.repo_id || job.title, 0);
    }
  } catch {
    // The jobs tab surfaces this properly; a missing adoption is not fatal here.
  }
}

/* --- token -------------------------------------------------------------- */

async function loadToken() {
  try {
    state.token = await get('/hub/token');
  } catch (error) {
    toast(error.message, { level: 'danger' });
  }
}

function openTokenDialog() {
  const input = h('input', {
    type: 'password', placeholder: 'hf_…', autocomplete: 'off', spellcheck: false,
  });
  const save = async (value) => {
    try {
      state.token = await put('/hub/token', { token: value });
      toast(value ? 'Token saved' : 'Stored token cleared', { level: 'ok' });
      control.close();
      renderLocal();
    } catch (error) {
      toast(error.message, { level: 'danger', title: 'Could not save the token' });
    }
  };

  const control = modal('HuggingFace token', [
    h('p', { class: 'muted small' },
      'The token is stored server-side in the dashboard database and injected into download '
      + 'containers as HF_TOKEN. It only affects rate limits (500 anonymous API calls per five '
      + 'minutes, 1000 authenticated) and access to gated or private repos — nothing else on '
      + 'this page needs it.'),
    state.token.source === 'env'
      ? notice('info', 'A token is currently coming from the environment (HF_TOKEN on the '
        + 'dashboard process). That one wins; saving here has no effect until it is unset.')
      : null,
    h('div', { class: 'field' },
      h('label', null, h('span', null, 'Token')),
      input,
      h('span', { class: 'help' },
        state.token.configured
          ? `A token is configured (source: ${state.token.source}). Saving replaces it; `
            + 'saving an empty field clears the stored one.'
          : 'Create one at huggingface.co/settings/tokens with read access.')),
  ], {
    actions: [
      h('button', { onClick: () => save('') }, 'Clear stored token'),
      h('button', { class: 'btn-primary', onClick: () => save(input.value.trim()) }, 'Save'),
    ],
  });
}

/* --- view --------------------------------------------------------------- */

export async function render(container, ctx) {
  ensureStyles('models', CSS);
  state.ctx = ctx;
  state.local = null;
  state.results = [];
  state.downloads = new Map();
  state.nodes = {
    downloads: h('div', null),
    tabs: h('div', null),
    local: h('div', null),
    results: h('div', null),
  };

  const searchInput = h('input', {
    type: 'search', placeholder: 'Search HuggingFace models…', value: state.query.q,
  });
  const debounced = debounce(runSearch, 320);
  searchInput.addEventListener('input', () => {
    state.query.q = searchInput.value.trim();
    debounced();
  });
  searchInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') runSearch();
  });

  const select = (key, options) => {
    const el = h('select', null, options.map(([value, label]) =>
      h('option', { value, selected: state.query[key] === value }, label)));
    el.addEventListener('change', () => { state.query[key] = el.value; runSearch(); });
    return el;
  };

  mount(container,
    h('div', { class: 'page-head' },
      h('div', null,
        h('h1', null, 'Models'),
        h('p', null, 'The HuggingFace cache is shared by every container on a box — vLLM '
          + 'servers, fine-tuning runs and Heretic all read the same blobs, so pulling once is '
          + 'enough. Writes happen inside a root container, which is why a pull shows up as a '
          + 'job. Each node keeps its own cache: pick one below and every pull and delete on '
          + 'this page lands there.')),
      h('div', { class: 'page-actions' },
        h('button', { onClick: openTokenDialog }, 'HF token…'),
        h('button', { onClick: () => { loadRegistry(); loadLocal(); loadToken(); } },
          'Refresh'))),
    state.nodes.downloads,
    panel('Model cache', {
      sub: 'per node',
      body: [state.nodes.tabs, state.nodes.local],
    }),
    panel('HuggingFace Hub', {
      sub: 'search',
      flush: true,
      body: [
        h('div', { style: { padding: '12px 13px', borderBottom: '1px solid var(--border)' } },
          h('div', { class: 'models-filters' },
            searchInput,
            select('pipeline_tag', PIPELINES),
            select('sort', SORTS),
            select('gated', GATED))),
        state.nodes.results,
      ],
    }));

  await loadToken();
  await Promise.all([loadRegistry(), loadLocal(), runSearch(), adoptRunningDownloads()]);

  const requested = ctx.routeDetail();
  if (requested) openDetail(requested);
}

export function dispose() {
  for (const entry of state.downloads.values()) entry.close?.();
  state.downloads.clear();
  state.nodes = {};
  state.ctx = null;
}
