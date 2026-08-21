/* Heretic: find the refusal direction, ablate it, serve the result.

   This view and the Fine-tune view are deliberately the same shape — status
   strip, model picker, generated form, memory pre-flight, launch, live run,
   history — because the two tools do structurally the same thing. */

import { get, post, stream } from '../api.js';
import {
  ago, badge, bytes, confirmDialog, debounce, empty, ensureStyles, field, h,
  logBox, modal, mount, notice, panel, spinner, stat, toast,
} from '../ui.js';

const KIND = 'heretic';

// Above the fold: the knobs that change what a run costs or what it produces.
// Everything else is real but rarely touched, so it lives behind a disclosure.
const PRIMARY = ['n_trials', 'n_startup_trials', 'export_strategy', 'quantization'];

// Same wording as the Serve tab's memory guard, so the three read as one thing.
const LEVEL = { ok: 'ok', warn: 'warn', block: 'danger' };
const LEAD = { ok: 'Fits. ', warn: 'Tight. ', block: 'Blocked. ' };

const BLURB =
  'Heretic looks for the direction in the residual stream that carries refusal, then ablates '
  + 'it from the attention output and MLP down projections. A TPE search (Optuna) tunes the '
  + 'per-layer ablation weights over many trials, trading refusal rate against KL divergence '
  + 'from the original model, and the trial you pick off the resulting Pareto front is merged '
  + 'and written out as a normal model directory.';

let view = null;

/* --- entry points -------------------------------------------------------- */

export async function render(container, ctx) {
  ensureStyles(KIND, CSS);
  view = {
    ctx,
    config: {},
    schema: null,
    status: null,
    catalog: { groups: [], count: 0 },
    customModel: false,
    jobs: [],
    verdict: null,
    checkSeq: 0,
    selected: null,
    busy: false,
    logClose: null,
    buildClose: null,
    busClose: null,
    refs: {},
  };
  shell(container);
  await Promise.all([loadSchema(), loadStatus(), loadCatalog(), loadJobs()]);
  view.busClose = stream('/jobs/stream/all', { job: onJobEvent, image: () => loadStatus() });
  runCheck();
}

export function dispose() {
  if (!view) return;
  view.logClose?.();
  view.buildClose?.();
  view.busClose?.();
  view = null;
}

/* --- layout -------------------------------------------------------------- */

function shell(container) {
  const refs = view.refs;
  refs.status = h('div');
  refs.model = h('div');
  refs.form = h('div');
  refs.check = h('div');
  refs.actions = h('div', { class: 'row wrap' });
  refs.run = h('div');
  refs.history = h('div');

  mount(container,
    h('div', { class: 'page-head' },
      h('div', null,
        h('h1', null, 'Heretic'),
        h('p', null, BLURB)),
      h('div', { class: 'page-actions' },
        h('button', { class: 'btn-sm', onClick: () => { loadCatalog(); loadJobs(); loadStatus(); } },
          'Refresh'))),
    refs.status,
    refs.run,
    h('div', { class: 'split' },
      panel('Configuration', {
        sub: 'config.toml',
        body: [refs.model, refs.form],
        foot: h('div', { class: 'stack' }, refs.check, refs.actions),
      }),
      panel('Runs', { sub: KIND, body: refs.history, flush: true })));

  mount(refs.run, empty('No run selected',
    'Launch an abliteration, or pick a run from the history to replay its log.'));
  mount(refs.check,
    h('div', { class: 'row' }, spinner(), h('span', { class: 'faint' }, 'sizing…')));
}

/* --- image status -------------------------------------------------------- */

async function loadStatus() {
  try {
    const status = await get('/heretic/status');
    if (!view) return;
    view.status = status;
  } catch (error) {
    mount(view.refs.status, notice('danger', error.message));
    return;
  }
  renderStatus();
  renderActions();
}

function renderStatus() {
  const info = view.status;
  const ref = h('input', {
    type: 'text', value: info.ref || 'master', placeholder: 'master or a commit SHA',
    style: { width: '260px' },
  });
  const build = h('button', {
    class: info.present ? 'btn-sm' : 'btn-primary',
    onClick: () => startBuild(ref.value.trim() || 'master'),
  }, info.present ? 'Rebuild image' : 'Build image');

  const log = h('div');
  view.refs.buildLog = log;

  mount(view.refs.status, panel('Worker image', {
    sub: info.image,
    actions: [badge(info.present ? 'succeeded' : 'absent', info.present ? 'built' : 'missing')],
    body: [
      info.present
        ? notice('ok',
            h('strong', null, 'Ready. '),
            h('span', null,
              `Built ${ago(Date.parse(info.created) / 1000)} from heretic@${info.ref || '?'}`),
            info.commit ? h('div', { class: 'faint small mono' }, info.commit) : null)
        : notice('warn',
            h('strong', null, 'The Heretic image has not been built yet. '),
            h('span', null,
              'It layers an overlay venv onto the vLLM image that is already pulled, so the '
              + 'build is a pip install rather than a download of another 20 GiB — a few '
              + 'minutes, once. Runs cannot start until it exists.')),
      h('div', { class: 'row wrap' },
        field('Heretic ref', ref, {
          help: 'HERETIC_REF defaults to master, so a rebuild months from now resolves a '
            + 'different commit. Pass a SHA when you need the run to be reproducible.',
        }),
        build),
      log,
    ],
  }));
}

async function startBuild(ref) {
  try {
    const { job_id: jobId } = await post('/heretic/build', { ref });
    if (!view) return;
    const box = logBox([]);
    mount(view.refs.buildLog, h('div', { class: 'stack' },
      h('div', { class: 'row' }, spinner(), h('span', { class: 'faint small' },
        `docker build · job ${jobId}`)),
      box));
    view.buildClose?.();
    view.buildClose = stream(`/jobs/${jobId}/stream`, {
      log: (payload) => box.append(payload.line),
      'progress-line': (payload) => box.append(payload.line),
      end: (payload) => {
        view.buildClose?.();
        view.buildClose = null;
        toast(`Image build ${payload.status}`,
          { level: payload.status === 'succeeded' ? 'ok' : 'danger' });
        loadStatus();
      },
    });
  } catch (error) {
    toast(error.message, { level: 'danger' });
  }
}

/* --- form ---------------------------------------------------------------- */

async function loadSchema() {
  const schema = await get('/heretic/defaults');
  if (!view) return;
  view.schema = schema;
  view.config = { ...schema.defaults };
  renderModelPicker();
  renderForm();
  renderActions();
}

async function loadCatalog() {
  try {
    const payload = await get('/hub/available');
    if (!view) return;
    view.catalog = payload;
    if (!payload.cache_ok && payload.cache_error) {
      toast(`Local cache: ${payload.cache_error}`, { level: 'warn' });
    }
  } catch (error) {
    toast(error.message, { level: 'warn' });
  }
  renderModelPicker();
}

// A sentinel for the "Other" option. Not a null byte: browsers may replace one
// in an attribute value with U+FFFD, and the equality check would then never fire.
const CUSTOM = '__llmd_other__';

function catalogEntries() {
  return view.catalog.groups.flatMap((group) => group.items);
}

function renderModelPicker() {
  if (!view.schema) return;

  const entries = catalogEntries();
  const current = view.config.model || '';
  // A value that is not in the catalogue — a Hub id that has never been pulled,
  // or the schema default after that model was deleted — has to stay selectable,
  // so the picker falls back to the free-text branch rather than silently
  // resetting to whatever happens to be first in the list.
  const known = entries.some((entry) => entry.value === current);
  if (current && !known) view.customModel = true;

  const hint = h('div', { class: 'help' });

  const custom = h('input', {
    type: 'text',
    list: 'heretic-model-ids',
    value: known ? '' : current,
    placeholder: 'Qwen/Qwen3-4B-Instruct-2507 or /home/user/models/outputs/…',
    onInput: (event) => {
      view.config.model = event.target.value.trim();
      modelHint(hint);
      scheduleCheck();
      renderActions();
    },
  });

  const select = h('select', {
    onChange: (event) => {
      const value = event.target.value;
      view.customModel = value === CUSTOM;
      view.config.model = view.customModel ? custom.value.trim() : value;
      renderModelPicker();
      scheduleCheck();
      renderActions();
    },
  },
  entries.length
    ? view.catalog.groups.map((group) => h('optgroup', { label: group.label },
        group.items.map((entry) => h('option', {
          value: entry.value,
          selected: !view.customModel && entry.value === current,
          title: entry.note || entry.value,
        }, entry.detail ? `${entry.label} — ${entry.detail}` : entry.label))))
    : h('option', { value: CUSTOM, disabled: true }, 'nothing on this box yet'),
  h('option', { value: CUSTOM, selected: view.customModel || !entries.length },
    'Other — a Hub id or a path…'));

  modelHint(hint);

  mount(view.refs.model,
    field('Model', h('div', { class: 'stack' },
      h('div', { class: 'hx-picker' },
        select,
        h('button', {
          class: 'btn-sm',
          title: 'Re-read the cache and finished runs',
          onClick: async (event) => {
            event.currentTarget.disabled = true;
            await loadCatalog();
          },
        }, 'Refresh')),
      view.customModel || !entries.length ? custom : null,
      h('datalist', { id: 'heretic-model-ids' },
        entries.map((entry) => h('option', { value: entry.value }))),
      hint), {
      flag: 'model',
      help: 'Cached models and the merged output of earlier Heretic and fine-tuning runs. '
        + 'Anything else — a Hub id that has not been pulled, or a path the container can see — '
        + 'goes under "Other"; an uncached Hub id is downloaded when the run starts. Models that '
        + 'set auto_map need trust_remote_code, which transformers asks for on stdin — a detached '
        + 'container has no terminal to answer with, so those cannot be abliterated here at all.',
    }));
}

function modelHint(node) {
  const value = view.config.model || '';
  const entry = catalogEntries().find((item) => item.value === value);
  if (entry) return mount(node, entry.note ? `${entry.detail} · ${entry.note}` : entry.detail);
  if (!value) return mount(node, `${view.catalog.count} model(s) available on this box`);
  return mount(node, value.startsWith('/')
    ? 'local path · must be visible inside the container'
    : 'not cached — the run downloads it first');
}


function renderForm() {
  const fields = view.schema.fields.filter((spec) => spec.name !== 'model');
  const primary = PRIMARY.map((name) => fields.find((spec) => spec.name === name)).filter(Boolean);
  const rest = fields.filter((spec) => !PRIMARY.includes(spec.name));

  mount(view.refs.form,
    h('div', { class: 'param-grid' }, primary.map(control)),
    h('details', { class: 'collapse' },
      h('summary', null, `Advanced (${rest.length})`),
      h('div', { class: 'param-grid' }, rest.map(control))),
    h('p', { class: 'help' },
      `Direction prompts come from ${view.schema.datasets.good} and `
      + `${view.schema.datasets.bad}; the held-out test splits of the same two score every `
      + 'trial.'));
}

/** One field of the settings model. The schema is the only source of truth for
 *  type, bounds and help text, so a new knob in app/heretic.py shows up here. */
function control(spec) {
  const write = (value) => {
    view.config[spec.name] = value;
    scheduleCheck();
  };
  let input;
  if (spec.type === 'enum') {
    input = h('select', { onChange: (event) => write(event.target.value) },
      spec.options.map((option) => h('option', { value: option }, option)));
    input.value = view.config[spec.name];
  } else if (spec.type === 'integer' || spec.type === 'number') {
    input = h('input', {
      type: 'number',
      value: view.config[spec.name],
      min: spec.minimum,
      max: spec.maximum,
      step: spec.type === 'integer' ? 1 : 'any',
      onInput: (event) => write(
        event.target.value === '' ? spec.default : Number(event.target.value)),
    });
  } else {
    input = h('input', {
      type: 'text',
      value: view.config[spec.name] ?? '',
      onInput: (event) => write(event.target.value),
    });
  }
  return field(spec.name.replaceAll('_', ' '), input, { flag: spec.name, help: spec.help });
}

/* --- pre-flight ---------------------------------------------------------- */

const scheduleCheck = debounce(() => runCheck(), 450);

async function runCheck() {
  if (!view) return;
  if (!view.config.model) {
    view.verdict = null;
    mount(view.refs.check, notice('info', 'Pick a model and this will size the run against '
      + 'what the host has free.'));
    renderActions();
    return;
  }
  const seq = ++view.checkSeq;
  try {
    const verdict = await post('/heretic/check', view.config);
    if (!view || seq !== view.checkSeq) return;
    view.verdict = verdict;
    mount(view.refs.check, verdictNotice(verdict));
  } catch (error) {
    if (!view || seq !== view.checkSeq) return;
    view.verdict = null;
    mount(view.refs.check, notice('warn', `Could not size this run: ${error.message}`));
  }
  renderActions();
}

function verdictNotice(verdict) {
  const estimate = verdict.estimate;
  return notice(LEVEL[verdict.level] || 'info',
    h('div', null, h('strong', null, LEAD[verdict.level] || ''), verdict.message),
    estimate
      ? h('div', { class: 'faint small mono' },
          `load ${bytes(estimate.load_bytes)} · merge spike ${bytes(estimate.merge_bytes)}`
          + ` · peak ${bytes(estimate.peak_bytes)}`)
      : null,
    (verdict.notes || []).length
      ? h('ul', { class: 'hx-notes' }, verdict.notes.map((note) => h('li', null, note)))
      : null);
}

/* --- launch -------------------------------------------------------------- */

function renderActions() {
  if (!view.refs.actions) return;
  const blocked = view.verdict?.level === 'block';
  const ready = Boolean(view.config.model) && Boolean(view.status?.present) && !view.busy;

  mount(view.refs.actions,
    h('button', {
      class: 'btn-primary',
      disabled: !ready || blocked,
      onClick: () => launch(false),
    }, view.busy ? spinner() : null, 'Launch abliteration'),
    blocked && ready
      ? h('button', { class: 'btn-danger btn-sm', onClick: () => launch(true) }, 'Launch anyway')
      : null,
    !view.status?.present
      ? h('span', { class: 'faint small' }, 'build the image first')
      : null);
}

async function launch(force) {
  if (force) {
    const go = await confirmDialog('Override the memory guard?',
      view.verdict?.message || 'The pre-flight says this will not fit.',
      { confirmLabel: 'Launch anyway' });
    if (!go) return;
  }
  view.busy = true;
  renderActions();
  try {
    const payload = await post(`/heretic/jobs${force ? '?force=true' : ''}`, view.config);
    toast(`Launched ${payload.job_id}`, { level: 'ok' });
    await loadJobs();
    selectJob(payload.job_id);
  } catch (error) {
    toast(error.message, { level: 'danger', title: 'Launch refused' });
  } finally {
    view.busy = false;
    renderActions();
  }
}

/* --- history ------------------------------------------------------------- */

async function loadJobs() {
  try {
    const payload = await get('/heretic/jobs?limit=50');
    if (!view) return;
    view.jobs = payload.jobs || [];
  } catch (error) {
    mount(view.refs.history, notice('danger', error.message));
    return;
  }
  renderHistory();
  view.ctx.setBadge(KIND, view.jobs.filter((job) => job.status === 'running').length);
}

function renderHistory() {
  if (!view.jobs.length) {
    mount(view.refs.history, empty('No runs yet', 'A launched run appears here and keeps its '
      + 'log, its trials and its output directory after it finishes.'));
    return;
  }
  mount(view.refs.history, h('div', { class: 'table-wrap' },
    h('table', null,
      h('thead', null, h('tr', null,
        h('th', null, 'Model'), h('th', null, 'Status'), h('th', { class: 'num' }, 'Trials'),
        h('th', null, ''))),
      h('tbody', null, view.jobs.map(historyRow)))));
}

function historyRow(job) {
  const progress = job.progress || {};
  const meta = job.meta || (job.spec || {}).meta || {};
  const running = !['succeeded', 'failed', 'cancelled'].includes(job.status);
  return h('tr', { class: view.selected === job.id ? 'hx-sel' : null },
    h('td', null,
      h('div', { class: 'truncate mono' }, meta.model || job.title),
      h('div', { class: 'faint small' }, `${job.id} · ${ago(job.created_at)}`)),
    h('td', null, badge(job.status),
      job.error ? h('div', { class: 'faint small truncate' }, job.error) : null),
    h('td', { class: 'num' }, progress.total ? `${progress.step || 0}/${progress.total}` : '—'),
    h('td', { class: 'right' }, h('div', { class: 'row' },
      h('button', { class: 'btn-sm btn-ghost', onClick: () => selectJob(job.id) }, 'View'),
      running
        ? h('button', { class: 'btn-sm btn-ghost', onClick: () => cancel(job.id) }, 'Cancel')
        : null,
      job.has_output
        ? h('button', { class: 'btn-sm', onClick: () => serve(job) }, 'Serve')
        : null)));
}

async function cancel(jobId) {
  const go = await confirmDialog('Cancel this run?',
    'The container is stopped. Optuna checkpoints the trials it has already finished, so the '
    + 'work is not lost — but nothing is exported.', { confirmLabel: 'Stop it' });
  if (!go) return;
  try {
    await post(`/jobs/${jobId}/cancel`);
  } catch (error) {
    toast(error.message, { level: 'danger' });
  }
}

/* --- serving a finished run ---------------------------------------------- */

async function serve(job) {
  const { ctx } = view;
  try {
    const payload = await post(`/heretic/jobs/${job.id}/serve`);
    const safety = payload.safety || {};
    modal('Server defined', h('div', { class: 'stack' },
      notice('ok',
        h('strong', null, `${payload.server.name} `),
        h('span', null, `on port ${payload.server.port}, serving `),
        h('code', null, payload.server.model)),
      h('p', { class: 'muted' }, payload.explanation),
      notice(LEVEL[safety.level] || 'info', safety.message || ''),
      h('p', { class: 'help' },
        'The definition is saved but nothing is running yet — start it from the Serve tab once '
        + 'the memory picture suits you.')), {
      actions: [
        h('button', { onClick: () => document.getElementById('modal').close() }, 'Stay here'),
        h('button', {
          class: 'btn-primary',
          onClick: () => {
            document.getElementById('modal').close();
            ctx.navigate('serve');
          },
        }, 'Open Serve'),
      ],
    });
  } catch (error) {
    toast(error.message, { level: 'danger', title: 'Cannot serve this run' });
  }
}

/* --- live run ------------------------------------------------------------ */

function onJobEvent(payload) {
  const job = payload?.job;
  if (!job || !view) return;
  if (job.kind === KIND) {
    const index = view.jobs.findIndex((row) => row.id === job.id);
    // The list endpoint decorates rows with meta/has_output; the bus does not,
    // so keep those and refresh from the API when a run finishes.
    if (index === -1) loadJobs();
    else view.jobs[index] = { ...view.jobs[index], ...job };
    renderHistory();
    if (job.status === 'succeeded') loadJobs();
  }
  if (view.selected === job.id) updateRun(job);
}

function selectJob(jobId) {
  view.selected = jobId;
  renderHistory();
  view.logClose?.();

  const box = logBox([]);
  const refs = view.refs;
  refs.runHead = h('div', { class: 'row wrap' });
  refs.runBar = h('div', { class: 'progress' }, h('span', { style: { width: '0%' } }));
  refs.runStats = h('div', { class: 'grid cols-4' });
  refs.runChart = h('div');
  refs.runLine = h('div', { class: 'hx-transient' });
  refs.runLog = box;

  mount(refs.run, panel('Run', {
    sub: jobId,
    actions: refs.runHead,
    body: [refs.runBar, refs.runStats, refs.runChart, refs.runLine, box],
  }));

  view.logClose = stream(`/jobs/${jobId}/stream`, {
    log: (payload) => box.append(payload.line),
    'progress-line': (payload) => mount(refs.runLine, payload.line),
    status: () => refreshRun(jobId),
    end: () => {
      view.logClose?.();
      view.logClose = null;
      refreshRun(jobId);
    },
  });
  refreshRun(jobId);
}

async function refreshRun(jobId) {
  try {
    const job = await get(`/jobs/${jobId}`);
    if (!view) return;
    updateRun(job);
  } catch (error) {
    toast(error.message, { level: 'warn' });
  }
}

function updateRun(job) {
  if (!view) return;
  const refs = view.refs;
  if (!refs.runStats || view.selected !== job.id) return;
  const progress = job.progress || {};
  const meta = (job.spec || {}).meta || {};
  const trials = progress.trials || [];
  const last = progress.last_metrics || trials[trials.length - 1] || {};

  mount(refs.runHead,
    h('span', { class: 'mono truncate' }, meta.model || job.title),
    badge(job.status),
    !['succeeded', 'failed', 'cancelled'].includes(job.status)
      ? h('button', { class: 'btn-sm btn-ghost', onClick: () => cancel(job.id) }, 'Cancel')
      : null);

  const percent = Number(progress.percent || 0);
  mount(refs.runBar, h('span', { style: { width: `${Math.min(100, percent)}%` } }));

  mount(refs.runStats,
    stat('phase', progress.phase || job.status, progress.export || progress.batch_probe || ''),
    stat('trial', progress.total ? `${progress.step || 0}/${progress.total}` : '—',
      progress.batch_size ? `batch ${progress.batch_size}` : ''),
    stat('refusals', last.total ? `${last.refusals}/${last.total}` : '—',
      progress.baseline?.total
        ? `original ${progress.baseline.refusals}/${progress.baseline.total}`
        : ''),
    stat('KL divergence', fmt(last.kl_divergence), 'vs the original model'),
    stat('elapsed', progress.elapsed || '—', progress.eta ? `${progress.eta} left` : ''),
    stat('GPU', progress.allocated_gb ? `${progress.allocated_gb} GB` : '—',
      progress.reserved_gb ? `${progress.reserved_gb} GB reserved` : ''),
    stat('host RSS', progress.rss_gb ? `${progress.rss_gb} GB` : '—'),
    stat('exported', progress.chosen_trial ? `trial ${progress.chosen_trial}` : '—',
      progress.saved_to || ''));

  mount(refs.runChart, pareto(trials, progress));
}

const fmt = (value) => (Number.isFinite(value) ? Number(value).toFixed(4) : '—');

/* --- Pareto scatter ------------------------------------------------------ */

const NS = 'http://www.w3.org/2000/svg';

function s(tag, props = null, ...children) {
  const el = document.createElementNS(NS, tag);
  if (props) {
    for (const [key, value] of Object.entries(props)) {
      if (value === null || value === undefined || value === false) continue;
      el.setAttribute(key, value === true ? '' : String(value));
    }
  }
  for (const child of children.flat(3)) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

const W = 520;
const H = 300;
const PAD = { top: 14, right: 16, bottom: 38, left: 52 };

/** Every scored trial as one point: refusal rate against KL divergence.
 *  Down and to the left is better, and the lower-left hull is the front the
 *  export index picks from — which is the whole reason to look at this. */
function pareto(trials, progress) {
  const points = trials
    .filter((trial) => Number.isFinite(trial.kl_divergence) && trial.total)
    .map((trial) => ({
      trial: trial.trial,
      x: (100 * trial.refusals) / trial.total,
      y: trial.kl_divergence,
    }));
  if (points.length < 1) {
    return h('p', { class: 'help' },
      'The scatter of trials appears here once the first one has been scored.');
  }

  const baseline = progress.baseline || {};
  const baseX = baseline.total ? (100 * baseline.refusals) / baseline.total : null;
  // The x domain is the full 0-100% refusal rate, not the data's range: the
  // whole point of the plot is how far a trial moved away from the baseline.
  const maxY = niceCeil(Math.max(0.01, ...points.map((p) => p.y)) * 1.05);
  const px = (x) => PAD.left + (x / 100) * (W - PAD.left - PAD.right);
  const py = (y) => H - PAD.bottom - (y / maxY) * (H - PAD.top - PAD.bottom);

  const front = [];
  let best = Infinity;
  for (const point of [...points].sort((a, b) => a.x - b.x || a.y - b.y)) {
    if (point.y < best) {
      best = point.y;
      front.push(point);
    }
  }
  const onFront = new Set(front.map((point) => point.trial));

  const ticks = (max, count) => Array.from({ length: count + 1 }, (_, i) => (max * i) / count);
  const chart = s('svg', {
    class: 'hx-chart', viewBox: `0 0 ${W} ${H}`, role: 'img',
    'aria-label': 'Trials plotted as refusal rate against KL divergence',
  },
    ticks(maxY, 4).map((value) => s('g', null,
      s('line', {
        class: 'hx-grid', x1: PAD.left, x2: W - PAD.right, y1: py(value), y2: py(value),
      }),
      s('text', { class: 'hx-tick', x: PAD.left - 7, y: py(value) + 3, 'text-anchor': 'end' },
        value.toFixed(3)))),
    ticks(100, 4).map((value) => s('text',
      { class: 'hx-tick', x: px(value), y: H - PAD.bottom + 14, 'text-anchor': 'middle' },
      `${value.toFixed(0)}%`)),
    s('line', { class: 'hx-axis', x1: PAD.left, x2: W - PAD.right, y1: py(0), y2: py(0) }),
    s('line', { class: 'hx-axis', x1: PAD.left, x2: PAD.left, y1: PAD.top, y2: py(0) }),
    s('text', { class: 'hx-label', x: (W + PAD.left) / 2, y: H - 6, 'text-anchor': 'middle' },
      'refusal rate'),
    s('text', {
      class: 'hx-label', x: 14, y: (H - PAD.bottom + PAD.top) / 2, 'text-anchor': 'middle',
      transform: `rotate(-90 14 ${(H - PAD.bottom + PAD.top) / 2})`,
    }, 'KL divergence'),
    front.length > 1
      ? s('polyline', {
          class: 'hx-front',
          points: front.map((point) => `${px(point.x)},${py(point.y)}`).join(' '),
        })
      : null,
    points.map((point) => s('circle', {
      class: onFront.has(point.trial) ? 'hx-pt front' : 'hx-pt',
      cx: px(point.x), cy: py(point.y), r: onFront.has(point.trial) ? 4.5 : 3.5,
    }, s('title', null,
      `trial ${point.trial} · ${point.x.toFixed(0)}% refusals · KL ${point.y.toFixed(4)}`))),
    baseX !== null
      ? s('path', {
          class: 'hx-base',
          d: `M${px(baseX)},${py(0) - 7}l6,7l-6,7l-6,-7z`,
        }, s('title', null, `original model · ${baseX.toFixed(0)}% refusals`))
      : null,
    chosenPoint(points, progress, px, py));

  return h('div', { class: 'stack' },
    chart,
    h('div', { class: 'legend' },
      h('span', null, h('i', { class: 'hx-swatch front' }), 'Pareto front'),
      h('span', null, h('i', { class: 'hx-swatch' }), `trials (${points.length})`),
      h('span', null, h('i', { class: 'hx-swatch base' }), 'original model'),
      h('span', { class: 'faint' }, 'lower and further left is better')));
}

/** Round a KL axis up to a 1/2/2.5/5 x 10^n bound so the ticks read cleanly. */
function niceCeil(value) {
  const base = 10 ** Math.floor(Math.log10(value));
  return (([1, 2, 2.5, 5, 10].find((step) => value <= step * base)) ?? 10) * base;
}


function chosenPoint(points, progress, px, py) {
  const chosen = points.find((point) => point.trial === progress.chosen_trial);
  if (!chosen) return null;
  return s('circle', { class: 'hx-chosen', cx: px(chosen.x), cy: py(chosen.y), r: 8 },
    s('title', null, `exported trial ${chosen.trial}`));
}

/* --- styles -------------------------------------------------------------- */

const CSS = `
.hx-picker { display: flex; gap: 8px; align-items: center; }
.hx-picker select { flex: 1; min-width: 0; }

.hx-notes { margin: 6px 0 0; padding-left: 18px; }
.hx-notes li { margin-bottom: 3px; }
.hx-transient {
  font-family: var(--mono); font-size: 11px; color: var(--text-faint);
  min-height: 16px; margin: 6px 0; overflow: hidden; white-space: nowrap; text-overflow: ellipsis;
}
.hx-sel > td { background: var(--accent-dim); }
.hx-chart {
  width: 100%; max-width: 560px; height: auto; display: block;
  margin: 10px 0 2px;
}
.hx-chart .hx-axis { stroke: var(--border-strong); }
.hx-chart .hx-grid { stroke: var(--border); stroke-dasharray: 2 4; }
.hx-chart .hx-tick { fill: var(--text-faint); font-family: var(--mono); font-size: 9px; }
.hx-chart .hx-label { fill: var(--text-dim); font-size: 10px; }
.hx-chart .hx-pt { fill: var(--text-faint); stroke: var(--panel); stroke-width: 2; }
.hx-chart .hx-pt.front { fill: var(--accent); }
.hx-chart .hx-front { fill: none; stroke: var(--accent); stroke-width: 2; opacity: .55; }
.hx-chart .hx-base { fill: var(--danger); }
.hx-chart .hx-chosen { fill: none; stroke: var(--ok); stroke-width: 2; }
.hx-swatch { width: 8px; height: 8px; border-radius: 99px; display: inline-block;
  margin-right: 5px; background: var(--text-faint); }
.hx-swatch.front { background: var(--accent); }
.hx-swatch.base { background: var(--danger); border-radius: 2px; transform: rotate(45deg); }
`;
