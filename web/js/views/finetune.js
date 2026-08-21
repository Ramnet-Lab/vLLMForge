/* Fine-tune: QLoRA training with Unsloth, then serving what came out.

   This view and the Heretic view are deliberately the same shape — status
   strip, model picker, generated form, memory pre-flight, launch, live run,
   history — because the two tools do structurally the same thing. */

import { get, post, stream } from '../api.js';
import {
  ago, badge, bytes, confirmDialog, count, debounce, duration, empty, ensureStyles, field, h,
  logBox, modal, mount, notice, panel, spinner, stat, toast,
} from '../ui.js';

const KIND = 'finetune';

// Groups worth seeing before scrolling: what to train, on what, and what comes
// out. The rest of the schema is real but rarely touched.
const PRIMARY_GROUPS = ['run', 'export'];

// check_memory already opens its message with "Fits:"/"Tight:"/"Refusing to
// start:", so this only has to pick the colour the Serve tab uses.
const LEVEL = { ok: 'ok', warn: 'warn', block: 'danger' };

const BLURB =
  'Trains a LoRA adapter on a chat or text dataset with Unsloth on top of a 4-bit base, then '
  + 'exports either the adapter alone or a merged standalone model. Everything the run needs is '
  + 'written to config.json in its output directory, so a failed run can be replayed by hand.';

let view = null;

/* --- entry points -------------------------------------------------------- */

export async function render(container, ctx) {
  ensureStyles(KIND, CSS);
  view = {
    ctx,
    config: {},
    schema: null,
    status: null,
    local: [],
    datasets: { datasets: [], files: [] },
    preview: null,
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
  await Promise.all([loadSchema(), loadStatus(), loadLocal(), loadDatasets(), loadJobs()]);
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
  refs.dataset = h('div');
  refs.model = h('div');
  refs.form = h('div');
  refs.check = h('div');
  refs.actions = h('div', { class: 'row wrap' });
  refs.run = h('div');
  refs.history = h('div');

  mount(container,
    h('div', { class: 'page-head' },
      h('div', null,
        h('h1', null, 'Fine-tune'),
        h('p', null, BLURB)),
      h('div', { class: 'page-actions' },
        h('button', {
          class: 'btn-sm',
          onClick: () => { loadLocal(); loadDatasets(); loadJobs(); loadStatus(); },
        }, 'Refresh'))),
    refs.status,
    refs.run,
    h('div', { class: 'split' },
      h('div', { class: 'stack' },
        panel('Dataset', { body: refs.dataset }),
        panel('Configuration', {
          sub: 'config.json',
          body: [refs.model, refs.form],
          foot: h('div', { class: 'stack' }, refs.check, refs.actions),
        })),
      panel('Runs', { sub: KIND, body: refs.history, flush: true })));

  mount(refs.run, empty('No run selected',
    'Start a fine-tune, or pick a run from the history to replay its log.'));
  mount(refs.check,
    h('div', { class: 'row' }, spinner(), h('span', { class: 'faint' }, 'sizing…')));
}

/* --- image status -------------------------------------------------------- */

async function loadStatus() {
  try {
    const status = await get('/finetune/status');
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
  const imports = info.unsloth || {};
  const args = h('input', {
    type: 'text', placeholder: 'UNSLOTH_VERSION=2026.8.19, TRL_VERSION=1.10.0',
  });
  const log = h('div');
  view.refs.buildLog = log;

  mount(view.refs.status, panel('Worker image', {
    sub: info.image,
    actions: [
      badge(info.image_present ? 'succeeded' : 'absent', info.image_present ? 'built' : 'missing'),
      h('button', { class: 'btn-sm btn-ghost', onClick: probe }, 'Verify imports'),
    ],
    body: [
      info.image_present
        ? notice(imports.ok === false ? 'warn' : 'ok',
            h('strong', null, imports.ok ? 'Ready. ' : 'Built. '),
            h('span', null, imports.ok
              ? `unsloth ${imports.unsloth} · trl ${imports.trl} · peft ${imports.peft}`
                + ` · transformers ${imports.transformers} · bitsandbytes ${imports.bitsandbytes}`
                + ` · sm_${(imports.capability || []).join('')}`
              : imports.error
                ? `unsloth did not import last time it was checked: ${imports.error}`
                : 'Imports have not been verified in this image yet.'))
        : notice('warn',
            h('strong', null, 'The training image has not been built yet. '),
            h('span', null,
              `It layers pip installs onto ${info.base_image}, which is already pulled, so the `
              + 'build is minutes rather than a 20 GiB download — and it only has to happen '
              + 'once. Runs cannot start until it exists.')),
      h('div', { class: 'row wrap' },
        field('Build args', args, {
          help: 'Optional pins, comma separated. Empty means unpinned, which is what was '
            + 'verified on this box.',
        }),
        h('button', {
          class: info.image_present ? 'btn-sm' : 'btn-primary',
          onClick: () => startBuild(parseArgs(args.value)),
        }, info.image_present ? 'Rebuild image' : 'Build image')),
      log,
    ],
  }));
}

function parseArgs(raw) {
  const out = {};
  for (const chunk of String(raw).split(',')) {
    const [key, ...rest] = chunk.split('=');
    if (key.trim() && rest.length) out[key.trim()] = rest.join('=').trim();
  }
  return out;
}

async function probe() {
  toast('Running unsloth inside the image — this takes about a minute.', { level: 'ok' });
  try {
    const status = await get('/finetune/status?probe=true');
    if (!view) return;
    view.status = status;
    renderStatus();
  } catch (error) {
    toast(error.message, { level: 'danger' });
  }
}

async function startBuild(buildArgs) {
  try {
    const { job_id: jobId } = await post('/finetune/build', { build_args: buildArgs });
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

/* --- schema & form ------------------------------------------------------- */

async function loadSchema() {
  const schema = await get('/finetune/defaults');
  if (!view) return;
  view.schema = schema;
  view.config = structuredClone(schema.defaults);
  renderModelPicker();
  renderForm();
  renderDataset();
  renderActions();
}

function props() {
  return view.schema.schema.properties;
}

function datasetProps() {
  return view.schema.schema.$defs.DatasetSpec.properties;
}

async function loadLocal() {
  try {
    const payload = await get('/hub/local');
    if (!view) return;
    view.local = payload.ok
      ? (payload.repos || []).filter((repo) => repo.repo_type === 'model')
      : [];
    if (!payload.ok) toast(`Local cache: ${payload.error}`, { level: 'warn' });
  } catch (error) {
    toast(error.message, { level: 'warn' });
  }
  renderModelPicker();
}

function renderModelPicker() {
  if (!view.schema) return;
  const hint = h('div', { class: 'help' });
  const input = h('input', {
    type: 'text',
    list: 'finetune-local-models',
    value: view.config.model || '',
    placeholder: 'unsloth/Qwen3-0.6B',
    onInput: (event) => {
      view.config.model = event.target.value.trim();
      cacheHint(hint, view.config.model);
      scheduleCheck();
      renderActions();
    },
  });
  cacheHint(hint, view.config.model || '');

  mount(view.refs.model,
    field('Base model', h('div', { class: 'stack' },
      input,
      h('datalist', { id: 'finetune-local-models' },
        view.local.map((repo) => h('option', { value: repo.repo_id }))),
      hint), { flag: 'model', help: props().model.description }));
}

function cacheHint(node, value) {
  const repo = view.local.find((entry) => entry.repo_id === value);
  mount(node, repo
    ? `cached · ${bytes(repo.size_on_disk)} on disk`
    : value.startsWith('/')
      ? 'local path · must be visible inside the container'
      : value
        ? 'not in the local cache — the run downloads it first'
        : `${view.local.length} models cached locally`);
}

function renderForm() {
  const groups = view.schema.groups;
  const primary = groups.filter((group) => PRIMARY_GROUPS.includes(group.id));
  const rest = groups.filter((group) => !PRIMARY_GROUPS.includes(group.id));
  mount(view.refs.form,
    primary.map(section),
    h('details', { class: 'collapse' },
      h('summary', null, `Advanced — ${rest.map((group) => group.title.toLowerCase()).join(', ')}`),
      rest.map(section)));
}

function section(group) {
  // `model` sits in the picker above and `dataset` has its own panel; both are
  // in the schema's "run" group, so drop them here rather than duplicating.
  const names = group.fields.filter((name) => !['model', 'dataset'].includes(name));
  return h('div', { class: 'param-section' },
    h('h3', null, group.title),
    group.blurb ? h('p', { class: 'blurb' }, group.blurb) : null,
    group.id === 'export' ? exportWarning() : null,
    h('div', { class: 'param-grid' },
      names.map((name) => control(name, props()[name], () => view.config[name],
        (value) => { view.config[name] = value; scheduleCheck(); }))));
}

function exportWarning() {
  return notice('warn',
    h('strong', null, 'An adapter alone will not serve. '),
    h('span', null,
      'A LoRA saved from a 4-bit base records the bnb-4bit repo as its base model, and vLLM '
      + 'cannot load that. Serving an adapter therefore means pointing the server at the fp16 '
      + 'base with --enable-lora (the Serve action below does exactly that), or exporting '
      + 'merged_16bit and serving a standalone model. That trap costs people an afternoon.'),
    h('ul', { class: 'ft-notes' },
      (view.schema.notes || []).slice(1).map((note) => h('li', null, note))));
}

/** One field, built from its JSON-schema property. Bounds, help and choices all
 *  come from the backend, so a new knob in app/finetune.py appears here. */
function control(name, prop, read, write) {
  const choices = view.schema.choices[name];
  const value = read();
  let input;

  if (prop.type === 'boolean') {
    input = h('input', {
      type: 'checkbox', checked: Boolean(value),
      onChange: (event) => write(event.target.checked),
    });
    return field(label(name), input, { flag: name, help: prop.description, inline: true });
  }
  if (prop.type === 'array') {
    const options = choices || [];
    input = h('div', { class: 'ft-checks' }, options.map((option) => h('label', null,
      h('input', {
        type: 'checkbox',
        checked: (value || []).includes(option),
        onChange: (event) => {
          const next = new Set(read() || []);
          if (event.target.checked) next.add(option);
          else next.delete(option);
          write(options.filter((entry) => next.has(entry)));
        },
      }),
      h('span', { class: 'mono' }, option))));
    return field(label(name), input, { flag: name, help: prop.description });
  }
  if (choices || prop.enum) {
    const options = choices || prop.enum;
    const numeric = prop.type === 'integer' || prop.type === 'number';
    input = h('select', {
      onChange: (event) => write(numeric ? Number(event.target.value) : event.target.value),
    }, options.map((option) => h('option', { value: option },
      option === '' ? '(tokenizer default)' : String(option))));
    input.value = String(value);
    return field(label(name), input, { flag: name, help: prop.description });
  }
  if (prop.type === 'integer' || prop.type === 'number') {
    input = h('input', {
      type: 'number',
      value,
      min: prop.minimum,
      max: prop.maximum,
      step: prop.type === 'integer' ? 1 : 'any',
      onInput: (event) => write(
        event.target.value === '' ? prop.default ?? 0 : Number(event.target.value)),
    });
    return field(label(name), input, { flag: name, help: prop.description });
  }
  input = h('input', {
    type: 'text', value: value ?? '', onInput: (event) => write(event.target.value),
  });
  return field(label(name), input, { flag: name, help: prop.description });
}

const label = (name) => name.replaceAll('_', ' ');

/* --- datasets ------------------------------------------------------------ */

async function loadDatasets() {
  try {
    const datasets = await get('/finetune/datasets');
    if (!view) return;
    view.datasets = datasets;
  } catch (error) {
    toast(error.message, { level: 'warn' });
    return;
  }
  renderDataset();
}

function renderDataset() {
  if (!view.schema) return;
  const spec = view.config.dataset;
  const write = (key, value) => { spec[key] = value; scheduleCheck(); renderDataset(); };
  const registered = view.datasets.datasets || [];
  const loose = (view.datasets.files || []).filter((file) => !file.registered);

  const source = h('select', { onChange: (event) => write('source', event.target.value) },
    datasetProps().source.enum.map((option) => h('option', { value: option },
      option === 'hub' ? 'Hub dataset' : 'Uploaded JSONL')));
  source.value = spec.source;

  let reference;
  if (spec.source === 'local') {
    const names = [...registered.map((row) => row.name), ...loose.map((file) => file.name)];
    reference = h('select', { onChange: (event) => write('reference', event.target.value) },
      h('option', { value: '' }, names.length ? '— pick a file —' : 'nothing uploaded yet'),
      names.map((name) => h('option', { value: name }, name)));
    reference.value = spec.reference;
  } else {
    reference = h('input', {
      type: 'text', value: spec.reference, placeholder: 'yahma/alpaca-cleaned',
      onInput: (event) => { spec.reference = event.target.value.trim(); scheduleCheck(); },
    });
  }

  const advanced = ['split', 'config', 'text_field', 'messages_field', 'max_rows'];
  mount(view.refs.dataset,
    h('div', { class: 'param-grid' },
      field('source', source, { flag: 'dataset.source', help: datasetProps().source.description }),
      field('reference', reference, {
        flag: 'dataset.reference', help: datasetProps().reference.description,
      })),
    spec.source === 'local' ? uploader() : null,
    datasetPreview(),
    h('details', { class: 'collapse' },
      h('summary', null, 'Columns & slicing'),
      h('div', { class: 'param-grid' },
        advanced.map((name) => control(name, datasetProps()[name], () => spec[name],
          (value) => { spec[name] = value; scheduleCheck(); })))),
    h('p', { class: 'help' }, `Uploads land in ${view.datasets.dataset_dir}, mounted read-only `
      + 'at /datasets inside the training container.'));
}

function uploader() {
  const file = h('input', { type: 'file', accept: '.jsonl,.json', class: 'ft-file' });
  const name = h('input', { type: 'text', placeholder: 'name (defaults to the file name)' });
  const replace = h('input', { type: 'checkbox' });
  return h('div', { class: 'ft-upload' },
    h('div', { class: 'row wrap' },
      file,
      name,
      h('label', { class: 'row small faint' }, replace, 'overwrite'),
      h('button', {
        class: 'btn-sm',
        onClick: () => upload(file.files?.[0], name.value.trim(), replace.checked),
      }, 'Upload JSONL')),
    h('p', { class: 'help' },
      'One JSON object per line. Recognised shapes: a text column, messages/conversations '
      + 'chat turns, prompt + completion, or instruction + output. The file is parsed before it '
      + 'is stored, so a broken row fails here rather than twenty minutes into a run.'));
}

async function upload(file, name, replace) {
  if (!file) {
    toast('Choose a .jsonl file first', { level: 'warn' });
    return;
  }
  const body = new FormData();
  body.append('file', file);
  body.append('name', name);
  body.append('replace', String(replace));
  try {
    // FormData cannot go through api.js, which is JSON-only.
    const response = await fetch('/api/finetune/datasets/upload', { method: 'POST', body });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload?.detail || `upload failed (${response.status})`);
    if (!view) return;
    view.preview = payload;
    view.config.dataset.source = 'local';
    view.config.dataset.reference = payload.reference;
    toast(`${payload.name}: ${count(payload.rows)} rows, detected ${payload.format}`,
      { level: 'ok' });
    await loadDatasets();
    scheduleCheck();
  } catch (error) {
    toast(error.message, { level: 'danger', title: 'Upload rejected' });
  }
}

function datasetPreview() {
  const spec = view.config.dataset;
  const row = (view.datasets.datasets || []).find((entry) => entry.name === spec.reference);
  const info = view.preview?.reference === spec.reference ? view.preview : row;
  if (!info || spec.source !== 'local') return null;
  const rows = info.preview || [];
  return h('div', { class: 'ft-preview' },
    h('div', { class: 'row wrap small' },
      h('span', { class: 'tag' }, `format: ${info.format}`),
      h('span', { class: 'tag' }, `${count(info.rows)} rows`),
      h('span', { class: 'tag' }, bytes(info.size_bytes))),
    rows.map((entry, index) => h('div', { class: 'ft-row' },
      h('div', { class: 'faint small' }, `row ${index + 1}`),
      Object.entries(entry).map(([key, value]) => h('div', null,
        h('span', { class: 'ft-key' }, `${key}: `),
        h('span', null, String(value)))))));
}

/* --- pre-flight ---------------------------------------------------------- */

const scheduleCheck = debounce(() => runCheck(), 450);

async function runCheck() {
  if (!view?.schema) return;
  const seq = ++view.checkSeq;
  try {
    const verdict = await post('/finetune/check', view.config);
    if (!view || seq !== view.checkSeq) return;
    view.verdict = verdict;
    mount(view.refs.check, notice(LEVEL[verdict.level] || 'info',
      h('div', null, verdict.message),
      verdict.requested_bytes
        ? h('div', { class: 'faint small mono' },
            `estimated peak ${bytes(verdict.requested_bytes)} of `
            + `${bytes(verdict.budget.total_bytes)}`)
        : null));
  } catch (error) {
    if (!view || seq !== view.checkSeq) return;
    view.verdict = null;
    mount(view.refs.check, notice('warn', `Could not size this run: ${error.message}`));
  }
  renderActions();
}

/* --- launch -------------------------------------------------------------- */

function renderActions() {
  if (!view.refs.actions) return;
  const blocked = view.verdict?.level === 'block';
  const ready = Boolean(view.config.dataset?.reference)
    && Boolean(view.status?.image_present) && !view.busy;

  mount(view.refs.actions,
    h('button', {
      class: 'btn-primary',
      disabled: !ready || blocked,
      onClick: () => launch(false),
    }, view.busy ? spinner() : null, 'Start fine-tune'),
    blocked && ready
      ? h('button', { class: 'btn-danger btn-sm', onClick: () => launch(true) }, 'Start anyway')
      : null,
    !view.status?.image_present
      ? h('span', { class: 'faint small' }, 'build the image first')
      : !view.config.dataset?.reference
        ? h('span', { class: 'faint small' }, 'pick a dataset first')
        : null);
}

async function launch(force) {
  if (force) {
    const go = await confirmDialog('Override the memory guard?',
      view.verdict?.message || 'The pre-flight says this will not fit.',
      { confirmLabel: 'Start anyway' });
    if (!go) return;
  }
  view.busy = true;
  renderActions();
  try {
    const payload = await post(`/finetune/jobs${force ? '?force=true' : ''}`, view.config);
    toast(`Launched ${payload.job_id} → ${payload.run_dir}`, { level: 'ok' });
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
    const payload = await get('/finetune/jobs?limit=50');
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
    mount(view.refs.history, empty('No runs yet',
      'A run keeps its log, its loss curve and its output directory after it finishes.'));
    return;
  }
  mount(view.refs.history, h('div', { class: 'table-wrap' },
    h('table', null,
      h('thead', null, h('tr', null,
        h('th', null, 'Run'), h('th', null, 'Status'), h('th', { class: 'num' }, 'Steps'),
        h('th', null, ''))),
      h('tbody', null, view.jobs.map(historyRow)))));
}

function historyRow(job) {
  const progress = job.progress || {};
  const meta = (job.spec || {}).meta || {};
  const running = !['succeeded', 'failed', 'cancelled'].includes(job.status);
  return h('tr', { class: view.selected === job.id ? 'ft-sel' : null },
    h('td', null,
      h('div', { class: 'truncate mono' }, meta.model || job.title),
      h('div', { class: 'faint small' }, `${job.id} · ${ago(job.created_at)}`),
      (job.artifacts || []).length
        ? h('div', { class: 'row wrap' },
            job.artifacts.map((name) => h('span', { class: 'tag' }, name)))
        : null),
    h('td', null, badge(job.status),
      job.error ? h('div', { class: 'faint small truncate' }, job.error) : null),
    h('td', { class: 'num' },
      progress.total_steps ? `${progress.step || 0}/${progress.total_steps}` : '—'),
    h('td', { class: 'right' }, h('div', { class: 'row' },
      h('button', { class: 'btn-sm btn-ghost', onClick: () => selectJob(job.id) }, 'View'),
      running
        ? h('button', { class: 'btn-sm btn-ghost', onClick: () => cancel(job.id) }, 'Cancel')
        : null,
      job.status === 'succeeded' && (job.artifacts || []).length
        ? h('button', { class: 'btn-sm', onClick: () => serveDialog(job) }, 'Serve')
        : null)));
}

async function cancel(jobId) {
  const go = await confirmDialog('Cancel this run?',
    'The training container is stopped. Checkpoints already written to the run directory stay '
    + 'there, but nothing is exported.', { confirmLabel: 'Stop it' });
  if (!go) return;
  try {
    await post(`/jobs/${jobId}/cancel`);
  } catch (error) {
    toast(error.message, { level: 'danger' });
  }
}

/* --- serving a finished run ---------------------------------------------- */

function serveDialog(job) {
  const meta = (job.spec || {}).meta || {};
  const suggested = (meta.config?.name || meta.model || job.id).split('/').pop();
  const name = h('input', { type: 'text', value: `ft-${suggested}`.slice(0, 60) });
  const util = h('input', { type: 'number', value: 0.25, min: 0.05, max: 0.9, step: 0.05 });
  const maxLen = h('input', { type: 'number', value: 4096, min: 256, step: 256 });
  const port = h('input', { type: 'number', placeholder: 'auto', min: 1024, max: 65535 });

  const ctl = modal(`Serve ${meta.export || 'run'} from ${job.id}`, h('div', { class: 'stack' },
    h('p', { class: 'muted' }, meta.export === 'adapter'
      ? 'This run exported an adapter, so the server is defined against the fp16 base with '
        + '--enable-lora and the adapter attached under the name below. Check the model field '
        + 'afterwards: the base repo is inferred by stripping the bnb-4bit suffix.'
      : 'This run exported a standalone model directory, so it is served directly — no LoRA '
        + 'flags involved.'),
    h('div', { class: 'param-grid' },
      field('server name', name, { flag: 'name' }),
      field('gpu memory utilization', util, {
        flag: 'gpu_memory_utilization',
        help: 'GPU memory is host memory here. 0.25 of 121 GiB is 30 GiB — enough for a small '
          + 'model beside whatever is already resident.',
      }),
      field('max model len', maxLen, { flag: 'max_model_len' }),
      field('port', port, { flag: 'port', help: 'Blank picks the next free one.' }))), {
    actions: [
      h('button', { onClick: () => ctl.close() }, 'Cancel'),
      h('button', {
        class: 'btn-primary',
        onClick: async () => {
          ctl.close();
          await serve(job, {
            name: name.value.trim(),
            gpu_memory_utilization: Number(util.value),
            max_model_len: Number(maxLen.value) || null,
            port: port.value ? Number(port.value) : null,
          });
        },
      }, 'Create server'),
    ],
  });
}

async function serve(job, payload) {
  const { ctx } = view;
  try {
    const result = await post(`/finetune/jobs/${job.id}/serve`, payload);
    let opened;
    opened = modal('Server defined', h('div', { class: 'stack' },
      notice('ok',
        h('strong', null, `${result.server.name} `),
        h('span', null, `on port ${result.server.port}, serving `),
        h('code', null, result.server.model)),
      h('p', { class: 'muted' }, result.note),
      h('p', { class: 'help' },
        'The definition is saved but nothing is running yet — start it from the Serve tab.')), {
      actions: [
        h('button', { onClick: () => opened.close() }, 'Stay here'),
        h('button', {
          class: 'btn-primary',
          onClick: () => {
            opened.close();
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
    // The list endpoint decorates rows with the artefacts on disk; the bus does
    // not, so keep what we have and re-list when a run reaches a terminal state.
    if (index === -1 || job.status === 'succeeded') loadJobs();
    else view.jobs[index] = { ...view.jobs[index], ...job };
    renderHistory();
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
  refs.runResult = h('div');
  refs.runLine = h('div', { class: 'ft-transient' });
  refs.runLog = box;

  mount(refs.run, panel('Run', {
    sub: jobId,
    actions: refs.runHead,
    body: [refs.runBar, refs.runStats, refs.runChart, refs.runResult, refs.runLine, box],
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
  const history = progress.loss_history || [];

  mount(refs.runHead,
    h('span', { class: 'mono truncate' }, meta.model || job.title),
    badge(job.status),
    !['succeeded', 'failed', 'cancelled'].includes(job.status)
      ? h('button', { class: 'btn-sm btn-ghost', onClick: () => cancel(job.id) }, 'Cancel')
      : null);

  mount(refs.runBar,
    h('span', { style: { width: `${Math.min(100, Number(progress.percent || 0))}%` } }));

  mount(refs.runStats,
    stat('phase', progress.phase || job.status, meta.export || ''),
    stat('step', progress.total_steps ? `${progress.step || 0}/${progress.total_steps}` : '—',
      progress.epoch !== undefined ? `epoch ${Number(progress.epoch).toFixed(2)}` : ''),
    stat('loss', Number.isFinite(progress.loss) ? Number(progress.loss).toFixed(4) : '—',
      history.length ? `from ${Number(history[0][1]).toFixed(3)}` : ''),
    stat('learning rate',
      Number.isFinite(progress.learning_rate) ? progress.learning_rate.toExponential(2) : '—',
      Number.isFinite(progress.grad_norm) ? `grad norm ${progress.grad_norm.toFixed(2)}` : ''),
    stat('ETA', Number.isFinite(progress.eta) ? duration(progress.eta) : '—',
      Number.isFinite(progress.train_runtime) ? `ran ${duration(progress.train_runtime)}` : ''));

  mount(refs.runChart, sparkline(history));
  // The manager lifts the worker's closing @@RESULT@@ payload onto the job
  // row; older rows may still carry it inside progress.
  mount(refs.runResult, result(job.result && Object.keys(job.result).length
    ? job.result
    : progress.result));
}

function result(payload) {
  if (!payload) return null;
  return notice('ok',
    h('strong', null, `Exported ${payload.export}. `),
    h('span', null, payload.merged_dir
      ? `Merged model at ${payload.merged_dir} (container path).`
      : `Adapter at ${payload.adapter_dir} (container path), rank ${payload.lora_rank}, `
        + `base ${payload.adapter_base_model}.`),
    h('div', { class: 'faint small mono' },
      `final train loss ${Number(payload.metrics?.train_loss ?? NaN).toFixed(4)}`
      + ` · ${duration(payload.metrics?.train_runtime)}`));
}

/* --- loss sparkline ------------------------------------------------------ */

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
const H = 180;
const PAD = { top: 12, right: 14, bottom: 26, left: 52 };

/** Training loss against step. One series, so no legend — the caption names it. */
function sparkline(history) {
  if (history.length < 2) {
    return h('p', { class: 'help' },
      'The loss curve appears here once the trainer has logged a second step.');
  }
  const steps = history.map(([step]) => step);
  const losses = history.map(([, loss]) => loss);
  const minStep = Math.min(...steps);
  const maxStep = Math.max(...steps);
  const minLoss = Math.min(...losses);
  const maxLoss = Math.max(...losses);
  const spread = maxLoss - minLoss || 1;
  const px = (step) => PAD.left
    + ((step - minStep) / (maxStep - minStep || 1)) * (W - PAD.left - PAD.right);
  const py = (loss) => PAD.top
    + (1 - (loss - minLoss) / spread) * (H - PAD.top - PAD.bottom);

  const line = history.map(([step, loss]) => `${px(step)},${py(loss)}`).join(' ');
  const last = history[history.length - 1];

  return h('div', { class: 'stack' },
    s('svg', {
      class: 'ft-chart', viewBox: `0 0 ${W} ${H}`, role: 'img',
      'aria-label': `Training loss from ${losses[0].toFixed(3)} to ${last[1].toFixed(3)}`,
    },
      [maxLoss, minLoss + spread / 2, minLoss].map((value) => s('g', null,
        s('line', {
          class: 'ft-grid', x1: PAD.left, x2: W - PAD.right, y1: py(value), y2: py(value),
        }),
        s('text', { class: 'ft-tick', x: PAD.left - 7, y: py(value) + 3, 'text-anchor': 'end' },
          value.toFixed(3)))),
      s('polygon', {
        class: 'ft-area',
        points: `${px(minStep)},${py(minLoss)} ${line} ${px(maxStep)},${py(minLoss)}`,
      }),
      s('polyline', { class: 'ft-line', points: line }),
      s('circle', { class: 'ft-last', cx: px(last[0]), cy: py(last[1]), r: 4 },
        s('title', null, `step ${last[0]} · loss ${last[1].toFixed(4)}`)),
      s('text', { class: 'ft-tick', x: PAD.left, y: H - 8 }, `step ${minStep}`),
      s('text', { class: 'ft-tick', x: W - PAD.right, y: H - 8, 'text-anchor': 'end' },
        `step ${maxStep}`)),
    h('p', { class: 'help' },
      `Training loss, ${history.length} logged points, `
      + `${losses[0].toFixed(3)} → ${last[1].toFixed(3)}.`));
}

/* --- styles -------------------------------------------------------------- */

const CSS = `
.ft-notes { margin: 6px 0 0; padding-left: 18px; }
.ft-notes li { margin-bottom: 3px; }
.ft-transient {
  font-family: var(--mono); font-size: 11px; color: var(--text-faint);
  min-height: 16px; margin: 6px 0; overflow: hidden; white-space: nowrap; text-overflow: ellipsis;
}
.ft-sel > td { background: var(--accent-dim); }
.ft-checks { display: flex; flex-wrap: wrap; gap: 4px 12px; }
.ft-checks label { display: flex; align-items: center; gap: 5px; font-size: 12px; }
.ft-file { width: auto; font-size: 12px; }
.ft-upload { border-top: 1px solid var(--border); padding-top: 12px; margin-top: 4px; }
.ft-preview {
  background: var(--bg-sunken); border: 1px solid var(--border);
  border-radius: var(--radius-s); padding: 10px 12px; margin: 12px 0;
}
.ft-preview .ft-row {
  font-family: var(--mono); font-size: 11px; line-height: 1.5; color: var(--text-dim);
  border-top: 1px solid var(--border); padding-top: 6px; margin-top: 6px;
  overflow: hidden; text-overflow: ellipsis;
}
.ft-preview .ft-key { color: var(--accent); }
.ft-chart {
  width: 100%; max-width: 560px; height: auto; display: block;
  margin: 10px 0 2px;
}
.ft-chart .ft-grid { stroke: var(--border); stroke-dasharray: 2 4; }
.ft-chart .ft-tick { fill: var(--text-faint); font-family: var(--mono); font-size: 9px; }
.ft-chart .ft-line { fill: none; stroke: var(--accent); stroke-width: 2; }
.ft-chart .ft-area { fill: var(--accent); opacity: .12; stroke: none; }
.ft-chart .ft-last { fill: var(--accent); stroke: var(--panel); stroke-width: 2; }
`;
