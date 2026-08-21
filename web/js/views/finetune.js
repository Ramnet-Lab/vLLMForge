/* Fine-tune: QLoRA training with Unsloth, then serving what came out.

   This view and the Heretic view are deliberately the same shape — status
   strip, model picker, generated form, memory pre-flight, launch, live run,
   history — because the two tools do structurally the same thing. */

import { ApiError, del, get, post, stream } from '../api.js';
import {
  ago, badge, bytes, confirmDialog, count, debounce, duration, empty, ensureStyles, field, h,
  logBox, modal, mount, notice, panel, spinner, stat, toast, when,
} from '../ui.js';

const KIND = 'finetune';

// Groups worth seeing before scrolling: what to train, on what, and what comes
// out. The rest of the schema is real but rarely touched.
const PRIMARY_GROUPS = ['run', 'export'];

// check_memory already opens its message with "Fits:"/"Tight:"/"Refusing to
// start:", so this only has to pick the colour the Serve tab uses.
const LEVEL = { ok: 'ok', warn: 'warn', block: 'danger' };

// The three axes worth sorting a training set on. The Hub offers more, but
// nothing else says anything about whether a dataset is any good.
const SORTS = [
  ['downloads', 'Most downloaded'],
  ['likes', 'Most liked'],
  ['lastModified', 'Recently updated'],
];

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
    datasets: { datasets: [], files: [], cached: [] },
    details: new Map(),
    hub: { q: '', sort: 'downloads', results: null, seq: 0 },
    pulls: new Map(),
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
  searchHub(view.hub.q);
  adoptPulls();
  runCheck();
}

export function dispose() {
  if (!view) return;
  view.logClose?.();
  view.buildClose?.();
  view.busClose?.();
  for (const entry of view.pulls.values()) entry.close?.();
  view = null;
}

/* --- layout -------------------------------------------------------------- */

function shell(container) {
  const refs = view.refs;
  refs.status = h('div');
  refs.dataset = h('div');
  refs.sources = h('div');
  refs.pulls = h('div');
  refs.hubResults = h('div');
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
        panel('Dataset', { body: [refs.dataset, refs.sources, refs.pulls, hubBrowser()] }),
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
  renderSources();
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
  renderSources();
  renderHubResults();
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
      onInput: (event) => {
        spec.reference = event.target.value.trim();
        scheduleCheck();
        // Typing an id by hand deserves the same answer as picking one from the
        // browser: which column the worker would train on, before the run.
        if (spec.reference.includes('/') && !spec.reference.endsWith('/')) {
          lookupShape(spec.reference);
        }
      },
    });
  }

  const advanced = ['split', 'config', 'text_field', 'messages_field', 'max_rows'];
  mount(view.refs.dataset,
    h('div', { class: 'param-grid' },
      field('source', source, { flag: 'dataset.source', help: datasetProps().source.description }),
      field('reference', reference, {
        flag: 'dataset.reference', help: datasetProps().reference.description,
      })),
    spec.source === 'hub' ? trainingNotice(spec.reference) : null,
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

/* --- where a dataset can come from --------------------------------------- */

/* Dataset ids are `owner/name` and the hub routes take them as a path
   parameter, so the slash has to survive encoding. */
const repoPath = (repoId) => repoId.split('/').map(encodeURIComponent).join('/');

/** The config wants a bare file name for an upload and the repo id for a Hub
 *  dataset — those are what the launch resolves against /datasets and the
 *  shared cache respectively. */
// Module scope, not per render: a debouncer rebuilt on every keystroke never
// actually debounces.
const lookupShape = debounce((repoId) => {
  if (view && view.config.dataset.reference === repoId) applyShape(repoId);
}, 600);


function selectDataset(source, reference) {
  const spec = view.config.dataset;
  spec.source = source;
  spec.reference = reference;
  renderDataset();
  renderSources();
  renderActions();
  scheduleCheck();
  if (source === 'hub') applyShape(reference);
}

/** Which column the worker can read is only knowable from the detail call, so
 *  picking a Hub dataset fetches it and fills the column fields in. A dataset
 *  the datasets server never converted simply leaves them alone. */
async function applyShape(repoId) {
  try {
    await loadDetail(repoId);
  } catch {
    // An unreachable datasets server is no reason to refuse the pick; the
    // column fields keep whatever they had.
    return;
  }
  if (!view) return;
  const spec = view.config.dataset;
  if (spec.source !== 'hub' || spec.reference !== repoId) return;
  const training = view.details.get(repoId)?.training;
  if (!training?.note) return renderDataset();

  // Both fields are rewritten together, never just the one that applies:
  // switching from a conversational set to an instruction one used to leave
  // messages_field pointing at a column the new dataset does not have.
  //
  // An instruction/output pair has no single column worth training on — the
  // notice says to render the two together — so it clears both and lets the
  // user choose. The worker reads text_field first and only falls through to
  // the chat path when that column is absent, which is why a conversational set
  // must leave text_field empty or the rendered turns would never be built.
  const chat = training.format === 'messages';
  const usable = training.field && training.format !== 'instruction';
  spec.text_field = usable && !chat ? training.field : '';
  spec.messages_field = usable && chat ? training.field : '';
  scheduleCheck();
  renderDataset();
}

function trainingNotice(repoId) {
  const training = view.details.get(repoId)?.training;
  if (!training?.note) return null;
  return notice(LEVEL[training.level] || 'info',
    h('strong', null, training.field
      ? `Trains on '${training.field}'. `
      : 'No column this can train on. '),
    h('span', null, training.note));
}

function renderSources() {
  if (!view.schema) return;
  const uploads = view.datasets.datasets || [];
  const loose = (view.datasets.files || []).filter((file) => !file.registered);
  const cached = view.datasets.cached || [];

  mount(view.refs.sources,
    h('h3', { class: 'ft-sub' }, 'Available datasets'),
    view.datasets.cache_ok === false
      ? notice('warn', 'The shared cache could not be read, so datasets already pulled from the '
        + 'Hub are missing from this list.')
      : null,
    uploads.length + loose.length + cached.length
      ? [
        sourceGroup('Uploaded JSONL', uploads.map(uploadRow)),
        sourceGroup('Loose files in the dataset dir', loose.map(looseRow)),
        sourceGroup('Hub datasets in the cache', cached.map(cachedRow)),
      ]
      : empty('Nothing to train on yet',
        'Upload a JSONL above, or search the Hub below and pull a dataset.'));
}

function sourceGroup(title, rows) {
  if (!rows.length) return null;
  return h('div', { class: 'ft-group' },
    h('div', { class: 'ft-group-head' },
      h('span', null, title),
      h('span', null, String(rows.length))),
    h('div', { class: 'ft-list' }, rows));
}

function sourceRow(source, reference, meta, actions = null) {
  const spec = view.config.dataset;
  const chosen = spec.source === source && spec.reference === reference;
  return h('div', { class: `result${chosen ? ' ft-chosen' : ''}` },
    h('div', { class: 'r-main' },
      h('div', { class: 'row wrap' },
        h('span', { class: 'r-id truncate' }, reference),
        chosen ? badge('succeeded', 'selected') : null),
      h('div', { class: 'r-meta' }, meta.filter(Boolean).map((entry) => h('span', null, entry)))),
    h('div', { class: 'r-actions' },
      actions,
      h('button', {
        class: chosen ? 'btn-sm' : 'btn-sm btn-primary',
        disabled: chosen,
        onClick: () => selectDataset(source, reference),
      }, chosen ? 'Selected' : 'Use')));
}

function uploadRow(row) {
  return sourceRow('local', row.name,
    [`${count(row.rows)} rows`, row.format, bytes(row.size_bytes), `added ${ago(row.created_at)}`]);
}

function looseRow(file) {
  return sourceRow('local', file.name,
    [bytes(file.size_bytes), `modified ${ago(file.modified_at)}`, 'never parsed by an upload']);
}

function cachedRow(row) {
  const remove = h('button', { class: 'btn-sm btn-danger' }, 'Delete');
  remove.addEventListener('click', () => deleteCached(row, remove));
  return sourceRow('hub', row.reference,
    [bytes(row.size_bytes), `${row.nb_files} files`, `pulled ${ago(row.last_modified)}`],
    [h('button', { class: 'btn-sm', onClick: () => openDetail(row.repo_id) }, 'Details'), remove]);
}

async function deleteCached(row, button) {
  const go = await confirmDialog(`Delete ${row.repo_id}?`,
    `This removes the dataset from the shared cache at ${row.path}, freeing `
    + `${bytes(row.size_bytes)}. Any run still pointing at it downloads it again first.`,
    { confirmLabel: 'Delete from cache' });
  if (!go || !view) return;
  button.disabled = true;
  try {
    const result = await del(`/hub/datasets/local/${repoPath(row.repo_id)}`);
    toast(`Deleted ${row.repo_id} — ${bytes(result.freed_bytes)} freed`, { level: 'ok' });
    view.details.delete(row.repo_id);
    await loadDatasets();
  } catch (error) {
    toast(error.message, { level: 'danger', title: 'Delete failed' });
    button.disabled = false;
  }
}

/* --- the Hub ------------------------------------------------------------- */

function hubBrowser() {
  const search = h('input', {
    type: 'search', placeholder: 'Search HuggingFace datasets…', value: view.hub.q,
  });
  const debounced = debounce(() => searchHub(search.value.trim()), 320);
  search.addEventListener('input', debounced);
  search.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') searchHub(search.value.trim());
  });

  const sort = h('select', null,
    SORTS.map(([value, text]) => h('option', { value }, text)));
  sort.value = view.hub.sort;
  sort.addEventListener('change', () => {
    view.hub.sort = sort.value;
    searchHub(search.value.trim());
  });

  return h('div', { class: 'ft-hub' },
    h('h3', { class: 'ft-sub' }, 'HuggingFace Hub'),
    h('div', { class: 'row wrap' }, search, sort),
    view.refs.hubResults);
}

async function searchHub(q) {
  if (!view) return; // a debounced keystroke can land after the tab was left
  const seq = ++view.hub.seq;
  view.hub.q = q;
  mount(view.refs.hubResults,
    h('div', { class: 'row', style: { padding: '14px 2px' } }, spinner(), 'Searching the Hub…'));
  const params = new URLSearchParams({ q, sort: view.hub.sort, limit: '25' });
  try {
    const payload = await get(`/hub/datasets/search?${params}`);
    if (!view || seq !== view.hub.seq) return; // a newer keystroke already won
    view.hub.results = payload.results || [];
  } catch (error) {
    if (!view || seq !== view.hub.seq) return;
    mount(view.refs.hubResults, notice('danger', error.message));
    return;
  }
  renderHubResults();
}

function renderHubResults() {
  if (!view.refs.hubResults || view.hub.results === null) return;
  const cached = new Set((view.datasets.cached || []).map((row) => row.repo_id));
  mount(view.refs.hubResults, view.hub.results.length
    ? view.hub.results.map((row) => hubRow(row, cached.has(row.id)))
    : empty('No datasets matched', 'The Hub matches on the id, so try a shorter query.'));
}

function hubRow(row, isCached) {
  return h('div', { class: 'result' },
    h('div', { class: 'r-main' },
      h('div', { class: 'row wrap' },
        h('span', { class: 'r-id' }, row.id),
        isCached ? badge('succeeded', 'in cache') : null,
        row.gated ? badge('starting', 'gated') : null,
        row.private ? badge('info', 'private') : null),
      h('div', { class: 'r-meta' },
        h('span', null, row.author || '—'),
        h('span', null, `${count(row.downloads)} downloads`),
        h('span', null, `${count(row.likes)} likes`),
        h('span', { title: when(row.updated) }, `updated ${ago(row.updated)}`)),
      (row.tags || []).length
        ? h('div', { class: 'row wrap ft-tags' },
          row.tags.slice(0, 6).map((tag) => h('span', { class: 'tag' }, tag)))
        : null),
    h('div', { class: 'r-actions' },
      h('button', { class: 'btn-sm', onClick: () => openDetail(row.id) }, 'Details'),
      isCached
        ? h('button', {
          class: 'btn-sm btn-primary', onClick: () => selectDataset('hub', row.id),
        }, 'Use')
        : null,
      h('button', {
        class: isCached ? 'btn-sm' : 'btn-sm btn-primary',
        onClick: () => pullDataset(row.id),
      }, 'Pull')));
}

/* --- dataset detail ------------------------------------------------------ */

async function loadDetail(repoId) {
  const known = view?.details.get(repoId);
  if (known) return known;
  const detail = await get(`/hub/datasets/${repoPath(repoId)}`);
  view?.details.set(repoId, detail);
  return detail;
}

async function openDetail(repoId) {
  const body = h('div', null, h('div', { class: 'row' }, spinner(), 'Reading the dataset…'));
  const actions = h('div', { class: 'row' });
  const control = modal(repoId, body, { wide: true, actions: [actions] });

  let detail;
  try {
    detail = await loadDetail(repoId);
  } catch (error) {
    mount(body, notice('danger', error.message));
    return;
  }
  mount(body, detailBody(detail));

  const complete = (detail.estimate?.fetch_bytes ?? 1) === 0;
  mount(actions,
    h('button', {
      onClick: () => { control.close(); selectDataset('hub', repoId); },
    }, 'Train on this'),
    h('button', {
      class: 'btn-primary',
      onClick: () => { control.close(); pullDataset(repoId, detail); },
    }, complete ? 'Re-check and pull' : 'Pull'));
}

function detailBody(detail) {
  const estimate = detail.estimate || {};
  const training = detail.training || {};
  const columns = detail.columns || [];
  const splits = detail.splits || [];
  const rows = (detail.sample_rows || []).slice(0, 3);
  const facts = [
    ['Configs', (detail.configs || []).join(', ') || 'a single unnamed config'],
    ['Splits', splits.length
      ? splits.map((entry) => `${entry.config}/${entry.split}`).join(', ')
      : 'the datasets server has not converted this repo'],
    ['Popularity', `${count(detail.downloads)} downloads · ${count(detail.likes)} likes`],
    ['Updated', when(detail.updated)],
    ['Revision', `${detail.revision} @ ${(detail.sha || '').slice(0, 12) || '—'}`],
  ];

  return [
    training.note
      ? notice(LEVEL[training.level] || 'info',
        h('strong', null, training.field
          ? `Trainable: ${training.format} in '${training.field}'. `
          : 'Not trainable as it stands. '),
        h('span', null, training.note))
      : null,
    h('div', { class: 'grid cols-3', style: { margin: '12px 0' } },
      stat('Download size', bytes(detail.total_bytes || 0), `${estimate.files_total || 0} files`),
      stat('Already cached', bytes(estimate.cached_bytes || 0),
        `${estimate.files_cached || 0} of ${estimate.files_total || 0} files`),
      stat('To fetch', bytes(estimate.fetch_bytes || 0),
        estimate.fetch_bytes ? 'over the network' : 'nothing to do')),
    h('dl', { class: 'ft-kv' },
      facts.map(([key, value]) => [h('dt', null, key), h('dd', null, value)])),
    detail.tags?.length
      ? h('div', { class: 'row wrap ft-tags' },
        detail.tags.map((tag) => h('span', { class: 'tag' }, tag)))
      : null,
    h('h4', { class: 'ft-sub' }, `Columns (${columns.length})`),
    columns.length
      ? h('div', { class: 'table-wrap' },
        h('table', null,
          h('tbody', null, columns.map((column) => h('tr', null,
            h('td', { class: 'mono' }, column.name),
            h('td', { class: 'faint mono' }, column.type || '—'),
            h('td', { class: 'right' },
              column.name === training.field ? badge('succeeded', 'trains on this') : null))))))
      : h('p', { class: 'help' }, 'No parsed schema, so the columns are unknown until the '
        + 'dataset is pulled and opened by hand.'),
    rows.length
      ? [
        h('h4', { class: 'ft-sub' }, `Sample rows (${rows.length})`),
        h('div', { class: 'ft-preview' }, rows.map(sampleRow)),
      ]
      : null,
    h('h4', { class: 'ft-sub' }, `Files (${(detail.files || []).length})`),
    h('div', { class: 'ft-files' },
      h('table', null,
        h('tbody', null, (detail.files || []).map((file) => h('tr', null,
          h('td', { class: 'mono' }, file.path),
          h('td', { class: 'num' }, bytes(file.size))))))),
  ];
}

function sampleRow(row, index) {
  return h('div', { class: 'ft-row' },
    h('div', { class: 'faint small' }, `row ${index + 1}`),
    Object.entries(row).map(([key, value]) => h('div', null,
      h('span', { class: 'ft-key' }, `${key}: `),
      h('span', null, cell(value)))));
}

/* A sample row arrives as real JSON, so a chat column is an array of objects
   and String() on it reads "[object Object]". */
function cell(value) {
  const text = typeof value === 'string' ? value : (JSON.stringify(value) ?? '');
  return text.length > 240 ? `${text.slice(0, 240)}…` : text;
}

/* --- pulling ------------------------------------------------------------- */

const speed = (value) => (Number.isFinite(value) && value > 0 ? `${bytes(value)}/s` : '—');

async function pullDataset(repoId, known = null) {
  let detail = known;
  if (!detail) {
    try {
      detail = await loadDetail(repoId);
    } catch (error) {
      toast(error.message, { level: 'danger', title: 'Could not price the pull' });
      return;
    }
  }
  const estimate = detail.estimate || {};
  const free = view?.ctx.telemetry()?.disk?.free_bytes ?? null;
  const tooBig = free !== null && (estimate.fetch_bytes || 0) > free;
  const lines = [
    `${bytes(estimate.fetch_bytes || 0)} to fetch of ${bytes(detail.total_bytes || 0)} `
    + `(${bytes(estimate.cached_bytes || 0)} already cached).`,
  ];
  if (free !== null) lines.push(`Disk free: ${bytes(free)}.`);
  if (tooBig) {
    lines.push('That does not fit — the pull dies part-way and leaves partial blobs behind.');
  }

  const go = await confirmDialog(`Pull ${repoId}?`, lines.join(' '),
    { danger: tooBig, confirmLabel: tooBig ? 'Pull anyway' : 'Pull' });
  if (!go || !view) return;

  try {
    const { job_id: jobId } = await post('/hub/datasets/download', { repo_id: repoId });
    attachPull(jobId, repoId, estimate.fetch_bytes || 0);
    toast(`Pulling ${repoId} — job ${jobId}`, { level: 'ok' });
  } catch (error) {
    const level = error instanceof ApiError && error.status === 409 ? 'warn' : 'danger';
    toast(error.message, { level, title: 'Pull not started' });
  }
}

function pullCard(jobId, repoId) {
  const bar = h('span', { style: { width: '0%' } });
  const nums = h('div', { class: 'dl-nums' });
  const transient = h('div', { class: 'ft-transient' });
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

  const node = h('div', { class: 'ft-dl' },
    h('div', { class: 'dl-head' },
      h('span', { class: 'dl-id truncate' }, repoId),
      status,
      h('span', { class: 'spacer' }),
      h('span', { class: 'faint small mono' }, jobId),
      cancel),
    h('div', { class: 'progress' }, bar),
    nums,
    transient,
    h('details', { class: 'collapse' }, h('summary', null, 'Log'), log));

  return { node, bar, nums, transient, status, log, cancel };
}

function applyPullProgress(card, progress) {
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
    progress.eta ? h('span', null, `ETA ${duration(progress.eta)}`) : null);
}

function attachPull(jobId, repoId, plannedBytes = 0) {
  if (view.pulls.has(jobId)) return;
  const card = pullCard(jobId, repoId);
  const entry = { card, close: null };
  view.pulls.set(jobId, entry);
  renderPulls();
  if (plannedBytes) applyPullProgress(card, { phase: 'starting', total_bytes: plannedBytes });

  const finish = (jobStatus) => {
    if (!view) return;
    card.status.className = `badge ${jobStatus}`;
    mount(card.status, jobStatus);
    card.cancel.remove();
    entry.close?.();
    entry.close = null;
    // Whatever landed just invalidated the cached estimate for this repo.
    view.details.delete(repoId);
    loadDatasets();
    toast(`${repoId}: pull ${jobStatus}`, { level: jobStatus === 'succeeded' ? 'ok' : 'warn' });
  };

  entry.close = stream(`/jobs/${jobId}/stream`, {
    log: (payload) => card.log.append(payload.line),
    // The parsed dict from the download parser. The marker lines it reads are
    // kept out of the log broadcast, so this is what moves the bar.
    progress: (payload) => {
      const progress = payload?.progress;
      if (progress && Object.keys(progress).length) applyPullProgress(card, progress);
    },
    // A redraw of one line rather than a new one; appending would bury the log
    // under thousands of near-identical rows.
    'progress-line': (payload) => { card.transient.textContent = payload?.line ?? ''; },
    status: (payload) => {
      card.status.className = `badge ${payload?.status || 'pending'}`;
      mount(card.status, payload?.status || 'pending');
      if (payload?.progress && Object.keys(payload.progress).length) {
        applyPullProgress(card, payload.progress);
      }
    },
    end: (payload) => {
      const job = payload?.job;
      if (job?.progress && Object.keys(job.progress).length) applyPullProgress(card, job.progress);
      finish(payload?.status || 'succeeded');
    },
  });
}

function renderPulls() {
  if (!view.refs.pulls) return;
  const cards = [...view.pulls.values()].map((entry) => entry.card.node);
  mount(view.refs.pulls, cards.length
    ? h('div', { class: 'ft-pulls' },
      h('div', { class: 'row' },
        h('span', { class: 'faint small' }, `${cards.length} pulls in this session`),
        h('span', { class: 'spacer' }),
        h('button', {
          class: 'btn-sm btn-ghost',
          onClick: () => {
            for (const [jobId, entry] of [...view.pulls]) {
              if (!entry.close) view.pulls.delete(jobId); // finished ones only
            }
            renderPulls();
          },
        }, 'Clear finished')),
      cards)
    : null);
}

/** A pull outlives the tab it was started from, so reattach to the ones still
 *  running instead of leaving them invisible until they finish. */
async function adoptPulls() {
  try {
    const { jobs } = await get('/jobs?kind=download&limit=20');
    if (!view) return;
    for (const job of jobs) {
      const meta = (job.spec || {}).meta || {};
      if (meta.repo_type !== 'dataset') continue;
      if (['succeeded', 'failed', 'cancelled'].includes(job.status)) continue;
      attachPull(job.id, meta.repo_id || job.title);
    }
  } catch {
    // The Jobs tab is where an unreadable job list gets reported properly.
  }
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

.ft-sub { margin: 18px 0 8px; font-size: 12px; font-weight: 620; color: var(--text-dim); }
.ft-group { margin-bottom: 12px; }
.ft-group-head {
  display: flex; gap: 8px; margin-bottom: 5px; font-size: 11px;
  text-transform: uppercase; letter-spacing: .06em; color: var(--text-faint);
}
.ft-list { border: 1px solid var(--border); border-radius: var(--radius-s); }
.ft-list .result { padding: 8px 11px; }
.ft-list .result.ft-chosen { background: var(--accent-dim); }
.ft-hub input[type="search"] { flex: 1; min-width: 190px; }
.ft-hub select { width: auto; }
.ft-tags { gap: 5px; margin-top: 6px; }
.ft-pulls { margin: 12px 0; }
.ft-dl {
  border: 1px solid var(--border); border-radius: var(--radius-s);
  padding: 10px 12px; margin-top: 8px;
}
.ft-dl .dl-head { display: flex; align-items: center; gap: 9px; margin-bottom: 7px; }
.ft-dl .dl-id { font-family: var(--mono); font-size: 12.5px; font-weight: 560; }
.ft-dl .dl-nums {
  display: flex; flex-wrap: wrap; gap: 12px; margin-top: 6px;
  font-family: var(--mono); font-size: 11px; color: var(--text-dim);
}
.ft-dl .dl-nums b { color: var(--text); font-weight: 600; }
.ft-dl .logbox { max-height: 200px; margin-top: 8px; }
.ft-kv {
  display: grid; grid-template-columns: max-content minmax(0, 1fr);
  gap: 5px 16px; margin: 0; font-size: 12.5px;
}
.ft-kv dt { color: var(--text-faint); }
.ft-kv dd { margin: 0; font-family: var(--mono); font-size: 12px; overflow-wrap: anywhere; }
.ft-files {
  max-height: 220px; overflow: auto;
  border: 1px solid var(--border); border-radius: var(--radius-s);
}
`;
