/* Playground: chat against any running engine with the full parameter surface.

   Every control is built from GET /api/chat/params, which is generated from the
   target server's own OpenAPI schema — nothing here hard-codes a field name.
   That matters more than it sounds: vLLM 0.24 replaced guided_json and friends
   with a single structured_outputs object, and the request model accepts unknown
   keys silently, so a UI that emitted the old names would look like it worked
   and would quietly generate unconstrained text. */

import {
  h, mount, panel, empty, spinner, field, badge, toast, modal, confirmDialog,
  copyButton, debounce, duration, when, ensureStyles,
} from '../ui.js';
import { get, post, put, del, postStream } from '../api.js';

const ENDPOINT_POLL_MS = 15000;

/* /api/chat/params reports stream_options as transport-owned and never says
   whether the engine declares it, so the toggle is the escape hatch for an
   endpoint that rejects it rather than a capability read from the schema. */
const USAGE_HINT = 'Adds stream_options.include_usage so the engine reports real token counts';

/* Legacy structured-output names, folded into structured_outputs on load so a
   preset saved against an older engine keeps constraining the output. */
const LEGACY_STRUCTURED = {
  guided_json: 'json',
  guided_regex: 'regex',
  guided_choice: 'choice',
  guided_grammar: 'grammar',
};

const state = {
  endpoints: [],
  endpointId: '',
  model: '',
  spec: null,
  values: {},
  system: '',
  messages: [],
  chatId: null,
  title: '',
  includeUsage: true,
  filter: '',
  advancedOpen: false,
  presets: [],
  presetId: '',
  busy: false,
};

let els = {};
let controller = null;
let poll = null;
let live = null;

/* --- helpers ------------------------------------------------------------ */

const titleCase = (name) =>
  name.replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase());

const isEmptyValue = (value) =>
  value === undefined
  || value === null
  || value === ''
  || (Array.isArray(value) && value.length === 0)
  || (typeof value === 'object' && !Array.isArray(value) && Object.keys(value).length === 0);

function specFields() {
  if (!state.spec) return [];
  return [...state.spec.featured.flatMap((section) => section.fields), ...state.spec.advanced];
}

function knownNames() {
  return new Set(specFields().map((f) => f.name));
}

function fieldKind(f) {
  if (f.name === 'structured_outputs' && f.structured_options) return 'structured';
  if (f.enum) return 'enum';
  if (f.type === 'boolean') return 'bool';
  if (Array.isArray(f.default) || f.type === 'array') {
    return f.items === 'json' || f.items === 'object' ? 'json' : 'array';
  }
  if (f.type === 'object' || f.type === 'json') return 'json';
  // Only the curated hints carry usable bounds; the schema's own minimum for
  // seed and truncate_prompt_tokens is the int64 extreme, which is not a slider.
  if (f.type === 'number' || f.type === 'integer') {
    return Number.isFinite(f.min) && Number.isFinite(f.max) ? 'slider' : 'number';
  }
  return 'text';
}

function helpFor(f) {
  const parts = [];
  if (f.description) parts.push(f.description);
  if (f.default !== undefined) parts.push(`default ${JSON.stringify(f.default)}`);
  else if (f.suggested !== undefined) parts.push(`try ${f.suggested}`);
  return parts.join(' · ');
}

function migrate(values) {
  const out = {};
  const structured = { ...(values.structured_outputs || {}) };
  let moved = 0;
  for (const [key, value] of Object.entries(values)) {
    if (key in LEGACY_STRUCTURED) {
      if (!isEmptyValue(value)) {
        structured[LEGACY_STRUCTURED[key]] = value;
        moved += 1;
      }
      continue;
    }
    if (key !== 'structured_outputs') out[key] = value;
  }
  if (!isEmptyValue(structured)) out.structured_outputs = structured;
  if (moved) {
    toast(`Moved ${moved} legacy guided_* field into structured_outputs — this engine ignores the old names.`,
      { level: 'warn', title: 'Preset migrated' });
  }
  return out;
}

/** Drop anything this engine's schema does not declare: the server would accept
 *  it with a 200 and silently do nothing, which is worse than losing it here. */
function pruneUnknown(values) {
  const known = knownNames();
  const out = {};
  const dropped = [];
  for (const [key, value] of Object.entries(values)) {
    if (known.has(key)) out[key] = value;
    else dropped.push(key);
  }
  if (dropped.length) {
    toast(`${dropped.join(', ')} — not accepted by this endpoint, dropped.`, { level: 'warn' });
  }
  return out;
}

function currentEndpoint() {
  return state.endpoints.find((e) => e.id === state.endpointId) || null;
}

/* --- request assembly --------------------------------------------------- */

function historyMessages({ draft = '' } = {}) {
  const out = [];
  if (state.system.trim()) out.push({ role: 'system', content: state.system });
  for (const message of state.messages) {
    if (message.streaming) continue;
    out.push({ role: message.role, content: message.content });
  }
  if (draft.trim()) out.push({ role: 'user', content: draft });
  return out;
}

function buildBody({ draft = '', stream = true } = {}) {
  const body = { model: state.model, messages: historyMessages({ draft }) };
  for (const [key, value] of Object.entries(state.values)) {
    if (!isEmptyValue(value)) body[key] = value;
  }
  if (stream) {
    body.stream = true;
    if (state.includeUsage) body.stream_options = { include_usage: true };
  }
  return body;
}

const shellQuote = (text) => `'${String(text).replace(/'/g, `'\\''`)}'`;

function curlFor(body) {
  const endpoint = currentEndpoint();
  const url = `${endpoint ? endpoint.url : 'http://127.0.0.1:8000'}/v1/chat/completions`;
  return [
    `curl -N ${url} \\`,
    "  -H 'Content-Type: application/json' \\",
    `  -d ${shellQuote(JSON.stringify(body))}`,
  ].join('\n');
}

/* --- parameter controls -------------------------------------------------- */

function setValue(name, value) {
  if (isEmptyValue(value)) delete state.values[name];
  else state.values[name] = value;
  refreshRaw();
}

function clearButton(name, onDone) {
  return h('button', {
    class: 'pg-clear',
    title: 'Send the engine default instead',
    onClick: () => { delete state.values[name]; refreshRaw(); onDone(); },
  }, '✕');
}

function control(f, rerender) {
  const kind = fieldKind(f);
  const value = state.values[f.name];

  if (kind === 'bool') {
    return h('input', {
      type: 'checkbox',
      checked: value === undefined ? Boolean(f.default) : Boolean(value),
      onChange: (event) => { setValue(f.name, event.currentTarget.checked); rerender(); },
    });
  }

  if (kind === 'enum') {
    return h('select', {
      onChange: (event) => { setValue(f.name, event.currentTarget.value); rerender(); },
    },
    h('option', { value: '', selected: value === undefined }, '— engine default —'),
    f.enum.map((option) =>
      h('option', { value: option, selected: value === option }, String(option))));
  }

  if (kind === 'slider') {
    const start = f.suggested ?? f.default ?? f.min;
    const number = h('input', {
      type: 'number', min: f.min, max: f.max, step: f.step ?? 'any',
      value: value === undefined ? '' : value,
      placeholder: String(start),
    });
    const range = h('input', {
      type: 'range', min: f.min, max: f.max, step: f.step ?? 'any',
      value: value === undefined ? start : value,
    });
    const commit = (raw) => {
      const parsed = raw === '' ? undefined : Number(raw);
      setValue(f.name, Number.isFinite(parsed) ? parsed : undefined);
      rerender();
    };
    range.addEventListener('input', () => { number.value = range.value; });
    range.addEventListener('change', () => commit(range.value));
    number.addEventListener('change', () => {
      if (number.value !== '') range.value = number.value;
      commit(number.value);
    });
    return h('div', { class: 'field-row' }, range, number);
  }

  if (kind === 'number') {
    return h('input', {
      type: 'number',
      step: f.type === 'integer' ? 1 : 'any',
      value: value === undefined ? '' : value,
      placeholder: f.default === undefined ? 'engine default' : String(f.default),
      onChange: (event) => {
        const raw = event.currentTarget.value;
        setValue(f.name, raw === '' ? undefined : Number(raw));
        rerender();
      },
    });
  }

  if (kind === 'array') {
    return tagEntry(f, rerender);
  }

  if (kind === 'json') {
    return jsonEntry(f, rerender);
  }

  if (kind === 'structured') {
    return structuredEntry(f, rerender);
  }

  return h('input', {
    type: 'text',
    value: value === undefined ? '' : String(value),
    placeholder: f.default === undefined ? 'engine default' : String(f.default),
    onChange: (event) => { setValue(f.name, event.currentTarget.value); rerender(); },
  });
}

function tagEntry(f, rerender, { path = null } = {}) {
  // `stop` and friends are string-or-array on the wire, so a saved preset can
  // legitimately hold a bare string where this control wants a list.
  const asList = (value) => (Array.isArray(value) ? value : isEmptyValue(value) ? [] : [value]);
  const readList = () => asList(path ? state.values[f.name]?.[path] : state.values[f.name]);
  const writeList = (list) => {
    if (path) setSub(f.name, path, list);
    else setValue(f.name, list);
    rerender();
  };
  const numeric = f.items === 'integer' || f.items === 'number';
  const input = h('input', {
    type: 'text',
    placeholder: numeric ? 'token id, then Enter' : 'value, then Enter',
    onKeyDown: (event) => {
      if (event.key !== 'Enter') return;
      event.preventDefault();
      const raw = event.currentTarget.value.trim();
      if (!raw) return;
      const item = numeric ? Number(raw) : raw;
      if (numeric && !Number.isFinite(item)) {
        toast(`${raw} is not a number`, { level: 'warn' });
        return;
      }
      writeList([...readList(), item]);
    },
  });
  const tags = readList().map((item, index) =>
    h('span', { class: 'tag' }, String(item),
      h('button', {
        title: 'Remove',
        onClick: () => writeList(readList().filter((_, i) => i !== index)),
      }, '✕')));
  return h('div', { class: 'stack' }, tags.length ? h('div', { class: 'pg-tags' }, tags) : null, input);
}

function jsonEntry(f, rerender, { path = null, rows = 4 } = {}) {
  const value = path ? state.values[f.name]?.[path] : state.values[f.name];
  const message = h('span', { class: 'pg-err' });
  const area = h('textarea', {
    rows,
    value: value === undefined ? '' : JSON.stringify(value, null, 2),
    placeholder: Array.isArray(f.default) || f.items ? '[]' : '{}',
    onInput: (event) => {
      const raw = event.currentTarget.value.trim();
      event.currentTarget.classList.remove('pg-bad');
      message.textContent = '';
      if (!raw) {
        if (path) setSub(f.name, path, undefined); else setValue(f.name, undefined);
        return;
      }
      try {
        const parsed = JSON.parse(raw);
        if (path) setSub(f.name, path, parsed); else setValue(f.name, parsed);
      } catch (error) {
        event.currentTarget.classList.add('pg-bad');
        message.textContent = error.message;
      }
    },
    onChange: rerender,
  });
  return h('div', { class: 'stack' }, area, message);
}

function setSub(name, key, value) {
  const next = { ...(state.values[name] || {}) };
  if (isEmptyValue(value)) delete next[key];
  else next[key] = value;
  setValue(name, next);
}

/** structured_outputs is one object with mutually-informative members, so it
 *  gets a sub-form built from the engine's StructuredOutputsParams schema. */
function structuredEntry(f, rerender) {
  const options = Object.entries(f.structured_options)
    .filter(([key]) => !key.startsWith('_'));
  const current = state.values[f.name] || {};

  const rows = options.map(([key, meta]) => {
    let widget;
    if (meta.type === 'boolean') {
      widget = h('input', {
        type: 'checkbox',
        checked: Boolean(current[key]),
        onChange: (event) => { setSub(f.name, key, event.currentTarget.checked || undefined); rerender(); },
      });
    } else if (meta.type === 'array') {
      widget = tagEntry({ name: f.name, type: meta.type, items: meta.items }, rerender, { path: key });
    } else if (key === 'json') {
      // The engine takes either a JSON Schema object or its string form; keep
      // whatever the user typed if it does not parse, rather than dropping it.
      widget = h('textarea', {
        rows: 6,
        value: current.json === undefined
          ? ''
          : (typeof current.json === 'string' ? current.json : JSON.stringify(current.json, null, 2)),
        placeholder: '{"type": "object", "properties": {…}}',
        onChange: (event) => {
          const raw = event.currentTarget.value.trim();
          let parsed = raw;
          try { parsed = JSON.parse(raw); } catch { /* send the raw string */ }
          setSub(f.name, key, raw ? parsed : undefined);
          rerender();
        },
      });
    } else {
      widget = h('input', {
        type: 'text',
        value: current[key] === undefined ? '' : String(current[key]),
        onChange: (event) => { setSub(f.name, key, event.currentTarget.value || undefined); rerender(); },
      });
    }
    return field(key, widget, { inline: meta.type === 'boolean', changed: current[key] !== undefined });
  });

  return h('div', { class: 'pg-so' },
    h('p', { class: 'faint small pg-so-hint' },
      'Set exactly one of json, regex, choice, grammar or json_object.'),
    rows);
}

function paramRow(f, rerender) {
  const isSet = state.values[f.name] !== undefined;
  const kind = fieldKind(f);
  const row = field(titleCase(f.name), control(f, rerender), {
    help: helpFor(f),
    flag: f.name,
    inline: kind === 'bool',
    changed: isSet,
  });
  row.classList.add('pg-param');
  if (isSet) row.querySelector('label').append(clearButton(f.name, rerender));
  return row;
}

function matchesFilter(f) {
  if (!state.filter) return true;
  const needle = state.filter.toLowerCase();
  return f.name.toLowerCase().includes(needle)
    || String(f.description || '').toLowerCase().includes(needle);
}

function renderParams() {
  const box = els.params;
  if (!box) return;
  if (!state.endpointId) {
    return mount(box, empty('No engine selected', 'Start a server on the Serve tab, then come back.'));
  }
  if (!state.spec) return mount(box, h('div', { class: 'row' }, spinner(), 'Reading the engine schema…'));

  // Every edit rebuilds this panel, so the two bits of pure view state that a
  // rebuild would otherwise throw away are carried across by hand.
  const scroller = box.parentElement;
  const scrollTop = scroller ? scroller.scrollTop : 0;
  const rerender = () => { renderParams(); };

  const sections = state.spec.featured
    .map((section) => ({ ...section, fields: section.fields.filter(matchesFilter) }))
    .filter((section) => section.fields.length);
  const advanced = state.spec.advanced.filter(matchesFilter);

  mount(box,
    sections.map((section) =>
      h('div', { class: 'param-section' },
        h('h3', null, section.title),
        h('div', null, section.fields.map((f) => paramRow(f, rerender))))),
    advanced.length
      ? h('details', {
          class: 'collapse',
          open: state.advancedOpen,
          onToggle: (event) => { state.advancedOpen = event.currentTarget.open; },
        },
        h('summary', null, `All parameters (${advanced.length})`),
        advanced.map((f) => paramRow(f, rerender)))
      : null,
    !sections.length && !advanced.length
      ? empty('Nothing matches', `No parameter name or description contains “${state.filter}”.`)
      : null);

  if (scroller) scroller.scrollTop = scrollTop;
}

/* --- raw request drawer -------------------------------------------------- */

function refreshRaw() {
  if (!els.raw) return;
  const body = buildBody({ draft: els.composer ? els.composer.value : '' });
  const json = JSON.stringify(body, null, 2);
  mount(els.raw,
    h('div', { class: 'row wrap pg-raw-head' },
      h('span', { class: 'faint small' },
        `${Object.keys(body).length} keys · ${body.messages.length} messages`),
      h('span', { class: 'spacer' }),
      copyButton(json, 'Copy JSON'),
      copyButton(() => curlFor(body), 'Copy curl')),
    h('div', { class: 'cmdbox pg-raw' }, json),
    h('p', { class: 'faint small pg-raw-cap' },
      'Straight at the engine, bypassing this dashboard:'),
    h('div', { class: 'cmdbox pg-raw' }, curlFor(body)));
}

/* --- chat transcript ----------------------------------------------------- */

function messageActions(index, message) {
  const buttons = [];
  if (message.role === 'assistant' && !message.streaming) {
    buttons.push(copyButton(() => message.content, 'Copy'));
    buttons.push(h('button', {
      class: 'btn-sm btn-ghost',
      title: 'Discard this reply and ask again',
      onClick: () => regenerate(index),
    }, 'Retry'));
  }
  if (!message.streaming) {
    buttons.push(h('button', {
      class: 'btn-sm btn-ghost',
      onClick: () => { message.editing = true; renderChat(); },
    }, 'Edit'));
    buttons.push(h('button', {
      class: 'btn-sm btn-ghost',
      onClick: async () => {
        if (!await confirmDialog('Delete message', 'Remove this turn from the conversation?',
          { confirmLabel: 'Delete' })) return;
        state.messages.splice(index, 1);
        renderChat();
        refreshRaw();
      },
    }, 'Delete'));
  }
  return h('div', { class: 'pg-msg-actions' }, buttons);
}

function statsMeta(stats) {
  if (!stats) return null;
  const cells = [];
  if (Number.isFinite(stats.ttft)) {
    cells.push(h('span', { title: 'Time to first token' }, `TTFT ${stats.ttft.toFixed(2)}s`));
  }
  if (Number.isFinite(stats.total)) {
    // duration() rounds to whole seconds, which reads as "0s" for a short reply.
    cells.push(h('span', { title: 'Wall clock' },
      stats.total < 10 ? `${stats.total.toFixed(2)}s` : duration(stats.total)));
  }
  if (Number.isFinite(stats.tokens)) {
    const how = stats.approx
      ? 'Counted stream chunks — the engine sent no usage block'
      : 'From the engine\'s usage block';
    cells.push(h('span', { title: how }, `${stats.approx ? '~' : ''}${stats.tokens} tok`));
  }
  if (Number.isFinite(stats.tps)) {
    cells.push(h('span', { title: 'Decode rate, excluding time to first token' },
      `${stats.tps.toFixed(1)} tok/s`));
  }
  if (stats.prompt) cells.push(h('span', { title: 'Prompt tokens' }, `${stats.prompt} in`));
  if (stats.finish) cells.push(h('span', { class: 'faint' }, stats.finish));
  return cells;
}

function renderMessage(message, index) {
  if (message.editing) {
    const area = h('textarea', { rows: 6, value: message.content });
    return h('div', { class: `msg ${message.role}` },
      h('div', { class: 'who' }, message.role),
      h('div', { class: 'pg-body stack' }, area,
        h('div', { class: 'row' },
          h('button', {
            class: 'btn-sm btn-primary',
            onClick: () => {
              message.content = area.value;
              message.editing = false;
              renderChat();
              refreshRaw();
            },
          }, 'Save'),
          h('button', {
            class: 'btn-sm',
            onClick: () => { message.editing = false; renderChat(); },
          }, 'Cancel'))));
  }

  const bubble = h('div', { class: 'bubble' });
  const refs = {};

  if (message.role === 'assistant') {
    refs.reasoningText = h('div', { class: 'reasoning' }, message.reasoning || '');
    refs.summary = h('summary', null, reasoningLabel(message));
    refs.reasoning = h('details', {
      class: 'collapse',
      style: { display: message.reasoning ? 'block' : 'none' },
    }, refs.summary, refs.reasoningText);
    bubble.append(refs.reasoning);
  }

  refs.content = h('span', null, message.content || '');
  bubble.append(refs.content);
  if (message.streaming && !message.content) bubble.append(spinner());

  const extras = Object.entries(message.extra || {});
  if (extras.length) {
    bubble.append(...extras.map(([index2, text]) =>
      h('div', { class: 'pg-choice' }, h('span', { class: 'tag' }, `choice ${index2}`), ' ', text)));
  }

  refs.meta = h('div', { class: 'msg-meta' }, statsMeta(message.stats), messageActions(index, message));

  if (message.streaming) live = refs;

  return h('div', { class: `msg ${message.role}` },
    h('div', { class: 'who' }, message.role),
    h('div', { class: 'pg-body' }, bubble, refs.meta));
}

const reasoningLabel = (message) =>
  message.streaming ? 'Thinking…' : `Reasoning · ${message.reasoning.length} chars`;

function renderChat() {
  live = null;
  const log = els.chat;
  if (!log) return;
  if (!state.messages.length) {
    return mount(log, empty('No turns yet', 'Type below and hit Send, or load a saved conversation.'));
  }
  mount(log, h('div', { class: 'chat-log' }, state.messages.map(renderMessage)));
  log.scrollTop = log.scrollHeight;
}

function pinned() {
  const log = els.chat;
  return Boolean(log) && log.scrollHeight - log.scrollTop - log.clientHeight < 80;
}

/* --- streaming ----------------------------------------------------------- */

function frameError(payload) {
  // The proxy wraps the engine's error body as a string; unwrap it when it is
  // itself JSON so the user sees "The model X does not exist", not a blob.
  try {
    const inner = JSON.parse(payload);
    return inner?.error?.message || inner?.message || payload;
  } catch {
    return payload;
  }
}

async function runCompletion() {
  const endpoint = currentEndpoint();
  if (!endpoint) return;

  const body = buildBody({ stream: true });
  const message = { role: 'assistant', content: '', reasoning: '', extra: {}, streaming: true, stats: {} };
  state.messages.push(message);
  state.busy = true;
  updateControls();
  renderChat();
  refreshRaw();

  controller = new AbortController();
  const started = performance.now();
  let firstToken = null;
  let deltas = 0;
  let usage = null;
  let finish = '';

  try {
    for await (const data of postStream('/chat/completions',
      { endpoint_id: state.endpointId, path: '/v1/chat/completions', body }, controller.signal)) {
      if (data === '[DONE]') break;
      let frame;
      try { frame = JSON.parse(data); } catch { continue; }
      if (frame.error) throw new Error(frameError(frame.error));
      if (frame.usage) usage = frame.usage;

      for (const choice of frame.choices || []) {
        if (choice.finish_reason) finish = choice.finish_reason;
        const delta = choice.delta || {};
        // vLLM 0.24 with --reasoning-parser qwen3 emits `reasoning`; other
        // builds and other engines emit `reasoning_content`.
        const thinking = delta.reasoning_content ?? delta.reasoning;
        const text = delta.content;
        if (thinking === undefined && (text === undefined || text === '')) continue;
        if (firstToken === null) firstToken = performance.now();
        deltas += 1;

        if (choice.index) {
          message.extra[choice.index] = (message.extra[choice.index] || '') + (text || '');
          continue;
        }
        if (thinking) {
          message.reasoning += thinking;
          if (live) {
            live.reasoning.style.display = 'block';
            live.reasoningText.textContent = message.reasoning;
            live.summary.textContent = reasoningLabel(message);
          }
        }
        if (text) {
          message.content += text;
          if (live) live.content.textContent = message.content;
        }
      }
      if (live && pinned()) els.chat.scrollTop = els.chat.scrollHeight;
    }
  } catch (error) {
    if (error.name === 'AbortError') message.stats.finish = 'stopped';
    else {
      message.stats.finish = 'error';
      toast(error.message, { level: 'danger', title: 'Generation failed' });
    }
  } finally {
    const total = (performance.now() - started) / 1000;
    const ttft = firstToken === null ? undefined : (firstToken - started) / 1000;
    const tokens = usage?.completion_tokens ?? (deltas || undefined);
    const decode = ttft === undefined ? total : total - ttft;
    message.stats = {
      ...message.stats,
      ttft,
      total,
      tokens,
      approx: !usage,
      prompt: usage?.prompt_tokens,
      tps: tokens && decode > 0 ? tokens / decode : undefined,
      finish: message.stats.finish || finish,
    };
    message.streaming = false;
    controller = null;
    state.busy = false;
    live = null;
    updateControls();
    renderChat();
    refreshRaw();
  }
}

function send() {
  const draft = els.composer.value.trim();
  if (!draft && state.messages.at(-1)?.role !== 'user') return;
  if (draft) {
    state.messages.push({ role: 'user', content: draft });
    els.composer.value = '';
  }
  runCompletion();
}

function regenerate(index) {
  state.messages.splice(index);
  renderChat();
  runCompletion();
}

function stop() {
  if (controller) controller.abort();
}

/* --- persistence --------------------------------------------------------- */

async function loadPresets() {
  try {
    const payload = await get('/chat/presets?kind=sampling');
    state.presets = payload.presets;
  } catch (error) {
    toast(error.message, { level: 'danger' });
  }
  renderPresets();
}

function selectedPreset() {
  return state.presets.find((preset) => String(preset.id) === state.presetId) || null;
}

function renderPresets() {
  if (!els.presets) return;
  mount(els.presets,
    h('select', {
      onChange: (event) => {
        state.presetId = event.currentTarget.value;
        const preset = selectedPreset();
        if (!preset) return;
        state.values = pruneUnknown(migrate(preset.data || {}));
        renderParams();
        refreshRaw();
      },
    },
    h('option', { value: '', selected: !state.presetId },
      state.presets.length ? '— apply a preset —' : 'no presets saved'),
    state.presets.map((preset) =>
      h('option', { value: String(preset.id), selected: String(preset.id) === state.presetId }, preset.name))),
    h('button', { class: 'btn-sm', onClick: savePresetDialog }, 'Save as…'),
    h('button', {
      class: 'btn-sm btn-ghost',
      disabled: !selectedPreset(),
      onClick: deleteSelectedPreset,
    }, 'Delete'));
}

function savePresetDialog() {
  // The API upserts on (kind, name), so pre-filling the current preset makes
  // "tweak and re-save" the default gesture rather than a duplicate.
  const input = h('input', {
    type: 'text', placeholder: 'e.g. deterministic', value: selectedPreset()?.name || '',
  });
  const ctl = modal('Save sampling preset',
    h('div', { class: 'stack' },
      h('p', { class: 'muted small' },
        `${Object.keys(state.values).length} parameter(s) will be stored under this name.`),
      input),
    {
      actions: [
        h('button', { onClick: () => ctl.close() }, 'Cancel'),
        h('button', {
          class: 'btn-primary',
          onClick: async () => {
            const name = input.value.trim();
            if (!name) return;
            ctl.close();
            try {
              await post('/chat/presets', { kind: 'sampling', name, data: state.values });
              toast(`Preset “${name}” saved`);
              await loadPresets();
            } catch (error) {
              toast(error.message, { level: 'danger' });
            }
          },
        }, 'Save'),
      ],
    });
  input.focus();
}

async function deleteSelectedPreset() {
  const preset = selectedPreset();
  if (!preset) return;
  if (!await confirmDialog('Delete preset', `Delete “${preset.name}”? This cannot be undone.`,
    { confirmLabel: 'Delete' })) return;
  try {
    await del(`/chat/presets/${preset.id}`);
    state.presetId = '';
    await loadPresets();
  } catch (error) {
    toast(error.message, { level: 'danger' });
  }
}

function conversationPayload() {
  const first = state.messages.find((m) => m.role === 'user');
  return {
    title: state.title || (first ? first.content.slice(0, 60) : 'Untitled'),
    endpoint: state.endpointId,
    model: state.model,
    params: state.values,
    messages: historyMessages(),
  };
}

async function saveConversation() {
  const payload = conversationPayload();
  try {
    if (state.chatId) await put(`/chat/conversations/${state.chatId}`, payload);
    else state.chatId = (await post('/chat/conversations', payload)).id;
    state.title = payload.title;
    toast(`Saved as “${payload.title}”`);
    updateControls();
  } catch (error) {
    toast(error.message, { level: 'danger' });
  }
}

async function openConversations() {
  let rows = [];
  try {
    rows = (await get('/chat/conversations')).conversations;
  } catch (error) {
    toast(error.message, { level: 'danger' });
    return;
  }

  const body = h('div');
  const ctl = modal('Saved conversations', body, { wide: true });

  const draw = () => {
    if (!rows.length) {
      return mount(body, empty('Nothing saved yet', 'Use Save on a conversation to keep it here.'));
    }
    mount(body, h('div', { class: 'table-wrap' },
      h('table', null,
        h('thead', null, h('tr', null,
          h('th', null, 'Title'), h('th', null, 'Model'), h('th', null, 'Updated'), h('th', null, ''))),
        h('tbody', null, rows.map((row) =>
          h('tr', null,
            h('td', null, row.title),
            h('td', { class: 'mono' }, row.model || '—'),
            h('td', { class: 'faint small' }, when(row.updated_at)),
            h('td', { class: 'right' },
              h('button', {
                class: 'btn-sm',
                onClick: async () => {
                  ctl.close();
                  await loadConversation(row.id);
                },
              }, 'Load'),
              h('button', {
                class: 'btn-sm btn-ghost',
                onClick: async () => {
                  if (!await confirmDialog('Delete conversation', `Delete “${row.title}”?`,
                    { confirmLabel: 'Delete' })) return;
                  try {
                    await del(`/chat/conversations/${row.id}`);
                    rows = rows.filter((r) => r.id !== row.id);
                    if (state.chatId === row.id) state.chatId = null;
                    draw();
                  } catch (error) {
                    toast(error.message, { level: 'danger' });
                  }
                },
              }, 'Delete'))))))));
  };
  draw();
}

async function loadConversation(chatId) {
  try {
    const row = await get(`/chat/conversations/${chatId}`);
    const messages = [...(row.messages || [])];
    state.system = messages[0]?.role === 'system' ? messages.shift().content : '';
    state.messages = messages.map((m) => ({ role: m.role, content: m.content || '', reasoning: '', extra: {} }));
    state.values = pruneUnknown(migrate(row.params || {}));
    state.chatId = row.id;
    state.title = row.title;
    if (row.model && currentEndpoint()?.models.includes(row.model)) state.model = row.model;
    if (els.system) els.system.value = state.system;
    renderChat();
    renderParams();
    updateControls();
    refreshRaw();
  } catch (error) {
    toast(error.message, { level: 'danger' });
  }
}

function newConversation() {
  state.messages = [];
  state.chatId = null;
  state.title = '';
  renderChat();
  updateControls();
  refreshRaw();
}

/* --- endpoints ----------------------------------------------------------- */

async function loadSpec() {
  state.spec = null;
  renderParams();
  try {
    state.spec = await get(`/chat/params?endpoint_id=${encodeURIComponent(state.endpointId)}`);
    state.values = pruneUnknown(state.values);
  } catch (error) {
    toast(error.message, { level: 'danger' });
    state.spec = { source: '', featured: [], advanced: [] };
  }
  renderParams();
  renderSource();
  refreshRaw();
}

function renderSource() {
  if (!els.source) return;
  const source = state.spec?.source || '';
  mount(els.source,
    source
      ? h('span', { class: 'faint small' },
          source === 'bundled snapshot'
            ? 'Controls come from the bundled schema snapshot — this endpoint did not serve /openapi.json.'
            : `Controls generated from ${source}/openapi.json`)
      : null);
}

async function refreshEndpoints({ initial = false } = {}) {
  let found = [];
  try {
    found = (await get('/chat/endpoints')).endpoints;
  } catch (error) {
    if (initial) toast(error.message, { level: 'danger' });
    return;
  }
  const changed = JSON.stringify(found) !== JSON.stringify(state.endpoints);
  state.endpoints = found;
  if (!changed && !initial) return;

  if (!found.some((e) => e.id === state.endpointId)) {
    state.endpointId = found[0]?.id || '';
    state.model = found[0]?.models[0] || '';
    renderEndpointPicker();
    updateControls();
    if (state.endpointId) await loadSpec();
    else { state.spec = null; renderParams(); }
    return;
  }
  renderEndpointPicker();
  updateControls();
}

function renderEndpointPicker() {
  if (!els.picker) return;
  const endpoint = currentEndpoint();
  mount(els.picker,
    h('select', {
      title: 'Healthy engines discovered on this host',
      onChange: async (event) => {
        state.endpointId = event.currentTarget.value;
        state.model = currentEndpoint()?.models[0] || '';
        renderEndpointPicker();
        updateControls();
        await loadSpec();
      },
    },
    state.endpoints.length
      ? state.endpoints.map((item) =>
          h('option', { value: item.id, selected: item.id === state.endpointId },
            `${item.label} · ${item.url}`))
      : h('option', { value: '' }, 'no healthy engine')),
    h('select', {
      title: 'Model served by this engine',
      onChange: (event) => { state.model = event.currentTarget.value; refreshRaw(); },
    },
    (endpoint?.models || []).map((name) =>
      h('option', { value: name, selected: name === state.model }, name))),
    endpoint ? badge(endpoint.managed ? 'running' : 'info', endpoint.managed ? 'managed' : 'foreign') : null,
    h('button', {
      class: 'btn-sm btn-ghost',
      onClick: () => refreshEndpoints({ initial: true }),
    }, 'Refresh'));
}

function updateControls() {
  if (!els.send) return;
  const ready = Boolean(state.endpointId) && !state.busy;
  els.send.disabled = !ready;
  els.stop.disabled = !state.busy;
  els.composer.disabled = !state.endpointId;
  els.saveBtn.disabled = !state.messages.length;
  els.titleLabel.textContent = state.chatId ? `saved · ${state.title}` : 'unsaved';
}

/* --- view ---------------------------------------------------------------- */

export async function render(container, ctx) {
  ensureStyles('playground', CSS);
  Object.assign(state, {
    endpoints: [], endpointId: '', model: '', spec: null, values: {}, system: '',
    messages: [], chatId: null, title: '', filter: '', advancedOpen: false,
    presets: [], presetId: '', busy: false,
  });

  els = {
    picker: h('div', { class: 'pg-toolbar' }),
    source: h('div'),
    params: h('div'),
    search: h('input', {
      type: 'search', class: 'param-search', placeholder: 'Filter parameters…',
      onInput: debounce((event) => { state.filter = event.target.value.trim(); renderParams(); }, 180),
    }),
    presets: h('div', { class: 'row wrap' }),
    chat: h('div', { class: 'pg-log' }),
    raw: h('div'),
    composer: h('textarea', {
      placeholder: 'Message… (Ctrl+Enter sends)',
      onInput: debounce(refreshRaw, 250),
      onKeyDown: (event) => {
        if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
          event.preventDefault();
          send();
        }
      },
    }),
    system: h('textarea', {
      rows: 3,
      placeholder: 'You are a helpful assistant…',
      onInput: debounce((event) => { state.system = event.target.value; refreshRaw(); }, 250),
    }),
    send: h('button', { class: 'btn-primary', onClick: send }, 'Send'),
    stop: h('button', { class: 'btn-danger', onClick: stop, disabled: true }, 'Stop'),
    saveBtn: h('button', { class: 'btn-sm', onClick: saveConversation }, 'Save'),
    titleLabel: h('span', { class: 'faint small' }),
  };

  const usageToggle = h('input', {
    type: 'checkbox',
    checked: state.includeUsage,
    onChange: (event) => { state.includeUsage = event.currentTarget.checked; refreshRaw(); },
  });

  mount(container,
    h('div', { class: 'page-head' },
      h('div', null,
        h('h1', null, 'Playground'),
        h('p', null,
          'Talk to any healthy engine on this host. Controls are generated from that engine’s own '
          + 'schema, and only the parameters you touch are sent.')),
      h('div', { class: 'page-actions' },
        els.titleLabel,
        h('button', { class: 'btn-sm', onClick: openConversations }, 'Conversations'),
        els.saveBtn,
        h('button', { class: 'btn-sm btn-ghost', onClick: newConversation }, 'New'))),

    h('div', { class: 'pg-split' },
      h('div', { class: 'stack' },
        panel('Endpoint', { body: h('div', { class: 'stack' }, els.picker, els.source) }),
        panel('Conversation', {
          flush: true,
          sub: 'system prompt, then turns',
          body: h('div', null,
            h('details', { class: 'collapse pg-system' },
              h('summary', null, 'System prompt'),
              els.system),
            els.chat),
          foot: h('div', { class: 'composer' },
            els.composer,
            h('div', { class: 'row wrap' },
              els.send, els.stop,
              h('span', { class: 'spacer' }),
              h('label', { class: 'row small faint', title: USAGE_HINT },
                usageToggle, 'usage stats'))),
        })),

      h('div', { class: 'stack pg-side' },
        panel('Parameters', {
          sub: 'only what you set is sent',
          actions: h('button', {
            class: 'btn-sm btn-ghost',
            onClick: () => {
              state.values = {};
              state.presetId = '';
              renderPresets();
              renderParams();
              refreshRaw();
            },
          }, 'Reset all'),
          body: h('div', { class: 'stack' },
            els.presets,
            els.search,
            h('div', { class: 'pg-scroll' }, els.params)),
        }),
        panel('Raw request', { sub: 'exactly what will be posted', body: els.raw }))));

  renderChat();
  renderPresets();
  refreshRaw();
  await refreshEndpoints({ initial: true });
  await loadPresets();
  poll = setInterval(() => refreshEndpoints(), ENDPOINT_POLL_MS);

  const wanted = ctx.routeDetail();
  if (wanted) await loadConversation(Number(wanted));
}

export function dispose() {
  if (controller) controller.abort();
  controller = null;
  if (poll) clearInterval(poll);
  poll = null;
  live = null;
  els = {};
}

/* --- styles -------------------------------------------------------------- */

const CSS = `
.pg-split { display: grid; grid-template-columns: minmax(0,1fr) minmax(0,430px); gap: var(--gap); align-items: start; }
@media (max-width: 1180px) { .pg-split { grid-template-columns: minmax(0,1fr); } }
.pg-side { position: sticky; top: 62px; }
.pg-scroll { max-height: calc(100vh - 320px); overflow-y: auto; padding-right: 4px; }
.pg-log { max-height: calc(100vh - 400px); min-height: 200px; overflow-y: auto; padding: 10px 14px; }
.pg-body { flex: 1; min-width: 0; }
.pg-msg-actions { display: flex; gap: 2px; margin-left: auto; opacity: 0; transition: opacity .12s; }
.msg:hover .pg-msg-actions, .pg-msg-actions:focus-within { opacity: 1; }
.pg-toolbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.pg-toolbar select { width: auto; max-width: 320px; }
.pg-tags { display: flex; flex-wrap: wrap; gap: 5px; }
.pg-tags .tag { display: inline-flex; align-items: center; gap: 5px; }
.pg-tags .tag button { padding: 0 2px; border: 0; background: none; color: var(--text-faint); font-size: 11px; }
.pg-param > label { gap: 6px; }
.pg-param .flagname { margin-left: auto; }
.pg-clear { padding: 0 3px; border: 0; background: none; color: var(--text-faint); font-size: 12px; }
.pg-clear:hover:not(:disabled) { background: none; color: var(--danger); }
.pg-bad { border-color: var(--danger); }
.pg-err { color: var(--danger); font-size: 11px; }
.pg-so { border: 1px solid var(--border); border-radius: var(--radius-s); padding: 11px 11px 1px; }
.pg-so .field { margin-bottom: 10px; }
.pg-so-hint { margin: 0 0 10px; }
.pg-system { padding: 0 14px; }
.pg-raw-head { margin-bottom: 8px; }
.pg-raw-cap { margin: 10px 0 4px; }
.pg-choice { margin-top: 9px; padding-top: 7px; border-top: 1px dashed var(--border); }
.pg-raw { max-height: 340px; overflow: auto; }
.msg .bubble .collapse > summary { color: var(--info); }
`;
