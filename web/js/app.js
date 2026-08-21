/* Shell: tab routing, the always-on vitals readout, and theme.

   Views are ES modules under js/views/ exporting `render(container, ctx)` and
   optionally `dispose()`. They are imported on first visit, so a broken view
   cannot stop the rest of the dashboard from loading. */

import * as api from './api.js';
import { h, mount, bytes, pct, toast, notice } from './ui.js';

const TABS = [
  { id: 'overview',  label: 'Overview',   module: './views/overview.js' },
  { id: 'models',    label: 'Models',     module: './views/models.js' },
  { id: 'serve',     label: 'Serve',      module: './views/serve.js' },
  { id: 'playground',label: 'Playground', module: './views/playground.js' },
  { id: 'finetune',  label: 'Fine-tune',  module: './views/finetune.js' },
  { id: 'heretic',   label: 'Heretic',    module: './views/heretic.js' },
  { id: 'jobs',      label: 'Jobs',       module: './views/jobs.js' },
];

const state = {
  info: null,
  telemetry: null,
  current: null,
  view: null,
  badges: {},
};

/* --- routing ------------------------------------------------------------ */

const tabFromHash = () => {
  const id = location.hash.replace(/^#\/?/, '').split('/')[0];
  return TABS.find((tab) => tab.id === id) || TABS[0];
};

async function activate(tab, { replace = false } = {}) {
  if (state.view?.dispose) {
    try { state.view.dispose(); } catch (error) { console.error('dispose failed', error); }
  }
  state.view = null;
  state.current = tab.id;

  for (const button of document.querySelectorAll('.tab')) {
    button.setAttribute('aria-current', button.dataset.tab === tab.id ? 'page' : 'false');
  }
  if (!replace && location.hash.replace(/^#\/?/, '').split('/')[0] !== tab.id) {
    location.hash = `#/${tab.id}`;
  }

  const container = document.getElementById('view');
  mount(container, h('div', { class: 'row', style: { padding: '40px', justifyContent: 'center' } },
    h('span', { class: 'spinner' })));

  try {
    const module = await import(tab.module);
    if (state.current !== tab.id) return;
    state.view = module;
    mount(container);
    await module.render(container, ctx());
  } catch (error) {
    console.error(error);
    mount(container, notice('danger',
      h('strong', null, `The ${tab.label} view failed to load. `),
      h('span', null, String(error && error.message ? error.message : error))));
  }
}

/* --- context handed to every view --------------------------------------- */

function ctx() {
  return {
    api,
    info: state.info,
    telemetry: () => state.telemetry,
    setBadge(tabId, value) {
      state.badges[tabId] = value;
      renderTabs();
    },
    navigate(tabId, detail) {
      const tab = TABS.find((t) => t.id === tabId);
      if (!tab) return;
      location.hash = detail ? `#/${tabId}/${detail}` : `#/${tabId}`;
    },
    routeDetail: () => location.hash.replace(/^#\/?/, '').split('/').slice(1).join('/'),
  };
}

/* --- chrome ------------------------------------------------------------- */

function renderTabs() {
  const nav = document.getElementById('tabs');
  mount(nav, TABS.map((tab) => h('button', {
    class: 'tab',
    dataset: { tab: tab.id },
    'aria-current': state.current === tab.id ? 'page' : 'false',
    onClick: () => activate(tab),
  },
  tab.label,
  state.badges[tab.id] ? h('span', { class: 'count' }, String(state.badges[tab.id])) : null)));
}

function renderVitals(snapshot) {
  const host = document.getElementById('vitals');
  if (!snapshot) return mount(host);

  const memory = snapshot.memory || {};
  const used = memory.used_fraction || 0;
  const level = used > 0.92 ? 'crit' : used > 0.82 ? 'warn' : '';
  const gpu = snapshot.gpu || {};

  mount(host,
    h('span', { class: `vital ${level}`, title: `${bytes(memory.available_bytes)} available of ${bytes(memory.total_bytes)}` },
      'MEM', h('b', null, pct(used)),
      h('span', { class: 'faint' }, `${bytes(memory.available_bytes)} free`)),
    Number.isFinite(gpu.utilization_gpu)
      ? h('span', { class: 'vital', title: 'GPU utilisation' }, 'GPU', h('b', null, `${Math.round(gpu.utilization_gpu)}%`))
      : null,
    Number.isFinite(gpu.temperature_gpu)
      ? h('span', { class: 'vital', title: 'GPU temperature' }, h('b', null, `${Math.round(gpu.temperature_gpu)}°C`))
      : null,
    Number.isFinite(gpu.power_draw)
      ? h('span', { class: 'vital', title: 'Board power draw' }, h('b', null, `${Math.round(gpu.power_draw)}W`))
      : null);
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem('llmd.theme', theme); } catch { /* private mode */ }
}

/* --- boot --------------------------------------------------------------- */

async function boot() {
  try {
    const saved = localStorage.getItem('llmd.theme');
    if (saved) document.documentElement.dataset.theme = saved;
  } catch { /* ignore */ }

  document.getElementById('theme-toggle').addEventListener('click', () => {
    applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
  });

  renderTabs();
  window.addEventListener('hashchange', () => {
    const tab = tabFromHash();
    if (tab.id !== state.current) activate(tab, { replace: true });
  });

  try {
    state.info = await api.get('/system/info');
    document.getElementById('brand-host').textContent =
      `${state.info.hostname} · vLLM ${state.info.vllm_version}`;
  } catch (error) {
    toast(`Could not reach the API: ${error.message}`, { level: 'danger', title: 'Offline', timeout: 0 });
  }

  api.stream('/system/telemetry/stream', {
    telemetry: (snapshot) => { state.telemetry = snapshot; renderVitals(snapshot); },
    memguard: (event) => toast(event.reason, {
      level: 'danger',
      title: 'Memory watchdog fired',
      timeout: 0,
    }),
  });

  await activate(tabFromHash(), { replace: true });
}

boot();
