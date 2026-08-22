/* Overview — the landing page.

   It answers three questions in priority order: how much memory is left before
   the box locks up, what is holding that memory, and what is running. The
   memory panel comes first and is the largest element on purpose — on a
   unified-memory host every other failure is recoverable and that one is not.

   With a second machine in the registry that first question has one answer per
   node, so the panel is one card per node rather than one machine's figures. */

import { get, stream } from '../api.js';
import {
  ago, badge, bytes, copyButton, duration, empty, ensureStyles, h, mount, notice,
  panel, pct, spinner, stat, toast, when,
} from '../ui.js';

const POLL_MS = 10000;
const LIVE_SERVER = new Set(['running', 'loading', 'starting', 'unhealthy']);

let ctxRef = null;
let poll = null;
let closeStream = null;
let images = null;
let systemInfo = null;
let lastFailure = '';

const region = {};

const utilText = (value) => (Number.isFinite(value) ? value.toFixed(2) : '—');

const reading = (value, suffix) =>
  (Number.isFinite(value) ? `${Math.round(value)}${suffix}` : '—');

/* --- cluster memory ----------------------------------------------------- */

function memoryBars(segments, total) {
  const scale = total || 1;
  return h('div', { class: 'stack' },
    h('div', { class: 'ov-track' },
      segments.filter((seg) => seg.value > 0).map((seg) => h('span', {
        class: `ov-seg ov-c-${seg.key}`,
        style: { width: `${(seg.value / scale) * 100}%` },
        title: `${seg.label}: ${bytes(seg.value)}`,
      }))),
    h('div', { class: 'legend ov-legend' },
      segments.map((seg) => h('span', { title: seg.note },
        h('i', { class: `ov-c-${seg.key}` }),
        `${seg.label} · ${bytes(seg.value)}`))));
}

/* Only the machine this dashboard runs on can be broken down this far:
   nvidia-smi reports per-process memory for its own box, and the committed
   fractions come from containers we can inspect without ssh. */
function localMemory(budget) {
  const total = budget.total_bytes || 1;
  const engines = Math.max(0, budget.committed_bytes || 0);
  // occupied_bytes is max(committed, measured), so the excess is allocation by
  // something that never declared a util fraction — a trainer, a stray process.
  const other = Math.max(0, (budget.occupied_bytes || 0) - engines);
  const reserve = Math.min(budget.reserve_bytes || 0, total);
  const headroom = Math.max(0, total - engines - other - reserve);

  const segments = [
    {
      key: 'engine',
      label: 'Engines committed',
      value: engines,
      note: `${utilText(budget.committed_util)} util summed over ${budget.tenants.length} engine(s)`,
    },
    {
      key: 'other',
      label: 'Other GPU processes',
      value: other,
      note: 'measured allocation with no utilisation fraction behind it',
    },
    {
      key: 'free',
      label: 'Free to allocate',
      value: headroom,
      note: `the largest new --gpu-memory-utilization that fits is ${utilText(budget.free_util)}`,
    },
    {
      key: 'reserve',
      label: 'Held back for the OS',
      value: reserve,
      note: 'the dashboard refuses any launch that would eat into this',
    },
  ];

  const tight = budget.free_util <= 0.02;
  const warm = !tight && budget.occupied_util > budget.warn_util;

  return h('div', { class: 'stack' },
    h('div', { class: 'ov-tiles' },
      stat('Host memory', bytes(total), 'shared by the CPU and the GPU'),
      stat('Committed', utilText(budget.committed_util), bytes(budget.committed_bytes)),
      stat('Free to allocate', utilText(budget.free_util), `of a ${utilText(budget.max_util)} ceiling`),
      stat('MemAvailable', bytes(budget.available_bytes), `${bytes(budget.free_bytes)} actually free`)),
    memoryBars(segments, total),
    tight || warm
      ? notice(tight ? 'danger' : 'warn',
          h('strong', null, tight ? 'No room for another engine here. ' : 'Getting tight. '),
          h('span', null, tight
            ? `Everything above ${utilText(budget.max_util)} total utilisation is refused; stop a `
              + 'server before starting or fine-tuning anything, or place the next one on a peer.'
            : `Total utilisation has passed the ${utilText(budget.warn_util)} warning line. Keep `
              + '--max-num-seqs low; concurrency spikes are what actually kill this box.'))
      : null,
    budget.tenants.length
      ? h('div', { class: 'table-wrap' },
          h('table', null,
            h('thead', null, h('tr', null,
              h('th', null, 'Holding memory'),
              h('th', null, 'Origin'),
              h('th', { class: 'num' }, 'Util'),
              h('th', { class: 'num' }, 'Reserved'))),
            h('tbody', null, budget.tenants.map((tenant) => h('tr', null,
              h('td', null,
                h('span', { class: 'nowrap' }, tenant.name),
                tenant.note ? h('div', { class: 'ov-note' }, tenant.note) : null),
              h('td', null, badge(tenant.managed ? 'info' : 'absent',
                tenant.managed ? 'managed' : 'hand-launched')),
              h('td', { class: 'num' }, `${utilText(tenant.util)}${tenant.implicit ? ' *' : ''}`),
              h('td', { class: 'num' }, bytes(tenant.bytes_committed)))))))
      : h('p', { class: 'ov-note' }, 'No vLLM engine is holding a reservation right now.'));
}

/* A peer is read over ssh: /proc/meminfo and docker ps, nothing else. Used is
   whatever the kernel does not count as available, which is a coarser split
   than the local card and is labelled as such. */
function peerMemory(node) {
  const total = node.total_bytes || 0;
  const available = Math.max(0, node.available_bytes || 0);
  const used = Math.max(0, total - available);
  const running = node.containers || [];

  const segments = [
    {
      key: 'other',
      label: 'In use',
      value: used,
      note: 'everything MemAvailable does not count as reclaimable, engines included',
    },
    {
      key: 'free',
      label: 'Available',
      value: available,
      note: 'what a new engine on this node could take before the guard steps in',
    },
  ];

  return h('div', { class: 'stack' },
    h('div', { class: 'ov-tiles' },
      stat('Host memory', bytes(total), 'shared by the CPU and the GPU'),
      stat('In use', bytes(used), `${Math.round((used / (total || 1)) * 100)}% of the node`),
      stat('MemAvailable', bytes(available), `${bytes(node.free_bytes)} actually free`),
      stat('Containers', String(running.length), 'running under its docker daemon')),
    memoryBars(segments, total),
    running.length
      ? h('p', { class: 'ov-note' }, running.map((container) => container.name).join(', '))
      : h('p', { class: 'ov-note' }, 'Nothing is running on this node.'),
    h('p', { class: 'ov-note' },
      'No per-process GPU figure: nvidia-smi reports on the machine it runs on, and this peer is '
      + 'reached over ssh. What it has committed through vLLM flags is on the Serve tab, next to '
      + 'the servers placed there.'));
}

function nodeHeader(node) {
  const running = node.containers || [];
  // The local node's registry note is "this machine", which its badge already says.
  const meta = [node.address, node.docker ? `docker ${node.docker}` : null,
    node.local ? null : node.note].filter(Boolean).join(' · ');

  return h('div', { class: 'row wrap ov-node-head' },
    h('strong', null, node.name),
    badge(node.local ? 'info' : 'plain', node.local ? 'this machine' : 'peer'),
    badge(node.reachable ? 'running' : 'failed', node.reachable ? 'reachable' : 'unreachable'),
    node.reachable
      ? badge(node.has_nvidia_runtime ? 'succeeded' : 'failed',
          node.has_nvidia_runtime ? 'nvidia runtime' : 'no nvidia runtime')
      : null,
    node.reachable
      ? h('span', {
          class: 'badge plain',
          title: running.map((container) => container.name).join('\n'),
        }, `${running.length} container(s)`)
      : null,
    h('span', { class: 'spacer' }),
    meta ? h('span', { class: 'ov-note' }, meta) : null);
}

function nodeCard(node, budget) {
  const down = !node.reachable;
  return h('div', { class: `ov-node${down ? ' down' : ''}` },
    nodeHeader(node),
    down
      ? notice('danger',
          h('strong', null, 'Unreachable. '),
          h('span', null, node.error
            || 'Its docker daemon did not answer. Nothing can be placed here until it does.'))
      : null,
    // /proc/meminfo is read locally and stays readable when the docker socket is
    // not, so this card keeps its figures. A peer's numbers come over the same
    // ssh that just failed, and 0 GiB would be a lie rather than a measurement.
    node.local ? localMemory(budget) : (down ? null : peerMemory(node)));
}

/** The cluster as one pool. A pooled engine's memory is the sum of what each
 *  node can commit, so "will this model fit" is answered by the combined
 *  ceiling, not by any single machine's. Only shown once there is more than one
 *  node — on a single box it would just restate the card below it. */
function combinedPool(combined, registry) {
  if (!combined || registry.length < 2) return null;

  const total = combined.total_bytes || 1;
  const used = combined.used_bytes || 0;
  const perNode = combined.reserve_bytes_per_node || 0;

  return h('div', { class: 'ov-pool' },
    h('div', { class: 'ov-pool-head' },
      h('strong', null, 'Pooled across the cluster'),
      h('span', { class: 'faint small' },
        `${combined.nodes} node${combined.nodes === 1 ? '' : 's'}`
        + (combined.unreachable ? ` · ${combined.unreachable} unreachable, not counted` : ''))),
    h('div', { class: 'grid cols-4' },
      stat('Combined memory', bytes(total), `${bytes(combined.available_bytes)} available`),
      stat('One pooled engine', bytes(combined.pooled_ceiling_bytes),
        `each node's ceiling summed, less ${bytes(perNode)} held back per node`),
      stat('Largest single node', bytes(combined.single_node_ceiling_bytes),
        'what fits without pooling'),
      stat('In use', bytes(used), pct(used / total))),
    h('p', { class: 'help' },
      'Pooling splits a model by layer across these machines, so a model larger than any one '
      + 'of them still fits. It is not free: the engine spans every node in the pool, and if one '
      + 'leaves, the engine aborts and has to be relaunched.'));
}

function clusterSection(payload) {
  const registry = payload.nodes || [];
  if (!registry.length) {
    // /api/nodes sshes to every peer, so it is the call here most likely to hang
    // or fail, and it always names this machine when it does answer. Losing it
    // must not take the local figures down with it.
    return h('div', { class: 'stack' },
      notice('warn',
        h('strong', null, 'No node list. '),
        h('span', null, 'GET /api/nodes named no machines, which it cannot do while it is '
          + 'working. Only this one is shown until it answers again.')),
      localMemory(payload.budget));
  }
  const local = registry.find((node) => node.local) || {};

  return h('div', { class: 'stack' },
    combinedPool(payload.combined, registry),
    registry.map((node) => nodeCard(node, payload.budget)),
    notice('info',
      h('strong', null, 'GPU memory is host memory on these machines. '),
      h('span', null,
        '--gpu-memory-utilization 0.50 reserves half of all '
        + `${bytes(local.total_bytes || payload.budget.total_bytes)} on the node it runs on, so the `
        + 'fractions of every engine on that node add up, and overcommitting does not run slowly — '
        + 'it freezes that box while CUDA graphs are captured. The fractions do not travel: a '
        + 'peer\'s ceiling is its own.')));
}

/* --- live telemetry ----------------------------------------------------- */

function telemetrySection(snapshot) {
  if (!snapshot) {
    return empty('Waiting for a sample', 'The telemetry stream has not delivered a snapshot yet.');
  }
  const gpu = snapshot.gpu || {};
  const load = snapshot.load || [];
  const procs = snapshot.gpu_processes || [];
  const measured = procs.reduce((sum, proc) => sum + (proc.used_bytes || 0), 0);

  return h('div', { class: 'stack' },
    h('div', { class: 'ov-tiles' },
      stat('GPU', reading(gpu.utilization_gpu, '%'), gpu.name || 'accelerator'),
      stat('Temperature', reading(gpu.temperature_gpu, '°C'),
        gpu.pstate ? `pstate ${gpu.pstate}` : ''),
      stat('Power', reading(gpu.power_draw, ' W'), 'no power cap is readable here'),
      stat('SM clock', reading(gpu.clocks_sm, ' MHz'),
        gpu.driver_version ? `driver ${gpu.driver_version}` : ''),
      stat('Load', load.length ? load.map((value) => value.toFixed(2)).join('  ') : '—',
        `${snapshot.cpu_count || 0} cores`)),
    h('p', { class: 'ov-note' },
      'nvidia-smi reports no memory total, used or free on this part — the memory is unified and '
      + 'there is no framebuffer to measure — so there is deliberately no GPU-memory gauge here. '
      + 'The host memory panel above is the GPU memory panel.'),
    procs.length
      ? h('div', { class: 'table-wrap' },
          h('table', null,
            h('thead', null, h('tr', null,
              h('th', null, 'GPU process'),
              h('th', { class: 'num' }, 'Holding'))),
            h('tbody', null, procs.map((proc) => h('tr', null,
              h('td', null, `pid ${proc.pid}${proc.name ? ` · ${proc.name}` : ''}`),
              h('td', { class: 'num' }, bytes(proc.used_bytes)))))))
      : h('p', { class: 'ov-note' }, 'No process is holding GPU memory.'),
    procs.length
      ? h('p', { class: 'ov-note' }, `${bytes(measured)} measured across ${procs.length} process(es).`)
      : null,
    snapshot.disk && snapshot.disk.total_bytes
      ? h('p', { class: 'ov-note' },
          `${bytes(snapshot.disk.free_bytes)} free on ${snapshot.disk.path}.`)
      : null);
}

/* --- servers ------------------------------------------------------------ */

function serversSection(payload) {
  const managed = payload.servers || [];
  const foreign = payload.foreign || [];
  // One machine needs no column telling you which machine it is.
  const clustered = (payload.nodes || []).length > 1;
  if (!managed.length && !foreign.length) {
    return empty('Nothing is serving', 'No managed or hand-launched vLLM container is running.',
      h('button', { class: 'btn-primary', onClick: () => ctxRef.navigate('serve') },
        'Define a server'));
  }

  // Foreign containers are discovered with a plain docker ps on this machine,
  // so anything without a node of its own is here.
  const row = (entry, { id = null, hand = false } = {}) => h('tr', null,
    h('td', null,
      h('div', { class: 'nowrap' }, entry.name),
      hand ? h('div', { class: 'ov-note' }, 'started outside the dashboard') : null),
    clustered
      ? h('td', null, badge(entry.node_local === false ? 'info' : 'plain',
          entry.node || payload.local || 'local'))
      : null,
    h('td', null, badge(entry.status, entry.status)),
    h('td', null, h('span', { class: 'truncate ov-model' }, entry.model || '—')),
    h('td', { class: 'num' }, entry.port || '—'),
    h('td', { class: 'num' }, utilText(entry.util)),
    h('td', { class: 'num' },
      h('button', {
        class: 'btn-sm btn-ghost',
        onClick: () => ctxRef.navigate('serve', id ? String(id) : ''),
      }, 'Open')));

  return h('div', { class: 'table-wrap' },
    h('table', null,
      h('thead', null, h('tr', null,
        h('th', null, 'Server'),
        clustered ? h('th', null, 'Node') : null,
        h('th', null, 'Status'),
        h('th', null, 'Model'),
        h('th', { class: 'num' }, 'Port'),
        h('th', { class: 'num' }, 'Util'),
        h('th', { class: 'num' }, ''))),
      h('tbody', null,
        managed.map((server) => row(server, { id: server.id })),
        foreign.map((entry) => row(entry, { hand: true })))));
}

/* --- jobs --------------------------------------------------------------- */

function jobsSection(payload) {
  const rows = payload.jobs || [];
  if (!rows.length) {
    return empty('No jobs yet', 'Downloads, fine-tuning runs and Heretic runs all show up here.');
  }
  return h('div', { class: 'table-wrap' },
    h('table', null,
      h('thead', null, h('tr', null,
        h('th', null, 'Job'),
        h('th', null, 'Status'),
        h('th', null, 'Progress'),
        h('th', { class: 'num' }, 'Started'),
        h('th', { class: 'num' }, ''))),
      h('tbody', null, rows.map((job) => {
        const progress = job.progress || {};
        const percent = Number(progress.percent);
        const elapsed = job.finished_at && job.started_at
          ? duration(job.finished_at - job.started_at)
          : null;
        return h('tr', null,
          h('td', null,
            h('div', { class: 'truncate ov-model' }, job.title || job.id),
            h('div', { class: 'ov-note' }, `${job.kind} · ${job.id}`)),
          h('td', null, badge(job.status, job.status)),
          h('td', null,
            Number.isFinite(percent)
              ? h('div', { class: 'ov-jobbar' },
                  h('div', { class: 'progress' },
                    h('span', { style: { width: `${Math.max(0, Math.min(100, percent))}%` } })),
                  h('div', { class: 'ov-note' },
                    `${percent.toFixed(0)}%${progress.phase ? ` · ${progress.phase}` : ''}`))
              : h('span', { class: 'ov-note' }, progress.phase || '—')),
          h('td', { class: 'num nowrap' },
            ago(job.created_at),
            elapsed ? h('div', { class: 'ov-note' }, `ran ${elapsed}`) : null),
          h('td', { class: 'num' },
            h('button', {
              class: 'btn-sm btn-ghost',
              onClick: () => ctxRef.navigate('jobs', job.id),
            }, 'Log')));
      }))));
}

/* --- watchdog ----------------------------------------------------------- */

function memguardSection(payload) {
  const events = payload.events || [];
  if (!events.length) return null;
  return panel('Memory watchdog', {
    sub: `${events.length} kill(s) · threshold ${payload.threshold_mib} MiB`,
    body: h('div', { class: 'stack' }, events.slice().reverse().map((event) => notice('warn',
      h('strong', null, `${when(event.ts)} — killed ${event.container}. `),
      h('span', null, event.reason)))),
  });
}

/* --- environment -------------------------------------------------------- */

const IMAGE_TAB = { heretic: 'heretic', finetune: 'finetune' };

function environmentSection(info, imagePayload) {
  const rows = [
    ['Host', `${info.hostname} · ${info.platform}`],
    ['Python', info.python],
    ['Docker', info.docker || 'unavailable'],
    ['vLLM', `${info.vllm_version} · ${info.vllm_flags} flags`],
    ['Image', info.vllm_image],
    ['HF cache', info.hf_cache],
    ['Outputs', info.output_dir],
    ['Datasets', info.dataset_dir],
    ['State', info.state_dir],
  ];

  const guard = info.memguard || {};
  const required = (imagePayload && imagePayload.required) || [];

  return h('div', { class: 'stack' },
    h('dl', { class: 'ov-kv' },
      rows.map(([label, value]) => [h('dt', null, label), h('dd', null, value)])),
    h('div', { class: 'row wrap' },
      badge(info.hf_token_set ? 'running' : 'absent',
        info.hf_token_set ? 'HF token set' : 'no HF token'),
      badge(guard.enabled ? 'running' : 'absent',
        guard.enabled ? `watchdog armed at ${guard.threshold_mib} MiB` : 'watchdog off')),
    required.length
      ? h('div', { class: 'stack' }, required.map((image) => h('div', { class: 'row' },
          badge(image.present ? 'succeeded' : 'failed', image.role),
          h('span', { class: 'truncate ov-model' }, image.tag),
          h('span', { class: 'spacer' }),
          image.present
            ? null
            // The vLLM base image is pulled, not built, and no view offers that.
            : IMAGE_TAB[image.role]
              ? h('button', {
                  class: 'btn-sm',
                  onClick: () => ctxRef.navigate(IMAGE_TAB[image.role]),
                }, 'Build')
              : copyButton(`docker pull ${image.tag}`, 'Copy pull command'))))
      : null,
    imagePayload && !info.hf_token_set
      ? h('p', { class: 'ov-note' },
          'Gated repositories and private models will fail to download until a token is saved '
          + 'on the Models tab.')
      : null);
}

/* --- wiring ------------------------------------------------------------- */

function fill(node, settled, build) {
  if (settled.status === 'fulfilled') {
    const content = build(settled.value);
    mount(node, content || h('p', { class: 'ov-note' }, 'Nothing to show.'));
    return true;
  }
  const message = settled.reason?.message || String(settled.reason);
  mount(node, notice('danger', h('span', null, message)));
  // One toast per distinct failure: the poll would otherwise raise the same
  // banner every ten seconds while the API is down.
  if (message !== lastFailure) {
    lastFailure = message;
    toast(message, { level: 'danger', title: 'Overview' });
  }
  return false;
}

async function refresh({ withImages = false } = {}) {
  const [budget, registry, servers, jobs, guard, imageResult] = await Promise.allSettled([
    get('/system/budget'),
    get('/nodes'),
    get('/servers'),
    get('/jobs?limit=8'),
    get('/system/memguard'),
    withImages ? get('/system/images') : Promise.resolve(images),
  ]);

  // The panel wants both calls: /api/nodes for who is in the cluster, and
  // /api/system/budget for the per-tenant detail only this machine can measure.
  // Without the budget there is nothing to draw; without the registry the panel
  // falls back to this machine alone rather than to an error.
  const cluster = budget.status === 'rejected' ? budget : {
    status: 'fulfilled',
    value: {
      ...(registry.status === 'fulfilled' ? registry.value : {}),
      budget: budget.value,
    },
  };

  const ok = [
    fill(region.memory, cluster, clusterSection),
    fill(region.servers, servers, serversSection),
    fill(region.jobs, jobs, jobsSection),
  ].every(Boolean);
  if (ok) lastFailure = '';

  if (guard.status === 'fulfilled') {
    mount(region.memguard, memguardSection(guard.value));
  }
  if (imageResult.status === 'fulfilled') images = imageResult.value;
  mount(region.environment, systemInfo
    ? environmentSection(systemInfo, images)
    : notice('danger', h('span', null, 'GET /api/system/info is unavailable.')));

  if (servers.status === 'fulfilled') {
    const payload = servers.value;
    const live = (payload.servers || []).filter((s) => LIVE_SERVER.has(s.status)).length
      + (payload.foreign || []).filter((s) => LIVE_SERVER.has(s.status)).length;
    ctxRef.setBadge('serve', live);
  }
  if (jobs.status === 'fulfilled') {
    ctxRef.setBadge('jobs', (jobs.value.jobs || []).filter((j) => j.status === 'running').length);
  }
}

export async function render(container, ctx) {
  ctxRef = ctx;
  ensureStyles('overview', CSS);
  images = null;
  systemInfo = ctx.info;
  lastFailure = '';

  for (const key of ['memory', 'telemetry', 'servers', 'jobs', 'memguard', 'environment']) {
    region[key] = h('div', null, key === 'memguard' ? null : spinner());
  }

  if (!systemInfo) {
    // The shell fetches /system/info once at boot and already toasted if that
    // failed; the environment panel still needs the payload, so try again.
    try { systemInfo = await get('/system/info'); } catch { /* the panel says so */ }
  }

  mount(container,
    h('div', { class: 'page-head' },
      h('div', null,
        h('h1', null, 'Overview'),
        h('p', null,
          'Everything that decides whether the next launch is safe: what holds memory now, '
          + 'what is running, and what the machine is doing.')),
      h('div', { class: 'page-actions' },
        h('button', { class: 'btn-sm', onClick: () => refresh({ withImages: true }) }, 'Refresh'))),
    region.memguard,
    panel('Cluster memory', {
      sub: 'GET /api/nodes · /api/system/budget',
      body: region.memory,
    }),
    h('div', { class: 'grid cols-2' },
      panel('Live telemetry', { sub: 'nvidia-smi · /proc', body: region.telemetry }),
      panel('Environment', { sub: 'images and paths', body: region.environment })),
    panel('Servers', {
      sub: 'managed and hand-launched',
      actions: h('button', { class: 'btn-sm btn-ghost', onClick: () => ctx.navigate('serve') },
        'Serve tab'),
      body: region.servers,
      flush: true,
    }),
    panel('Recent jobs', {
      actions: h('button', { class: 'btn-sm btn-ghost', onClick: () => ctx.navigate('jobs') },
        'Jobs tab'),
      body: region.jobs,
      flush: true,
    }));

  mount(region.telemetry, telemetrySection(ctx.telemetry()));

  closeStream = stream('/system/telemetry/stream', {
    telemetry: (snapshot) => mount(region.telemetry, telemetrySection(snapshot)),
    // The shell already raises the danger toast for this; only the history list
    // here needs to catch up.
    memguard: () => refresh(),
  });

  await refresh({ withImages: true });
  poll = setInterval(refresh, POLL_MS);
}

export function dispose() {
  if (poll) clearInterval(poll);
  poll = null;
  if (closeStream) closeStream();
  closeStream = null;
}

const CSS = `
.ov-pool { border: 1px solid var(--border-strong); border-radius: var(--radius);
  padding: 13px 14px; background: var(--bg-raised); }
.ov-pool-head { display: flex; align-items: baseline; justify-content: space-between;
  gap: 10px; margin-bottom: 12px; }
.ov-pool .help { margin: 12px 0 0; }

.ov-track {
  display: flex; height: 26px; border-radius: var(--radius-s); overflow: hidden;
  background: var(--bg-sunken); border: 1px solid var(--border);
}
.ov-seg { display: block; height: 100%; transition: width .4s ease; }
.ov-c-engine  { background: var(--accent); }
.ov-c-other   { background: var(--info); }
.ov-c-free    { background: var(--ok); }
.ov-c-reserve { background: var(--border-strong); }
.ov-legend span { display: inline-flex; align-items: center; }
.ov-tiles {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 14px;
}
.ov-kv {
  display: grid; grid-template-columns: minmax(72px, auto) minmax(0, 1fr);
  gap: 5px 14px; margin: 0; font-size: 12.5px; align-items: baseline;
}
.ov-kv dt { color: var(--text-faint); }
.ov-kv dd { margin: 0; font-family: var(--mono); font-size: 11.5px; overflow-wrap: anywhere; }
.ov-note { margin: 3px 0 0; font-size: 11.5px; color: var(--text-faint); line-height: 1.5; }
.ov-jobbar { width: 110px; }
.ov-model {
  display: inline-block; max-width: 34ch; vertical-align: bottom;
  font-family: var(--mono); font-size: 12px;
}
.ov-node {
  border: 1px solid var(--border); border-radius: var(--radius); padding: 12px 14px;
  display: flex; flex-direction: column; gap: 10px;
}
.ov-node.down { border-color: var(--danger); }
.ov-node-head strong { font-size: 13px; }
`;
