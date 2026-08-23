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
  panel, pct, spinner, stat, svg, toast, when,
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

// One decimal, and only when it says something: 48°C not 48.0°C, 12.1W not 12W.
const reading = (value, suffix) =>
  (Number.isFinite(value) ? `${Math.round(value * 10) / 10}${suffix}` : '—');

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
/* Both nodes get the same four readings, in the same order, so the cards can be
   compared at a glance. The local card adds what only this machine can know —
   which engines committed what — underneath, rather than in a different shape. */

/** Whether this node's engines spend a framebuffer or the host's own RAM.
 *  Everything below is labelled from this: the same bytes mean different things
 *  on the two kinds of machine, and a panel that says "Host memory" over a
 *  framebuffer figure is not a cosmetic problem — it is the reading an operator
 *  sizes a model against. */
function isDiscrete(node, budget) {
  const kind = node.pool_kind || (node.local && budget ? budget.pool_kind : '');
  return kind === 'discrete';
}

function nodeMemory(node, budget) {
  const running = node.containers || [];
  const isLocal = node.local && budget && budget.total_bytes;
  const discrete = isDiscrete(node, budget);
  const devices = node.device_count || 0;
  // Two different totals, and the panel needs both. node.total_bytes is ONE
  // device, because --gpu-memory-utilization is a fraction of one framebuffer.
  // The bars answer a different question — how much memory is on this machine
  // and how much of it is gone — and there the answer is every card added up.
  // Showing the per-device figure here made a 2x45 GiB box report 45 GiB and
  // hid a second card that was 86% full.
  const cumulative = discrete && (node.vram_total_bytes || 0) > 0;
  const perDevice = node.total_bytes || 0;
  const total = cumulative ? node.vram_total_bytes : perDevice;
  // Summed used, not total-minus-free: the two differ by driver overhead, and
  // the measured figure is the one that matches nvidia-smi.
  const used = cumulative
    ? Math.max(0, node.vram_used_bytes || 0)
    : Math.max(0, perDevice - Math.max(0, node.available_bytes || 0));
  const available = cumulative
    ? Math.max(0, node.vram_free_bytes || 0)
    : Math.max(0, node.available_bytes || 0);

  const segments = [
    {
      key: 'other',
      label: 'In use',
      value: used,
      note: discrete
        ? (devices > 1
            ? `what every engine and CUDA context holds across all ${devices} devices`
            : 'what the engines and any other CUDA context on this node hold in the framebuffer')
        : 'everything MemAvailable does not count as reclaimable, engines included',
    },
    {
      key: 'free',
      label: 'Available',
      value: available,
      note: cumulative && devices > 1
        ? 'summed across devices — a new engine draws on one of them, not on the sum'
        : 'what a new engine here could take before the guard steps in',
    },
  ];

  return h('div', { class: 'stack' },
    h('div', { class: 'ov-tiles' },
      discrete
        ? stat('GPU memory', bytes(total),
            devices > 1
              ? `${devices} devices · ${bytes(perDevice)} each — utilisation is a fraction of one`
              : 'the framebuffer, separate from host RAM')
        : stat('Host memory', bytes(total), 'shared by the CPU and the GPU'),
      stat('In use', bytes(used), total ? pct(used / total) : ''),
      stat('Available', bytes(available), total ? pct(available / total) : ''),
      // Host RAM stops being the headline on a discrete box but does not stop
      // mattering: --cpu-offload-gb and the loading process both draw on it.
      discrete && node.host_total_bytes
        ? stat('Host RAM', bytes(node.host_total_bytes),
            `${bytes(node.host_available_bytes || 0)} available · offload and loading only`)
        : stat('Containers', String(running.length),
            running.length ? running.map((c) => c.name).join(', ').slice(0, 40) : 'none running')),
    memoryBars(segments, total),
    isLocal ? localCommitment(budget) : null);
}

/** What only this machine can report: which engines have committed what, from
 *  their own command lines, and how much is left to give. */
function localCommitment(budget) {
  const tenants = budget.tenants || [];
  return h('div', { class: 'ov-commit' },
    h('div', { class: 'row wrap' },
      h('span', { class: 'faint small' },
        `Committed ${utilText(budget.committed_util)} of a ${utilText(budget.max_util)} ceiling · `
        + `${bytes(budget.free_bytes_to_commit ?? 0)} free for a new engine`
        + (budget.free_util ? `, or --gpu-memory-utilization ${utilText(budget.free_util)}` : ''))),
    tenants.length
      ? h('div', { class: 'table-wrap' },
          h('table', null,
            h('thead', null, h('tr', null,
              h('th', null, 'Container'), h('th', null, 'engine'),
              h('th', { class: 'num' }, 'util'),
              h('th', { class: 'num' }, 'reserves'), h('th', null, ''))),
            h('tbody', null, tenants.map((t) =>
              h('tr', null,
                h('td', { class: 'mono' }, t.name),
                h('td', null, badge('plain', t.engine === 'llamacpp' ? 'llama.cpp' : 'vLLM')),
                // utilText already renders — for a non-finite value, which is
                // exactly what an engine with no fraction reports. The bytes
                // beside it are the figure that is true for both.
                h('td', { class: 'num' }, utilText(t.util)),
                h('td', { class: 'num' }, bytes(t.bytes_committed)),
                h('td', { class: 'faint small' },
                  (t.managed ? '' : 'not managed here') + (t.note ? ` · ${t.note}` : '')))))))
      : h('div', { class: 'faint small' }, 'no engine has committed memory here'));
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
    // Utilisation, temperature and power used to repeat here per node. They are
    // in the shared panel above now — the same reading for both machines on one
    // axis says more than two copies of it in two cards.
    down ? null : nodeMemory(node, budget));
}

/* --- time series ---------------------------------------------------------
   One panel per metric, both nodes drawn on it. That split is deliberate: the
   two machines share a unit and belong on the same axis, but GPU percent,
   degrees and watts do not — a single axis carrying all of them produces a
   line whose height means nothing. Colour follows the node, held constant
   across every panel, so a colour means the same machine wherever it appears.

   Palette: categorical slots 1 and 2, validated for both surfaces
   (CVD ΔE 26.8 dark / 24.7 light, contrast >= 3:1). */

const SERIES_COLOURS = ['var(--series-1)', 'var(--series-2)'];
const PLOT = { w: 560, h: 150, padL: 40, padR: 8, padT: 12, padB: 22 };

function seriesColour(index) {
  return SERIES_COLOURS[index % SERIES_COLOURS.length];
}

/** Round a peak up to something a human reads off an axis. */
function niceMax(values, floor) {
  const peak = Math.max(floor, ...values.filter(Number.isFinite));
  const step = peak <= 10 ? 2 : peak <= 50 ? 10 : peak <= 120 ? 25 : 50;
  return Math.ceil(peak / step) * step;
}

const clock = (ts) => new Date(ts * 1000).toLocaleTimeString([], {
  hour: '2-digit', minute: '2-digit',
});

/** One line per GPU where the machine can tell its GPUs apart, one line per
 *  node where it cannot. The second case is every unified part, which reports
 *  no per-device rows and is charted exactly as it always was. */
function seriesOf(hist) {
  if (hist.series && hist.series.length) return hist.series;
  return (hist.nodes || []).map((name) => ({ id: name, node: name, device: null, label: name }));
}

/** A series reads from its own device when it has one. Falling back to the
 *  node-level figure would quietly draw the same line twice on a two-card box. */
function seriesValue(sample, series, key) {
  const node = sample.nodes[series.node];
  if (!node) return undefined;
  if (series.device === null || series.device === undefined) return node[key];
  const device = (node.devices || {})[series.device];
  return device ? device[key] : undefined;
}

function metricPanel(metric, hist) {
  const nodes = seriesOf(hist);
  const samples = hist.samples || [];
  if (samples.length < 2) {
    return h('div', { class: 'ov-plot' },
      h('div', { class: 'ov-plot-head' }, h('strong', null, metric.label)),
      h('div', { class: 'faint small ov-plot-empty' },
        'collecting — the first line appears once there are two samples'));
  }

  const t0 = samples[0].ts;
  const span = Math.max(1, samples[samples.length - 1].ts - t0);
  const at = (sample, series) => seriesValue(sample, series, metric.key);
  const all = samples.flatMap((sample) => nodes.map((node) => at(sample, node)));
  // A percentage keeps its full 0-100 axis: an idle GPU should look idle, not
  // like a mountain range of rescaled jitter. Everything else follows the data.
  const max = metric.max || niceMax(all, 1);

  const x = (ts) => PLOT.padL + ((ts - t0) / span) * (PLOT.w - PLOT.padL - PLOT.padR);
  const y = (v) => PLOT.h - PLOT.padB - (Math.min(v, max) / max) * (PLOT.h - PLOT.padT - PLOT.padB);

  const gridlines = [0, 0.25, 0.5, 0.75, 1].map((f) =>
    svg('line', { x1: PLOT.padL, x2: PLOT.w - PLOT.padR, y1: y(max * f), y2: y(max * f),
      class: 'ov-grid' }));
  const ticks = [0, 0.5, 1].map((f) =>
    svg('text', { x: PLOT.padL - 6, y: y(max * f) + 4, 'text-anchor': 'end', class: 'ov-tick' },
      String(Math.round(max * f))));
  const times = [
    svg('text', { x: PLOT.padL, y: PLOT.h - 6, class: 'ov-tick' }, clock(t0)),
    svg('text', { x: PLOT.w - PLOT.padR, y: PLOT.h - 6, 'text-anchor': 'end', class: 'ov-tick' },
      clock(samples[samples.length - 1].ts)),
  ];

  const lines = [];
  const dots = [];
  const values = [];
  nodes.forEach((node, index) => {
    const colour = seriesColour(index);
    const points = samples
      .filter((sample) => Number.isFinite(at(sample, node)))
      .map((sample) => `${x(sample.ts).toFixed(1)},${y(at(sample, node)).toFixed(1)}`);
    if (points.length >= 2) {
      lines.push(svg('polyline', { points: points.join(' '), fill: 'none', stroke: colour,
        'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }));
    }
    // The hover dot rides the line; it is parked off-canvas until the pointer
    // arrives, so it costs one node per series and no re-render.
    dots.push(svg('circle', { r: 4, fill: colour, stroke: 'var(--bg-raised)',
      'stroke-width': 2, class: 'ov-dot', cx: -20, cy: -20 }));
    const last = [...samples].map((sample) => at(sample, node)).reverse().find(Number.isFinite);
    values.push(h('b', null, reading(last, metric.unit)));
  });

  const keys = nodes.map((series, index) => h('span', { class: 'ov-key' },
    h('i', { style: { background: seriesColour(index) } }),
    h('span', null, series.label), values[index]));

  const cursor = svg('line', { y1: PLOT.padT, y2: PLOT.h - PLOT.padB, class: 'ov-cursor',
    x1: -20, x2: -20 });
  const stamp = h('span', { class: 'ov-stamp' }, '');

  // Grafana's habit: the legend doubles as the readout. Rather than float a
  // tooltip box, the numbers already on screen change to the sample under the
  // pointer, and go back to live when it leaves.
  const restore = () => {
    nodes.forEach((node, index) => {
      const last = [...samples].map((sample) => at(sample, node)).reverse().find(Number.isFinite);
      values[index].textContent = reading(last, metric.unit);
      dots[index].setAttribute('cx', -20);
      dots[index].setAttribute('cy', -20);
    });
    cursor.setAttribute('x1', -20);
    cursor.setAttribute('x2', -20);
    stamp.textContent = '';
  };

  const hover = (event) => {
    const box = event.currentTarget.getBoundingClientRect();
    if (!box.width) return;
    const vx = ((event.clientX - box.left) / box.width) * PLOT.w;
    const wanted = t0 + ((vx - PLOT.padL) / (PLOT.w - PLOT.padL - PLOT.padR)) * span;
    let hit = samples[0];
    for (const sample of samples) {
      if (Math.abs(sample.ts - wanted) < Math.abs(hit.ts - wanted)) hit = sample;
    }
    const hx = x(hit.ts);
    cursor.setAttribute('x1', hx);
    cursor.setAttribute('x2', hx);
    stamp.textContent = clock(hit.ts);
    nodes.forEach((node, index) => {
      const value = at(hit, node);
      values[index].textContent = reading(value, metric.unit);
      dots[index].setAttribute('cx', Number.isFinite(value) ? hx : -20);
      dots[index].setAttribute('cy', Number.isFinite(value) ? y(value) : -20);
    });
  };

  return h('div', { class: 'ov-plot' },
    h('div', { class: 'ov-plot-head' },
      h('strong', null, metric.label),
      stamp,
      h('span', { class: 'ov-keys' }, keys)),
    svg('svg', { viewBox: `0 0 ${PLOT.w} ${PLOT.h}`, class: 'ov-svg',
      role: 'img', 'aria-label': `${metric.label} over the last ${Math.round(span / 60)} minutes, `
        + `one line per series: ${nodes.map((series) => series.label).join(', ')}`,
      onMousemove: hover, onMouseleave: restore },
      gridlines, ticks, times, cursor, lines, dots,
      svg('rect', { x: 0, y: 0, width: PLOT.w, height: PLOT.h, fill: 'transparent' })));
}

function timeSeries(hist) {
  if (!hist || !seriesOf(hist).length) return null;
  const minutes = Math.round((hist.window_seconds || 1800) / 60);
  return h('div', { class: 'stack' },
    h('div', { class: 'ov-plot-grid' }, (hist.metrics || []).map((m) => metricPanel(m, hist))),
    h('div', { class: 'faint small' },
      `${seriesOf(hist).length} series on every panel, one colour each — a line per GPU where the `
      + `driver breaks them out, otherwise one per machine. Last ${minutes} minutes, `
      + `sampled every ${hist.interval_seconds}s; hover to read a point. `
      + 'A node that goes unreachable stops contributing and its line simply ends.'));
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
      nodeMemory({ local: true, total_bytes: payload.budget.total_bytes,
                   available_bytes: payload.budget.available_bytes, containers: [] },
                 payload.budget));
  }
  const local = registry.find((node) => node.local) || {};

  return h('div', { class: 'stack' },
    combinedPool(payload.combined, registry),
    timeSeries(payload.history),
    registry.map((node) => nodeCard(node, payload.budget)),
    isDiscrete(local, payload.budget)
      ? notice('info',
          h('strong', null, 'GPU memory is its own pool on these machines. '),
          h('span', null,
            '--gpu-memory-utilization 0.50 reserves half of the '
            + `${bytes(local.total_bytes || payload.budget.total_bytes)} framebuffer of one device, `
            + 'so the fractions of every engine on that node add up against the card, not against '
            + 'host RAM — which can read empty while the GPU is full. Overcommitting fails the '
            + 'launch with a CUDA OOM rather than freezing the box. The fractions do not travel: a '
            + 'peer\'s ceiling is its own.'))
      : notice('info',
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
  // Empty on a unified part, where there is no separate pool to gauge.
  const vram = snapshot.vram || {};
  const load = snapshot.load || [];
  const procs = snapshot.gpu_processes || [];
  const measured = procs.reduce((sum, proc) => sum + (proc.used_bytes || 0), 0);

  return h('div', { class: 'stack' },
    h('div', { class: 'ov-tiles' },
      // Utilisation, temperature and power live in the shared panel above, for
      // both machines at once. What is left here is what only this box has:
      // the device it is, the clock it is running at, and who is holding it.
      stat('Device', gpu.name || 'accelerator',
        gpu.driver_version ? `driver ${gpu.driver_version}` : ''),
      stat('SM clock', reading(gpu.clocks_sm, ' MHz'),
        gpu.pstate ? `pstate ${gpu.pstate}` : ''),
      stat('Load', load.length ? load.map((value) => value.toFixed(2)).join('  ') : '—',
        `${snapshot.cpu_count || 0} cores`)),
    vram.device_count
      ? h('div', { class: 'stack' },
          h('div', { class: 'ov-tiles' },
            stat('VRAM total', bytes(vram.total_bytes),
              vram.device_count > 1
                ? `${vram.device_count} devices · ${bytes(vram.per_device_total_bytes)} each`
                : 'one device'),
            stat('VRAM in use', bytes(vram.used_bytes),
              vram.total_bytes ? pct(vram.used_bytes / vram.total_bytes) : ''),
            stat('VRAM free', bytes(vram.free_bytes),
              vram.total_bytes ? pct(vram.free_bytes / vram.total_bytes) : '')),
          (vram.devices || []).length > 1
            ? h('div', { class: 'table-wrap' },
                h('table', null,
                  h('thead', null, h('tr', null,
                    h('th', null, 'Device'),
                    h('th', { class: 'num' }, 'In use'),
                    h('th', { class: 'num' }, 'Free'))),
                  h('tbody', null, (vram.devices || []).map((dev) => h('tr', null,
                    h('td', null, `${dev.index}: ${dev.name}`),
                    h('td', { class: 'num' }, bytes(dev.used_bytes)),
                    h('td', { class: 'num' }, bytes(dev.free_bytes)))))))
            : null,
          h('p', { class: 'ov-note' },
            'Per device, because --gpu-memory-utilization is a fraction of one framebuffer. '
            + 'A model that does not fit on the smallest device does not fit, however much the '
            + 'totals add up to.'))
      : h('p', { class: 'ov-note' },
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
    return empty('Nothing is serving', 'No managed or hand-launched engine container is running.',
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
    h('td', null, badge(entry.status, entry.status),
      entry.engine && entry.engine !== 'vllm'
        ? badge('plain', entry.engine_label || entry.engine) : null),
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
    ...(info.engines || [{ label: 'vLLM', version: info.vllm_version,
      flags: info.vllm_flags, image: info.vllm_image }]).map((engine) => [
      engine.label,
      `${engine.version} · ${engine.flags} flags · ${engine.image}`,
    ]),
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
          badge(image.present ? 'succeeded' : (image.essential === false ? 'absent' : 'failed'),
            image.role),
          h('span', { class: 'truncate ov-model' }, image.tag),
          h('span', { class: 'spacer' }),
          image.present
            ? null
            // Every image this dashboard requires is BUILT from docker/, not
            // pulled: offering `docker pull` for a local tag sent people to a
            // registry that has never heard of it. Two of them have a tab that
            // builds them; the vLLM one has no tab, so hand over the command.
            : IMAGE_TAB[image.role]
              ? h('button', {
                  class: 'btn-sm',
                  onClick: () => ctxRef.navigate(IMAGE_TAB[image.role]),
                }, 'Build')
              : copyButton(
                  `docker build -t ${image.tag} -f docker/${image.dockerfile || 'vllm.Dockerfile'}`
                  + ' docker/',
                  'Copy build command'))))
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
  const [budget, registry, history, servers, jobs, guard, imageResult] = await Promise.allSettled([
    get('/system/budget'),
    get('/nodes'),
    get('/nodes/history?minutes=30'),
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
      history: history.status === 'fulfilled' ? history.value : null,
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
/* One panel per metric, both machines drawn into it. Two columns on a normal
   window: at four across, a 560-unit viewBox scales down far enough that the
   axis labels stop being readable. */
.ov-plot-grid {
  display: grid; gap: var(--gap);
  grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
}
.ov-plot {
  border: 1px solid var(--border); border-radius: var(--radius);
  background: var(--bg-raised); padding: 10px 12px 4px;
}
.ov-plot-head {
  display: flex; align-items: baseline; gap: 10px;
  flex-wrap: wrap; margin-bottom: 2px;
}
.ov-plot-empty { padding: 24px 0 28px; }
.ov-svg { display: block; width: 100%; height: auto; }
.ov-svg .ov-grid { stroke: var(--border); stroke-width: 1; }
.ov-svg .ov-tick { fill: var(--text-faint); font-size: 10px; font-family: var(--mono); }
.ov-svg .ov-cursor { stroke: var(--border-strong); stroke-width: 1; stroke-dasharray: 3 3; }
.ov-svg .ov-dot { pointer-events: none; }

.ov-keys { display: flex; gap: 12px; flex-wrap: wrap; margin-left: auto; }
/* The swatch carries the identity; the text stays in ink tokens so it keeps its
   contrast against the panel no matter which hue the node drew. */
.ov-key { display: inline-flex; align-items: center; gap: 5px;
  font-size: 11px; color: var(--text-dim); }
.ov-key i { width: 9px; height: 9px; border-radius: 2px; flex: none; }
.ov-key b { font-family: var(--mono); color: var(--text); font-weight: 600; }
.ov-stamp { font-family: var(--mono); font-size: 11px; color: var(--text-faint); }

.ov-commit { margin-top: 12px; padding-top: 10px; border-top: 1px solid var(--border); }
.ov-commit .table-wrap { margin-top: 8px; }
.ov-node-tel { margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--border); }
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
