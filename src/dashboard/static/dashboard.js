// dashboard.js — live SSE client + Chart.js chart management

// ── Chart defaults ───────────────────────────────────────────────────────────
Chart.defaults.color = '#8b949e';
Chart.defaults.borderColor = '#30363d';
Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";

const MAX_POINTS = 400;   // keep last N update points on each chart

// ── Helpers ──────────────────────────────────────────────────────────────────
function makeLineChart(id, labels, datasets) {
  const ctx = document.getElementById(id).getContext('2d');
  return new Chart(ctx, {
    type: 'line',
    data: {labels, datasets},
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: true,
      plugins: { legend: { labels: { boxWidth: 12, padding: 10 } } },
      scales: {
        x: { ticks: { maxTicksLimit: 8, maxRotation: 0 } },
        y: { ticks: { maxTicksLimit: 6 } }
      },
      elements: { point: { radius: 0 }, line: { tension: 0.3, borderWidth: 2 } },
    },
  });
}

function ds(label, color, data=[]) {
  return { label, borderColor: color, backgroundColor: color + '22', data, fill: false };
}

function pushTrim(arr, val) {
  arr.push(val);
  if (arr.length > MAX_POINTS) arr.shift();
}

function updateChart(chart) {
  chart.update('none');  // no animation
}

// ── Chart data arrays ────────────────────────────────────────────────────────
const labels = [];

const rewardData   = [];
const plossData    = [];
const vlossData    = [];
const blossData    = [];
const tlossData    = [];
const entropyData  = [];
const lrData       = [];
const eplenData    = [];
const evalLabels   = [];
const wr0Data      = [];
const wr1Data      = [];

// EMA smoothed lines (α=0.95 → ~20업데이트 반영)
const EMA_ALPHA = 0.1;  // 있을때마다 업데이트: new_ema = alpha*raw + (1-alpha)*prev
const ema = {reward: null, vloss: null, ploss: null, bloss: null, entropy: null, eplen: null};
const rewardEMA  = [];
const vlossEMA   = [];
const plossEMA   = [];
const blossEMA   = [];
const entropyEMA = [];
const eplenEMA   = [];

function updateEMA(key, val) {
  if (val === null || val === undefined || isNaN(val)) return null;
  ema[key] = ema[key] === null ? val : EMA_ALPHA * val + (1 - EMA_ALPHA) * ema[key];
  return ema[key];
}

// ── Create charts ────────────────────────────────────────────────────────────
const chartReward  = makeLineChart('chart-reward',  labels, [
  ds('Reward',      '#58a6ff44', rewardData),
  {...ds('Trend',   '#58a6ff',   rewardEMA),  borderWidth: 3 },
]);
const chartLosses  = makeLineChart('chart-losses',  labels, [
  ds('Policy Raw',  '#f8514966', plossData),
  {...ds('Policy',  '#f85149',   plossEMA),   borderWidth: 3 },
  ds('Value Raw',   '#d2992266', vlossData),
  {...ds('Value',   '#d29922',   vlossEMA),   borderWidth: 3 },
  ds('Belief Raw',  '#bc8cff66', blossData),
  {...ds('Belief',  '#bc8cff',   blossEMA),   borderWidth: 3 },
]);
const chartEntropy = makeLineChart('chart-entropy', labels, [
  ds('Entropy Raw', '#3fb95044', entropyData),
  {...ds('Entropy', '#3fb950',   entropyEMA), borderWidth: 3 },
]);
const chartLR      = makeLineChart('chart-lr',      labels, [ds('LR',      '#ff7b72', lrData)]);
const chartEpLen   = makeLineChart('chart-eplen',   labels, [
  ds('Ep Len Raw',  '#79c0ff44', eplenData),
  {...ds('Ep Len',  '#79c0ff',   eplenEMA),   borderWidth: 3 },
]);
const chartWR      = makeLineChart('chart-winrate', evalLabels, [
  ds('P0 Win%', '#3fb950', wr0Data),
  ds('P1 Win%', '#f85149', wr1Data),
]);

// ── KPI helpers ──────────────────────────────────────────────────────────────
function fmt(v, digits=4) {
  if (v === null || v === undefined) return '—';
  const n = parseFloat(v);
  return isNaN(n) ? '—' : n.toFixed(digits);
}
function fmtPct(v) { return (v !== null && v !== undefined) ? (parseFloat(v)*100).toFixed(1)+'%' : '—'; }
function fmtSci(v) { return (v !== null && v !== undefined) ? parseFloat(v).toExponential(2) : '—'; }
function fmtBig(v) { if (!v) return '—'; const n=parseInt(v); return n>=1e6 ? (n/1e6).toFixed(2)+'M' : n>=1e3 ? (n/1e3).toFixed(1)+'K' : n; }

function setKPI(id, val, extra='') {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = val;
  el.className = 'value ' + extra;
}

// ── Log ──────────────────────────────────────────────────────────────────────
function addLog(text) {
  const li = document.createElement('li');
  const ts = new Date().toLocaleTimeString();
  li.innerHTML = `<span class="ts">${ts}</span>${text}`;
  const ul = document.getElementById('log-list');
  ul.prepend(li);
  // keep last 100 log lines
  while (ul.children.length > 100) ul.removeChild(ul.lastChild);
}

// ── SSE connection ────────────────────────────────────────────────────────────
const dot   = document.getElementById('status-dot');
const label = document.getElementById('status-label');

function connect() {
  const es = new EventSource('/stream');

  es.onopen = () => {
    dot.className = 'live';
    label.textContent = 'Live';
  };

  es.onmessage = (e) => {
    let d;
    try { d = JSON.parse(e.data); } catch (_) { return; }

    if (d.type === 'ping') return;

    if (d.type === 'update' || (!d.type && d.update !== undefined)) {
      const x = d.update !== undefined ? String(d.update) : String(labels.length);
      pushTrim(labels, x);
      pushTrim(rewardData,  d.mean_reward ?? null);
      pushTrim(plossData,   d.policy_loss ?? null);
      pushTrim(vlossData,   d.value_loss  ?? null);
      pushTrim(blossData,   d.belief_loss ?? null);
      pushTrim(entropyData, d.entropy     ?? null);
      pushTrim(lrData,      d.learning_rate !== undefined ? parseFloat(d.learning_rate) : null);
      pushTrim(eplenData,   d.mean_episode_length ?? null);

      // EMA smoothed
      pushTrim(rewardEMA,  updateEMA('reward',  d.mean_reward));
      pushTrim(vlossEMA,   updateEMA('vloss',   d.value_loss));
      pushTrim(plossEMA,   updateEMA('ploss',   d.policy_loss));
      pushTrim(blossEMA,   updateEMA('bloss',   d.belief_loss));
      pushTrim(entropyEMA, updateEMA('entropy', d.entropy));
      pushTrim(eplenEMA,   updateEMA('eplen',   d.mean_episode_length));

      [chartReward, chartLosses, chartEntropy, chartLR, chartEpLen].forEach(updateChart);

      setKPI('kpi-update',  d.update !== undefined ? d.update : '—');
      setKPI('kpi-steps',   fmtBig(d.timesteps));
      const rew = d.mean_reward ?? null;
      setKPI('kpi-reward',  fmt(rew, 3), rew !== null ? (rew >= 0 ? 'good' : 'bad') : '');
      setKPI('kpi-ploss',   fmt(d.policy_loss, 4));
      setKPI('kpi-vloss',   fmt(d.value_loss,  4));
      setKPI('kpi-bloss',   fmt(d.belief_loss, 4));
      setKPI('kpi-entropy', fmt(d.entropy, 4));
      setKPI('kpi-lr',      d.learning_rate !== undefined ? parseFloat(d.learning_rate).toExponential(2) : '—');
    }

    if (d.type === 'eval') {
      const x = d.update !== undefined ? String(d.update) : String(evalLabels.length);
      pushTrim(evalLabels, x);
      pushTrim(wr0Data,    d.player0_win_rate !== undefined ? d.player0_win_rate * 100 : null);
      pushTrim(wr1Data,    d.player1_win_rate !== undefined ? d.player1_win_rate * 100 : null);
      updateChart(chartWR);

      setKPI('kpi-wr0', fmtPct(d.player0_win_rate), (d.player0_win_rate ?? 0) > 0.5 ? 'good' : '');
      setKPI('kpi-wr1', fmtPct(d.player1_win_rate));
      addLog(`[Eval at ${x}] P0 ${fmtPct(d.player0_win_rate)}  P1 ${fmtPct(d.player1_win_rate)}`);
    }

    if (d.type === 'done') {
      dot.className = 'done';
      label.textContent = 'Training complete';
      addLog(`Training finished at ${fmtBig(d.timesteps)} steps`);
    }
  };

  es.onerror = () => {
    dot.className = '';
    label.textContent = 'Reconnecting…';
    es.close();
    setTimeout(connect, 3000);
  };
}

connect();
