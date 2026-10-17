/* Multi-Agent Digital Twin dashboard client.
 *
 * The browser is a pure view + input device:
 *   - state arrives over WebSocket (fallback: polling /state)
 *   - every button / click / key becomes an HTTP call; the server engine
 *     validates it and changes the simulation.  Nothing here mutates robots.
 *
 * Rendering is smooth: between 10 Hz state updates the robot poses are
 * extrapolated from their velocities (display only).
 */
(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const FOLLOWER_COLORS = ['#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e',
                           '#17becf', '#8c564b', '#e377c2', '#7f7f7f'];
  const LEADER_COLOR = '#d62728';
  const PACKET_COLORS = { OK: '#2ca02c', DELAYED: '#ff9900', LOST: '#d62728' };
  const TASK_COLORS = { PENDING: '#bbbbbb', ASSIGNED: '#ff9900', IN_PROGRESS: '#f2c200',
                        DONE: '#2ca02c', CANCELLED: '#dddddd' };
  const PAD = 1.0;                 // metres of padding around the world rectangle
  const TRAIL_LEN = 160;
  const KEY_MAP = { w: 'W', a: 'A', s: 'S', d: 'D', arrowup: 'W', arrowdown: 'S',
                    arrowleft: 'A', arrowright: 'D', ' ': 'SPACE', r: 'R', x: 'X', f: 'F' };
  const ONE_SHOT = new Set(['SPACE', 'R', 'X', 'F']);

  let state = null;                // latest server snapshot
  let stateWall = 0;               // performance.now() when it arrived
  let lastTime = -1;
  const trails = {};               // agent id -> [[x, y], ...]
  const held = new Set();          // keys currently held (for release on blur)
  let ws = null, pollTimer = null, agentCount = 0;

  // ------------------------------------------------------------------ helpers
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v)) ? '-' : Number(v).toFixed(d);
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const colorOf = (id) => id === 0 ? LEADER_COLOR : FOLLOWER_COLORS[(id - 1) % FOLLOWER_COLORS.length];

  let toastTimer = null;
  function toast(msg, ok = true) {
    const el = $('feedback');
    el.textContent = msg;
    el.className = 'feedback ' + (ok ? 'ok' : 'err');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.textContent = ''; }, 7000);
  }

  async function api(path, body, method = 'POST') {
    try {
      const res = await fetch(path, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: body === undefined ? undefined : JSON.stringify(body),
      });
      const data = await res.json().catch(() => ({ ok: false, message: `HTTP ${res.status}` }));
      if (data && data.message !== undefined) toast(data.message, !!data.ok);
      else if (data && Array.isArray(data.detail) && data.detail.length) {      // FastAPI validation error
        const d = data.detail[0], field = (d.loc || []).slice(1).join('.');
        toast(`INVALID INPUT${field ? ' (' + field + ')' : ''}: ${d.msg}`, false);
      } else if (!res.ok) toast(`HTTP ${res.status}`, false);
      return data;
    } catch (e) {
      toast('Server unreachable: ' + e.message, false);
      return { ok: false };
    }
  }
  const selAgent = () => $('sel-agent').value;

  // ------------------------------------------------------------------- canvas
  const canvas = $('world');
  const ctx = canvas.getContext('2d');
  let view = { s: 1, ox: 0, oy: 0, w: 0, h: 0 };   // px/m and world origin offsets

  function layoutCanvas() {
    if (!state) return;
    const W = state.world, wm = W.x_max - W.x_min + 2 * PAD, hm = W.y_max - W.y_min + 2 * PAD;
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.parentElement.clientWidth;
    const cssH = cssW * hm / wm;
    canvas.style.height = cssH + 'px';
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    view = { s: canvas.width / wm, ox: W.x_min - PAD, oy: W.y_max + PAD, w: canvas.width, h: canvas.height, dpr };
  }
  const toPx = (x, y) => [(x - view.ox) * view.s, (view.oy - y) * view.s];
  const toWorld = (px, py) => [px / view.s + view.ox, view.oy - py / view.s];

  // display pose: extrapolate between server updates (cosmetic only)
  function displayPose(a) {
    let dt = 0;
    if (state && !state.runner.paused && !state.halted) {
      dt = Math.min((performance.now() - stateWall) / 1000 * state.runner.speed, 0.2);
    }
    return { x: a.x + a.v * Math.cos(a.theta) * dt,
             y: a.y + a.v * Math.sin(a.theta) * dt,
             th: a.theta + a.omega * dt };
  }

  function draw() {
    requestAnimationFrame(draw);
    if (!state) return;
    if (canvas.width !== view.w || !view.w) layoutCanvas();
    const W = state.world, s = view.s;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, view.w, view.h);

    // grid + axis labels
    ctx.lineWidth = 1;
    ctx.strokeStyle = css('--panel-border');
    ctx.fillStyle = css('--muted');
    ctx.font = `${11 * view.dpr}px system-ui, sans-serif`;
    ctx.globalAlpha = 0.5;
    for (let x = Math.ceil(W.x_min - PAD); x <= W.x_max + PAD; x += 2) {
      const [px] = toPx(x, 0); ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, view.h); ctx.stroke();
    }
    for (let y = Math.ceil(W.y_min - PAD); y <= W.y_max + PAD; y += 2) {
      const [, py] = toPx(0, y); ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(view.w, py); ctx.stroke();
    }
    ctx.globalAlpha = 1;
    for (let x = Math.ceil((W.x_min - PAD) / 4) * 4; x <= W.x_max + PAD; x += 4) {
      const [px] = toPx(x, W.y_min - PAD); ctx.fillText(String(x), px + 3, view.h - 4 * view.dpr);
    }
    for (let y = Math.ceil((W.y_min - PAD) / 4) * 4; y <= W.y_max + PAD; y += 4) {
      const [, py] = toPx(W.x_min - PAD, y); ctx.fillText(String(y), 3, py - 3);
    }

    // world boundary + soft-guard zone
    const [bx0, by0] = toPx(W.x_min, W.y_max), [bx1, by1] = toPx(W.x_max, W.y_min);
    ctx.strokeStyle = css('--text'); ctx.lineWidth = 2 * view.dpr;
    ctx.strokeRect(bx0, by0, bx1 - bx0, by1 - by0);
    if (W.margin) {
      const m = W.margin * s;
      ctx.setLineDash([6 * view.dpr, 5 * view.dpr]); ctx.lineWidth = view.dpr;
      ctx.strokeStyle = css('--muted');
      ctx.strokeRect(bx0 + m, by0 + m, bx1 - bx0 - 2 * m, by1 - by0 - 2 * m);
      ctx.setLineDash([]);
    }

    const agents = state.agents, lead = displayPose(agents[0]);

    // trails
    ctx.lineWidth = 1.5 * view.dpr; ctx.globalAlpha = 0.45;
    for (const a of agents) {
      const tr = trails[a.id]; if (!tr || tr.length < 2) continue;
      ctx.strokeStyle = colorOf(a.id); ctx.beginPath();
      tr.forEach(([x, y], i) => { const [px, py] = toPx(x, y); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    // radio links: colour = what the network did with the latest POSE_UPDATE
    const lastPose = {};
    for (const e of state.messages) if (e.type === 'POSE_UPDATE') lastPose[e.receiver] = e.status;
    ctx.lineWidth = 1.6 * view.dpr;
    for (const a of agents) {
      if (a.id === 0) continue;
      const p = displayPose(a);
      const [x0, y0] = toPx(lead.x, lead.y), [x1, y1] = toPx(p.x, p.y);
      if (a.link === 'OFFLINE') { ctx.strokeStyle = PACKET_COLORS.LOST; ctx.setLineDash([2 * view.dpr, 4 * view.dpr]); }
      else { ctx.strokeStyle = PACKET_COLORS[lastPose[a.id] || 'OK']; ctx.setLineDash(a.link === 'DEGRADED' ? [7 * view.dpr, 4 * view.dpr] : []); }
      ctx.globalAlpha = 0.55; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    }
    ctx.setLineDash([]); ctx.globalAlpha = 1;

    // dotted line agent -> its own target
    ctx.lineWidth = view.dpr; ctx.setLineDash([2 * view.dpr, 4 * view.dpr]);
    for (const a of agents) {
      if (!a.target) continue;
      const p = displayPose(a), [x0, y0] = toPx(p.x, p.y), [x1, y1] = toPx(a.target[0], a.target[1]);
      ctx.strokeStyle = colorOf(a.id); ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    }
    ctx.setLineDash([]);

    // formation slots (hollow circles)
    ctx.strokeStyle = css('--muted'); ctx.lineWidth = 1.4 * view.dpr;
    for (const a of agents) {
      if (!a.slot) continue;
      const [px, py] = toPx(a.slot[0], a.slot[1]);
      ctx.beginPath(); ctx.arc(px, py, 0.28 * s, 0, 7); ctx.stroke();
    }

    // tasks
    ctx.font = `bold ${12 * view.dpr}px system-ui, sans-serif`; ctx.textAlign = 'center';
    for (const t of state.tasks) {
      const [px, py] = toPx(t.x, t.y), r = 0.3 * s;
      ctx.fillStyle = TASK_COLORS[t.status] || '#bbb'; ctx.strokeStyle = '#000'; ctx.lineWidth = view.dpr;
      ctx.fillRect(px - r, py - r, 2 * r, 2 * r); ctx.strokeRect(px - r, py - r, 2 * r, 2 * r);
      ctx.fillStyle = css('--text'); ctx.fillText(t.id, px, py + r + 13 * view.dpr);
    }

    // target markers (GOTO star, scatter X, pending scatter x)
    for (const t of state.targets) {
      const [px, py] = toPx(t.x, t.y);
      if (t.kind === 'goto') star(px, py, 0.55 * s, '#e41a1c');
      else if (t.kind === 'scatter') cross(px, py, 0.32 * s, '#984ea3', 3);
      else if (t.kind === 'scatter_pending') cross(px, py, 0.3 * s, '#888888', 2);
    }

    // robots
    ctx.font = `bold ${12 * view.dpr}px system-ui, sans-serif`;
    for (const a of agents) {
      const p = displayPose(a), [px, py] = toPx(p.x, p.y), k = s;
      const pts = [[0.50, 0], [-0.34, 0.30], [-0.20, 0], [-0.34, -0.30]];
      const c = Math.cos(p.th), sn = Math.sin(p.th);
      ctx.beginPath();
      pts.forEach(([u, v], i) => {
        const wx = u * c - v * sn, wy = u * sn + v * c;       // world offset (m)
        const X = px + wx * k, Y = py - wy * k;
        i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y);
      });
      ctx.closePath();
      ctx.fillStyle = colorOf(a.id); ctx.fill();
      ctx.strokeStyle = a.mode === 'STOPPED' ? '#c0392b' : '#000';
      ctx.lineWidth = (a.mode === 'STOPPED' ? 2.5 : 1) * view.dpr; ctx.stroke();
      ctx.fillStyle = css('--text'); ctx.fillText(a.short, px, py - 0.62 * s);
    }
    ctx.textAlign = 'start';
  }

  function star(x, y, r, color) {
    ctx.beginPath();
    for (let i = 0; i < 10; i++) {
      const rr = i % 2 ? r * 0.45 : r, ang = -Math.PI / 2 + i * Math.PI / 5;
      const X = x + rr * Math.cos(ang), Y = y + rr * Math.sin(ang);
      i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y);
    }
    ctx.closePath(); ctx.fillStyle = color; ctx.fill(); ctx.strokeStyle = '#000'; ctx.lineWidth = view.dpr; ctx.stroke();
  }
  function cross(x, y, r, color, w) {
    ctx.strokeStyle = color; ctx.lineWidth = w * view.dpr;
    ctx.beginPath(); ctx.moveTo(x - r, y - r); ctx.lineTo(x + r, y + r); ctx.moveTo(x - r, y + r); ctx.lineTo(x + r, y - r); ctx.stroke();
  }

  // ------------------------------------------------------------------- panels
  function onState(st) {
    if (st.time < lastTime) for (const k in trails) delete trails[k];     // simulation was reset
    if (st.time !== lastTime) {
      for (const a of st.agents) {
        const tr = trails[a.id] || (trails[a.id] = []);
        tr.push([a.x, a.y]); if (tr.length > TRAIL_LEN) tr.shift();
      }
    }
    lastTime = st.time;
    state = st; stateWall = performance.now();
    if (!view.w) layoutCanvas();
    updatePanels(st);
  }

  function syncSelect(sel, value) { if (document.activeElement !== sel && sel.value !== value) sel.value = value; }

  function updatePanels(st) {
    $('chip-time').textContent = `t = ${fmt(st.time, 1)} s`;
    $('chip-mode').textContent = `MODE ${st.sim_mode}`;
    $('chip-mission').textContent = `MISSION ${st.mission_state}`;
    $('chip-form').textContent = `FORMATION ${st.formation.name}`;
    $('btn-pause').textContent = st.runner.paused ? 'Resume clock' : 'Pause';
    syncSelect($('sel-simmode'), st.sim_mode);
    syncSelect($('sel-speed'), String(st.runner.speed));
    $('chk-loss').checked = !!st.network.forced_loss;

    const banner = $('halt-banner');
    banner.hidden = !st.halted;
    if (st.halted) banner.textContent = 'SIMULATION HALTED: ' + (st.fault || '');

    // agent selector (rebuilt only if the agent count changes)
    if (st.agents.length !== agentCount) {
      agentCount = st.agents.length;
      const sel = $('sel-agent'), cur = sel.value;
      sel.innerHTML = '<option value="ALL">ALL</option><option value="LEADER">LEADER</option>' +
        st.agents.filter(a => a.id > 0).map(a => `<option value="${a.id}">Agent ${a.id}</option>`).join('');
      sel.value = [...sel.options].some(o => o.value === cur) ? cur : 'ALL';
    }

    // agents table
    $('tbl-agents').querySelector('tbody').innerHTML = st.agents.map(a =>
      `<tr><td><span class="swatch" style="background:${colorOf(a.id)}"></span>${esc(a.short)}</td>` +
      `<td>${esc(a.mode)}</td><td>${fmt(a.x)}</td><td>${fmt(a.y)}</td><td>${fmt(a.v, 2)}</td>` +
      `<td class="link-${esc(a.link)}">${esc(a.link)}</td><td>${esc(a.task || '')}</td></tr>`).join('');

    // network
    const n = st.network;
    $('net-stats').innerHTML =
      `<div><b>sent</b> ${n.sent}</div><div><b>received</b> ${n.received}</div>` +
      `<div><b>lost</b> ${n.lost}</div><div><b>loss rate</b> ${fmt(n.loss_rate)} %</div>` +
      `<div><b>latency avg</b> ${fmt(n.avg_latency * 1000, 0)} ms</div><div><b>latency max</b> ${fmt(n.max_latency * 1000, 0)} ms</div>` +
      `<div><b>awaiting ACK</b> ${st.pending_commands}</div><div><b>boundary clamps</b> ${st.boundary_clamps}</div>` +
      `<div><b>form. error</b> ${fmt(st.formation_rms, 2)} m</div><div><b>min separation</b> ${fmt(st.min_distance, 2)} m</div>`;

    // tasks
    $('task-list').innerHTML = st.tasks.length ? st.tasks.map(t =>
      `<li class="st-${esc(t.status)}">Task ${esc(t.id)} (${fmt(t.x, 0)},${fmt(t.y, 0)}) p${t.priority} ` +
      `&rarr; ${t.agent === null ? 'unassigned' : 'Agent ' + t.agent} [${esc(t.status)}]</li>`).join('')
      : '<li class="muted">(no tasks)</li>';

    // communication log (real events); POSE_UPDATE stream summarised in one line
    const ev = st.messages.filter(e => e.type !== 'POSE_UPDATE').slice(-14);
    const pose = st.messages.filter(e => e.type === 'POSE_UPDATE');
    const lines = ev.map(e => {
      const tail = e.status === 'DELAYED' ? `DELAY=${fmt(e.latency, 2)}s` : e.status;
      const cls = e.status === 'LOST' ? 'lost' : e.status === 'DELAYED' ? 'delay' : '';
      return `<span class="${cls}">${fmt(e.t, 2).padStart(7)}  ${esc(e.from).padStart(2)}-&gt;${esc(e.to).padEnd(2)} ${esc(e.type).padEnd(11)} ${esc(tail)}</span>`;
    });
    if (pose.length) {
      const lost = pose.filter(e => e.status === 'LOST').length;
      lines.push(`-- POSE_UPDATE stream (last ${pose.length}): ${pose.length - lost} ok / ${lost} lost`);
    }
    setLog($('comm-log'), lines.join('\n') || '(no events yet)');

    // command history + events
    setLog($('cmd-history'), st.command_history.slice(0, 10).map(h =>
      `<span class="${h.ok ? '' : 'err'}">[${esc(h.clock)}] ${h.ok ? 'ok ' : 'ERR'} ${esc(h.text)}  ${esc(h.message)}</span>`).join('\n')
      || '(no commands yet)', false);
    setLog($('status-log'), st.status_log.slice(-8).map(e => `${fmt(e.t, 1).padStart(6)}  ${esc(e.text)}`).join('\n') || '(none)');

    // keyboard indicators
    document.querySelectorAll('#keys kbd[data-k]').forEach(k =>
      k.classList.toggle('down', st.keyboard[k.dataset.k] === 'PRESSED'));
    $('manual-cmd').textContent = `v_cmd ${fmt(st.manual_command.v, 2)}  w_cmd ${fmt(st.manual_command.omega, 2)}`;
  }

  function setLog(el, html, stickBottom = true) {
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
    el.innerHTML = html;
    if (stickBottom && atBottom) el.scrollTop = el.scrollHeight;
  }

  // ----------------------------------------------------------------- transport
  function connect() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.onopen = () => { setConn(true); stopPolling(); };
    ws.onmessage = (m) => { try { onState(JSON.parse(m.data)); } catch (e) { console.error(e); } };
    ws.onclose = () => { setConn(false); startPolling(); setTimeout(connect, 1500); };
    ws.onerror = () => { try { ws.close(); } catch (e) { /* ignore */ } };
  }
  function setConn(up) {
    const c = $('chip-conn');
    c.dataset.state = up ? 'up' : 'down';
    c.textContent = up ? 'live' : 'reconnecting...';
  }
  function startPolling() {
    if (pollTimer) return;
    pollTimer = setInterval(async () => {
      try { onState(await (await fetch('/state')).json()); } catch (e) { /* server down */ }
    }, 200);
  }
  function stopPolling() { clearInterval(pollTimer); pollTimer = null; }

  // -------------------------------------------------------------------- input
  document.querySelectorAll('button[data-cmd]').forEach(b => b.addEventListener('click', () => {
    const c = b.dataset.cmd;
    if (c === 'stop' || c === 'resume') api('/' + c, { agent: selAgent() });
    else api('/' + c, {});
  }));
  $('sel-simmode').addEventListener('change', (e) => api('/mode', { mode: e.target.value }));
  $('sel-formation').addEventListener('change', (e) => {
    if (e.target.value) api('/formation/' + e.target.value, {});
    e.target.value = '';
  });
  $('form-goto').addEventListener('submit', (e) => {
    e.preventDefault();
    api('/goto', { x: parseFloat($('goto-x').value), y: parseFloat($('goto-y').value), agent: selAgent() });
  });
  $('form-cmd').addEventListener('submit', (e) => {
    e.preventDefault();
    const t = $('cmd-text').value.trim();
    if (t) api('/command', { text: t });
    $('cmd-text').value = '';
  });
  $('form-task').addEventListener('submit', (e) => {
    e.preventDefault();
    api('/tasks', { id: $('task-id').value.trim(), x: parseFloat($('task-x').value),
                    y: parseFloat($('task-y').value), priority: parseInt($('task-prio').value || '1', 10) });
  });
  $('btn-allocate').addEventListener('click', () => api('/tasks/allocate', {}));
  $('btn-cleartasks').addEventListener('click', () => api('/tasks/clear', {}));
  $('chk-loss').addEventListener('change', (e) => api('/network/loss', { forced: e.target.checked }));
  $('btn-pause').addEventListener('click', () => api('/sim/pause', { paused: !(state && state.runner.paused) }));
  $('sel-speed').addEventListener('change', (e) => api('/sim/speed', { speed: parseFloat(e.target.value) }));
  $('btn-reset').addEventListener('click', () => api('/sim/reset', {
    control_mode: $('sel-control').value, loss_scale: parseFloat($('in-loss').value || '1'),
    sim_mode: $('sel-simmode').value }));

  // click on the map -> GOTO for the selected agent
  canvas.addEventListener('pointerdown', (e) => {
    if (!state || e.button !== 0) return;
    const r = canvas.getBoundingClientRect();
    const [x, y] = toWorld((e.clientX - r.left) * (canvas.width / r.width), (e.clientY - r.top) * (canvas.height / r.height));
    api('/goto', { x: Math.round(x * 100) / 100, y: Math.round(y * 100) / 100, agent: selAgent() });
  });

  // keyboard (ignored while typing in a field)
  const typing = (t) => t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA');
  window.addEventListener('keydown', (e) => {
    if (typing(e.target) || e.ctrlKey || e.metaKey || e.altKey) return;
    const k = KEY_MAP[e.key.toLowerCase()];
    if (!k) return;
    e.preventDefault();
    if (e.repeat && ONE_SHOT.has(k)) return;
    if (!held.has(k) || ONE_SHOT.has(k)) api('/key', { key: k === 'SPACE' ? ' ' : k.toLowerCase(), down: true }).then(() => {});
    held.add(k);
  });
  window.addEventListener('keyup', (e) => {
    if (typing(e.target)) return;
    const k = KEY_MAP[e.key.toLowerCase()];
    if (!k) return;
    e.preventDefault();
    held.delete(k);
    api('/key', { key: k === 'SPACE' ? ' ' : k.toLowerCase(), down: false });
  });
  window.addEventListener('blur', () => {          // never leave a key stuck down
    for (const k of held) api('/key', { key: k === 'SPACE' ? ' ' : k.toLowerCase(), down: false });
    held.clear();
  });
  window.addEventListener('resize', () => { view.w = 0; });

  // ---------------------------------------------------------------------- go
  startPolling();      // first paint without waiting for the socket
  connect();
  requestAnimationFrame(draw);
})();