/* ═══════════════════════════════════════════════════════════════════════════
   FDS Viewer — Frontend Application Logic (viewer.js)
   Bootstrap 5 multi-page version
   ═══════════════════════════════════════════════════════════════════════════ */

// ── Global state ──────────────────────────────────────────────────────────
let SIM_PATH = localStorage.getItem('fds_sim_path') || '';
let sliceMeta = [];      // slice metadata array from /api/slices
let obstMeta  = [];      // obstruction metadata from /api/obstructions
let devcMeta  = [];      // device metadata from /api/devices
let hrrMeta   = {};      // HRR metadata from /api/hrr
let animFrames = [];     // animation frame array
let animIdx    = 0;
let animTimer  = null;
let animPlaying = false;

// ── Helpers ───────────────────────────────────────────────────────────────

function showLoading(msg) {
  const overlay = document.getElementById('loadingOverlay');
  if (!overlay) return;
  document.getElementById('loadingText').textContent = msg || 'Loading...';
  overlay.classList.remove('d-none');
}

function hideLoading() {
  const overlay = document.getElementById('loadingOverlay');
  if (overlay) overlay.classList.add('d-none');
}

async function apiPost(url, body) {
  const resp = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  const data = await resp.json();
  if (!resp.ok || data.error) throw new Error(data.error || 'API error');
  return data;
}

async function apiGet(url) {
  const resp = await fetch(url);
  const data = await resp.json();
  if (!resp.ok || data.error) throw new Error(data.error || 'API error');
  return data;
}

function numOrNull(id) {
  const el = document.getElementById(id);
  if (!el) return null;
  const v = el.value;
  return (v === '' || v === null) ? null : parseFloat(v);
}

function showPlot(imgSrc, title) {
  const img = document.getElementById('plotImage');
  const ph  = document.getElementById('placeholderMsg');
  const badge = document.getElementById('displayBadge');
  if (ph)    ph.classList.add('d-none');
  if (img)   { img.src = imgSrc; img.classList.remove('d-none'); }
  if (badge) { badge.textContent = title || 'Rendered'; badge.className = 'badge bg-success'; }
}

function updateNavStatus() {
  const el = document.getElementById('navSimStatus');
  if (!el) return;
  if (SIM_PATH) {
    const name = SIM_PATH.split(/[/\\]/).filter(Boolean).pop();
    el.innerHTML = `<span class="badge bg-success"><i class="bi bi-check-circle"></i> ${name}</span>`;
  }
}


// ═══════════════════════════════════════════════════════════════════════════
//  INDEX PAGE — Load Simulation
// ═══════════════════════════════════════════════════════════════════════════

async function loadSimulation() {
  const pathEl = document.getElementById('simPath');
  if (!pathEl) return;
  const path = pathEl.value.trim();
  if (!path) return alert('Enter a simulation directory path');

  showLoading('Loading simulation data...');
  try {
    const data = await apiPost('/api/load', { path });
    SIM_PATH = path;
    localStorage.setItem('fds_sim_path', path);
    updateNavStatus();

    // Show summary card
    const card = document.getElementById('simSummaryCard');
    const grid = document.getElementById('simSummaryGrid');
    if (card && grid) {
      card.classList.remove('d-none');
      const info = data.info;
      grid.innerHTML = `
        <div class="col"><div class="p-2"><h4 class="text-info mb-0">${info.chid}</h4><small class="text-secondary">CHID</small></div></div>
        <div class="col"><div class="p-2"><h4 class="mb-0">${info.meshes}</h4><small class="text-secondary">Meshes</small></div></div>
        <div class="col"><div class="p-2"><h4 class="mb-0">${info.n_slices}</h4><small class="text-secondary">Slices</small></div></div>
        <div class="col"><div class="p-2"><h4 class="mb-0">${info.n_obstructions}</h4><small class="text-secondary">BNDF</small></div></div>
        <div class="col"><div class="p-2"><h4 class="mb-0">${info.n_devices}</h4><small class="text-secondary">Devices</small></div></div>
        <div class="col"><div class="p-2"><h4 class="mb-0">${info.n_hrr_columns || 0}</h4><small class="text-secondary">HRR</small></div></div>
        <div class="col"><div class="p-2"><h4 class="mb-0">${info.n_plot3d || 0}</h4><small class="text-secondary">Plot3D</small></div></div>
        <div class="col"><div class="p-2"><h4 class="mb-0">${info.n_smoke3d || 0}</h4><small class="text-secondary">Smoke3D</small></div></div>
        <div class="col"><div class="p-2"><h4 class="mb-0">${info.n_particles || 0}</h4><small class="text-secondary">Particles</small></div></div>
        <div class="col"><div class="p-2"><h4 class="mb-0">${info.n_isosurfaces || 0}</h4><small class="text-secondary">Isosurfaces</small></div></div>
      `;
    }
    hideLoading();
  } catch (e) {
    hideLoading();
    alert('Error: ' + e.message);
  }
}


// ═══════════════════════════════════════════════════════════════════════════
//  SLICE VIEWER
// ═══════════════════════════════════════════════════════════════════════════

async function fetchSliceList() {
  if (!SIM_PATH) return;
  updateNavStatus();
  const sel = document.getElementById('sliceSelect');
  if (!sel) return;
  try {
    sliceMeta = await apiGet('/api/slices?path=' + encodeURIComponent(SIM_PATH));
    sel.innerHTML = '<option value="">-- Select a slice --</option>';
    sliceMeta.forEach((s, i) => {
      const opt = document.createElement('option');
      opt.value = i;
      const idLabel = s.id ? ` ID="${s.id}"` : '';
      const orientNames = { 0: '3D', 1: 'X-normal', 2: 'Y-normal', 3: 'Z-normal' };
      const oName = orientNames[s.orientation] || `orient=${s.orientation}`;
      opt.textContent = `Slice ${i}${idLabel} — ${s.quantity} (${oName})`;
      sel.appendChild(opt);
    });
  } catch (e) {
    console.error('Failed to load slices:', e);
  }
}

function _getSelectedSlice() {
  const idx = parseInt(document.getElementById('sliceSelect').value);
  if (isNaN(idx)) return null;
  return sliceMeta[idx];
}

function onSliceSelected() {
  const s = _getSelectedSlice();
  const panel = document.getElementById('sliceInfoPanel');
  if (!s) { if (panel) panel.classList.add('d-none'); return; }
  if (panel) panel.classList.remove('d-none');

  // Info box
  const info = document.getElementById('sliceInfoBox');
  if (info) {
    const idLine = s.id ? `<strong>ID:</strong> ${s.id} | ` : '';
    info.innerHTML =
      `${idLine}<strong>${s.quantity}</strong> (${s.unit})<br>` +
      `Type: ${s.type} | Orient: ${s.orientation} | Cell-centered: ${s.cell_centered}<br>` +
      `Extent: [${s.extent.map(v => v.toFixed(2)).join(', ')}]<br>` +
      `Meshes: ${s.n_meshes} | Timesteps: ${s.n_timesteps} (${s.t_start.toFixed(1)}–${s.t_end.toFixed(1)} s)`;
  }

  // Populate timestep selectors
  _populateTimestepSelect('tsSelect', s.times);
  _populateTimestepSelect('tsMultiSelect', s.times);

  // Mesh selector
  const meshSel = document.getElementById('meshSelect');
  if (meshSel && s.meshes) {
    meshSel.innerHTML = '';
    s.meshes.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.mesh_index;
      opt.textContent = `Mesh ${m.mesh_index}` + (m.shape.length ? ` (${m.shape.join('x')})` : '');
      meshSel.appendChild(opt);
    });
  }

  // Profile directions
  const dirSel = document.getElementById('profileDir');
  if (dirSel && s.extent_dirs) {
    dirSel.innerHTML = '';
    s.extent_dirs.forEach(d => {
      const opt = document.createElement('option');
      opt.value = d; opt.textContent = d;
      dirSel.appendChild(opt);
    });
  }

  // Time-series coord labels
  if (s.extent_dirs && s.extent_dirs.length >= 2) {
    const l0 = document.getElementById('tsCoord0Label');
    const l1 = document.getElementById('tsCoord1Label');
    if (l0) l0.textContent = s.extent_dirs[0] + ' (m)';
    if (l1) l1.textContent = s.extent_dirs[1] + ' (m)';
  }

  // Reset view mode
  onViewModeChange();
}

function _populateTimestepSelect(selId, times) {
  const sel = document.getElementById(selId);
  if (!sel) return;
  sel.innerHTML = '';
  times.forEach(t => {
    const opt = document.createElement('option');
    opt.value = t;
    opt.textContent = `t = ${t.toFixed(1)} s`;
    sel.appendChild(opt);
  });
}

function onViewModeChange() {
  const mode = document.getElementById('viewMode')?.value || 'single';
  const panels = ['panelSingle', 'panelMulti', 'panelAnimation', 'panelProfile', 'panelTimeseries'];
  panels.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.classList.add('d-none');
  });
  const target = {
    single: 'panelSingle', multi: 'panelMulti', animation: 'panelAnimation',
    profile: 'panelProfile', timeseries: 'panelTimeseries'
  }[mode];
  if (target) document.getElementById(target)?.classList.remove('d-none');
}

function onModeChange() {
  const isGlobal = document.getElementById('modeGlobal')?.checked;
  const meshGrp = document.getElementById('meshSelectGroup');
  if (meshGrp) {
    meshGrp.classList.toggle('d-none', isGlobal);
  }
}

function _slicePayload() {
  const idx = parseInt(document.getElementById('sliceSelect').value);
  const s = sliceMeta[idx];
  const isGlobal = document.getElementById('modeGlobal')?.checked !== false;
  const payload = { path: SIM_PATH, slice_id: idx };
  if (s && s.id) payload.slice_id = s.id;
  else payload.slice_id = idx;
  payload.use_global = isGlobal;
  if (!isGlobal) {
    payload.mesh_index = parseInt(document.getElementById('meshSelect')?.value) || 0;
  }
  payload.cmap = document.getElementById('sliceCmap')?.value || 'jet';
  payload.vmin = numOrNull('sliceVMin');
  payload.vmax = numOrNull('sliceVMax');
  payload.show_colorbar = document.getElementById('showColorbar')?.checked !== false;
  payload.show_labels = document.getElementById('showLabels')?.checked !== false;
  return payload;
}


// ── Single Slice Render ───────────────────────────────────────────────────
async function renderSingleSlice() {
  const s = _getSelectedSlice();
  if (!s) return alert('Select a slice first');
  const t = parseFloat(document.getElementById('tsSelect').value);

  showLoading('Rendering slice...');
  try {
    const payload = _slicePayload();
    payload.timestep = t;
    const data = await apiPost('/api/slice/render', payload);
    showPlot(data.image_b64, `${s.quantity} | t = ${data.actual_time.toFixed(1)} s`);
    hideLoading();
  } catch (e) { hideLoading(); alert(e.message); }
}

// ── Multi Slice Render ────────────────────────────────────────────────────
async function renderMultiSlice() {
  const s = _getSelectedSlice();
  if (!s) return alert('Select a slice first');
  const sel = document.getElementById('tsMultiSelect');
  const times = Array.from(sel.selectedOptions).map(o => parseFloat(o.value));
  if (times.length < 2) return alert('Select at least 2 timesteps (hold Ctrl)');

  showLoading('Generating multi-time grid...');
  try {
    const payload = _slicePayload();
    payload.timesteps = times;
    const data = await apiPost('/api/slice/render_multi', payload);
    showPlot(data.image_b64, `${s.quantity} — ${times.length} timesteps`);
    hideLoading();
  } catch (e) { hideLoading(); alert(e.message); }
}

// ── Slice Animation ───────────────────────────────────────────────────────
async function generateSliceAnimation() {
  const s = _getSelectedSlice();
  if (!s) return alert('Select a slice first');

  showLoading('Generating animation frames...');
  try {
    const payload = _slicePayload();
    payload.t_start = numOrNull('animTStart') || 0;
    payload.t_end = numOrNull('animTEnd');
    payload.n_frames = parseInt(document.getElementById('animFrames').value) || 20;
    const data = await apiPost('/api/slice/animation_frames', payload);
    animFrames = data.frames;
    animIdx = 0;
    hideLoading();

    // Show play controls
    const pc = document.getElementById('animPlayControls');
    if (pc) pc.classList.remove('d-none');
    const scrubber = document.getElementById('animScrubber');
    if (scrubber) { scrubber.max = animFrames.length - 1; scrubber.value = 0; }
    _showAnimFrame(0);
  } catch (e) { hideLoading(); alert(e.message); }
}

function _showAnimFrame(i) {
  if (!animFrames[i]) return;
  showPlot(animFrames[i].image_b64, `Frame ${i + 1}/${animFrames.length} | t = ${animFrames[i].time.toFixed(1)} s`);
  const label = document.getElementById('animTimeLabel');
  if (label) label.textContent = `t = ${animFrames[i].time.toFixed(1)} s  [${i + 1}/${animFrames.length}]`;
  const scrubber = document.getElementById('animScrubber');
  if (scrubber) scrubber.value = i;
}

function toggleSlicePlayback() {
  animPlaying = !animPlaying;
  const icon = document.getElementById('btnPlayIcon');
  const text = document.getElementById('btnPlayText');
  if (animPlaying) {
    if (icon) icon.className = 'bi bi-pause-fill';
    if (text) text.textContent = 'Pause';
    const fps = parseInt(document.getElementById('animFPS').value) || 5;
    animTimer = setInterval(() => {
      animIdx = (animIdx + 1) % animFrames.length;
      _showAnimFrame(animIdx);
    }, 1000 / fps);
  } else {
    if (icon) icon.className = 'bi bi-play-fill';
    if (text) text.textContent = 'Play';
    if (animTimer) { clearInterval(animTimer); animTimer = null; }
  }
}

function scrubSliceAnim(val) {
  animIdx = parseInt(val);
  _showAnimFrame(animIdx);
}

async function downloadSliceGIF() {
  const s = _getSelectedSlice();
  if (!s) return;
  showLoading('Generating GIF...');
  try {
    const payload = _slicePayload();
    payload.t_start = numOrNull('animTStart') || 0;
    payload.t_end = numOrNull('animTEnd');
    payload.n_frames = parseInt(document.getElementById('animFrames').value) || 20;
    payload.fps = parseInt(document.getElementById('animFPS').value) || 5;
    const resp = await fetch('/api/download/slice/gif', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!resp.ok) { const err = await resp.json(); throw new Error(err.error); }
    const blob = await resp.blob();
    const fname = resp.headers.get('Content-Disposition')?.match(/filename="?(.+?)"?$/)?.[1] || 'slice_animation.gif';
    _downloadBlob(blob, fname);
    hideLoading();
  } catch (e) { hideLoading(); alert(e.message); }
}

// ── Profile ───────────────────────────────────────────────────────────────
async function renderSliceProfile() {
  const s = _getSelectedSlice();
  if (!s) return alert('Select a slice first');
  showLoading('Extracting profile...');
  try {
    const payload = _slicePayload();
    payload.direction = document.getElementById('profileDir').value;
    payload.position = parseFloat(document.getElementById('profilePos').value) || 0;
    payload.time = parseFloat(document.getElementById('profileTime').value) || 100;
    const data = await apiPost('/api/slice/profile', payload);
    showPlot(data.image_b64, 'Line Profile');
    hideLoading();
  } catch (e) { hideLoading(); alert(e.message); }
}

// ── Time-series ───────────────────────────────────────────────────────────
async function renderSliceTimeseries() {
  const s = _getSelectedSlice();
  if (!s) return alert('Select a slice first');
  const point = {};
  const v0 = parseFloat(document.getElementById('tsCoord0').value);
  const v1 = parseFloat(document.getElementById('tsCoord1').value);
  if (!isNaN(v0) && s.extent_dirs[0]) point[s.extent_dirs[0]] = v0;
  if (!isNaN(v1) && s.extent_dirs[1]) point[s.extent_dirs[1]] = v1;

  showLoading('Extracting time-series...');
  try {
    const payload = _slicePayload();
    payload.point = point;
    const data = await apiPost('/api/slice/timeseries', payload);
    showPlot(data.image_b64, 'Point Time-Series');
    hideLoading();
  } catch (e) { hideLoading(); alert(e.message); }
}


// ═══════════════════════════════════════════════════════════════════════════
//  BOUNDARY VIEWER
// ═══════════════════════════════════════════════════════════════════════════

async function fetchObstructionList() {
  if (!SIM_PATH) return;
  updateNavStatus();
  const sel = document.getElementById('obstSelect');
  if (!sel) return;
  try {
    obstMeta = await apiGet('/api/obstructions?path=' + encodeURIComponent(SIM_PATH));
    sel.innerHTML = '<option value="">-- Select obstruction --</option>';
    obstMeta.forEach((o, i) => {
      const opt = document.createElement('option');
      opt.value = i;
      const bbStr = o.bounding_box.length ? ` [${o.bounding_box.map(v => v.toFixed(1)).join(', ')}]` : '';
      opt.textContent = `Obst "${o.id}"${bbStr} — ${o.quantities.join(', ')}`;
      sel.appendChild(opt);
    });
  } catch (e) {
    console.error('Failed to load obstructions:', e);
  }
}

function _getSelectedObst() {
  const idx = parseInt(document.getElementById('obstSelect')?.value);
  if (isNaN(idx)) return null;
  return obstMeta[idx];
}

function onObstSelected() {
  const o = _getSelectedObst();
  const panel = document.getElementById('bndfInfoPanel');
  if (!o) { if (panel) panel.classList.add('d-none'); return; }
  if (panel) panel.classList.remove('d-none');

  // Info box
  const info = document.getElementById('bndfInfoBox');
  if (info) {
    info.innerHTML =
      `<strong>ID:</strong> ${o.id} | Index: ${o.index}<br>` +
      `Box: [${o.bounding_box.map(v => v.toFixed(2)).join(', ')}]<br>` +
      `Quantities: ${o.quantities.join(', ')}<br>` +
      `Orientations: ${o.orientation_labels.join(', ')}<br>` +
      `Meshes: ${o.n_meshes} | Timesteps: ${o.n_timesteps}`;
  }

  // Quantity dropdown
  const qtySel = document.getElementById('bndfQty');
  if (qtySel) {
    qtySel.innerHTML = '';
    o.quantities.forEach((q, i) => {
      const opt = document.createElement('option');
      opt.value = q;
      opt.textContent = q + (o.quantity_units && o.quantity_units[i] ? ` (${o.quantity_units[i]})` : '');
      qtySel.appendChild(opt);
    });
  }

  // Orientation dropdown
  const oSel = document.getElementById('bndfOrient');
  if (oSel) {
    oSel.innerHTML = '';
    o.orientations.forEach((ov, i) => {
      const opt = document.createElement('option');
      opt.value = ov;
      opt.textContent = o.orientation_labels[i] || ov;
      oSel.appendChild(opt);
    });
  }

  // Timestep selectors
  _populateTimestepSelect('bndfTsSelect', o.times);
  _populateTimestepSelect('bndfTsMultiSelect', o.times);

  onBndfViewModeChange();
}

function onBndfViewModeChange() {
  const mode = document.getElementById('bndfViewMode')?.value || 'single';
  const panels = ['bndfPanelSingle', 'bndfPanelMulti', 'bndfPanelAnimation', 'bndfPanelTimeseries'];
  panels.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.classList.add('d-none');
  });
  const target = {
    single: 'bndfPanelSingle', multi: 'bndfPanelMulti',
    animation: 'bndfPanelAnimation', timeseries: 'bndfPanelTimeseries'
  }[mode];
  if (target) document.getElementById(target)?.classList.remove('d-none');
}

function _bndfPayload() {
  const o = _getSelectedObst();
  return {
    path: SIM_PATH,
    obst_id: o ? (o.id || o.index) : 0,
    quantity: document.getElementById('bndfQty')?.value || '',
    orientation: parseInt(document.getElementById('bndfOrient')?.value) || 3,
    cmap: document.getElementById('bndfCmap')?.value || 'hot',
    vmin: numOrNull('bndfVMin'),
    vmax: numOrNull('bndfVMax'),
    show_colorbar: document.getElementById('bndfShowColorbar')?.checked !== false,
  };
}

// ── Single Boundary Render ────────────────────────────────────────────────
async function renderSingleBoundary() {
  const o = _getSelectedObst();
  if (!o) return alert('Select an obstruction first');
  const t = parseFloat(document.getElementById('bndfTsSelect').value);

  showLoading('Rendering boundary...');
  try {
    const payload = _bndfPayload();
    payload.timestep = t;
    const data = await apiPost('/api/boundary/render', payload);
    showPlot(data.image_b64, `${payload.quantity} | t = ${data.actual_time.toFixed(1)} s`);
    hideLoading();
  } catch (e) { hideLoading(); alert(e.message); }
}

// ── Multi Boundary Render ─────────────────────────────────────────────────
async function renderMultiBoundary() {
  const o = _getSelectedObst();
  if (!o) return alert('Select an obstruction first');
  const sel = document.getElementById('bndfTsMultiSelect');
  const times = Array.from(sel.selectedOptions).map(op => parseFloat(op.value));
  if (times.length < 2) return alert('Select at least 2 timesteps');

  showLoading('Generating multi-time grid...');
  try {
    const payload = _bndfPayload();
    payload.timesteps = times;
    const data = await apiPost('/api/boundary/render_multi', payload);
    showPlot(data.image_b64, `${payload.quantity} — ${times.length} timesteps`);
    hideLoading();
  } catch (e) { hideLoading(); alert(e.message); }
}

// ── Boundary Animation ────────────────────────────────────────────────────
async function generateBndfAnimation() {
  const o = _getSelectedObst();
  if (!o) return alert('Select an obstruction first');

  showLoading('Generating boundary animation...');
  try {
    const payload = _bndfPayload();
    payload.t_start = numOrNull('bndfAnimTStart') || 0;
    payload.t_end = numOrNull('bndfAnimTEnd');
    payload.n_frames = parseInt(document.getElementById('bndfAnimFrames').value) || 20;
    const data = await apiPost('/api/boundary/animation_frames', payload);
    animFrames = data.frames;
    animIdx = 0;
    hideLoading();

    const pc = document.getElementById('bndfAnimPlayControls');
    if (pc) pc.classList.remove('d-none');
    const scrubber = document.getElementById('bndfAnimScrubber');
    if (scrubber) { scrubber.max = animFrames.length - 1; scrubber.value = 0; }
    _showBndfAnimFrame(0);
  } catch (e) { hideLoading(); alert(e.message); }
}

function _showBndfAnimFrame(i) {
  if (!animFrames[i]) return;
  showPlot(animFrames[i].image_b64, `Frame ${i + 1}/${animFrames.length} | t = ${animFrames[i].time.toFixed(1)} s`);
  const label = document.getElementById('bndfAnimTimeLabel');
  if (label) label.textContent = `t = ${animFrames[i].time.toFixed(1)} s  [${i + 1}/${animFrames.length}]`;
  const scrubber = document.getElementById('bndfAnimScrubber');
  if (scrubber) scrubber.value = i;
}

function toggleBndfPlayback() {
  animPlaying = !animPlaying;
  const icon = document.getElementById('bndfBtnPlayIcon');
  const text = document.getElementById('bndfBtnPlayText');
  if (animPlaying) {
    if (icon) icon.className = 'bi bi-pause-fill';
    if (text) text.textContent = 'Pause';
    const fps = parseInt(document.getElementById('bndfAnimFPS').value) || 5;
    animTimer = setInterval(() => {
      animIdx = (animIdx + 1) % animFrames.length;
      _showBndfAnimFrame(animIdx);
    }, 1000 / fps);
  } else {
    if (icon) icon.className = 'bi bi-play-fill';
    if (text) text.textContent = 'Play';
    if (animTimer) { clearInterval(animTimer); animTimer = null; }
  }
}

function scrubBndfAnim(val) {
  animIdx = parseInt(val);
  _showBndfAnimFrame(animIdx);
}

async function downloadBndfGIF() {
  const o = _getSelectedObst();
  if (!o) return;
  showLoading('Generating boundary GIF...');
  try {
    const payload = _bndfPayload();
    payload.t_start = numOrNull('bndfAnimTStart') || 0;
    payload.t_end = numOrNull('bndfAnimTEnd');
    payload.n_frames = parseInt(document.getElementById('bndfAnimFrames').value) || 20;
    payload.fps = parseInt(document.getElementById('bndfAnimFPS').value) || 5;
    const resp = await fetch('/api/download/boundary/gif', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!resp.ok) { const err = await resp.json(); throw new Error(err.error); }
    const blob = await resp.blob();
    const fname = resp.headers.get('Content-Disposition')?.match(/filename="?(.+?)"?$/)?.[1] || 'boundary_animation.gif';
    _downloadBlob(blob, fname);
    hideLoading();
  } catch (e) { hideLoading(); alert(e.message); }
}

// ── Boundary Time-Series ──────────────────────────────────────────────────
async function renderBndfTimeseries() {
  const o = _getSelectedObst();
  if (!o) return alert('Select an obstruction first');

  showLoading('Extracting boundary time-series...');
  try {
    const data = await apiPost('/api/boundary/timeseries', _bndfPayload());
    showPlot(data.image_b64, 'Boundary Time-Series');
    hideLoading();
  } catch (e) { hideLoading(); alert(e.message); }
}


// ═══════════════════════════════════════════════════════════════════════════
//  DEVICE VIEWER
// ═══════════════════════════════════════════════════════════════════════════

async function fetchDeviceList() {
  if (!SIM_PATH) return;
  updateNavStatus();
  const checkboxList = document.getElementById('devcCheckboxList');
  if (!checkboxList) return;
  try {
    devcMeta = await apiGet('/api/devices?path=' + encodeURIComponent(SIM_PATH));
    if (devcMeta.length === 0) {
      checkboxList.innerHTML = '<small class="text-secondary">No devices found in this simulation.</small>';
      return;
    }

    // Build quantity filter dropdown
    const qtyFilter = document.getElementById('devcQtyFilter');
    if (qtyFilter) {
      const quantities = [...new Set(devcMeta.map(d => d.quantity).filter(Boolean))];
      qtyFilter.innerHTML = '<option value="">All Quantities</option>';
      quantities.forEach(q => {
        const opt = document.createElement('option');
        opt.value = q; opt.textContent = q;
        qtyFilter.appendChild(opt);
      });
    }

    // Build checkbox lists
    _buildDevcCheckboxes(checkboxList, 'devc_', devcMeta);
    const rightList = document.getElementById('devcRightCheckboxList');
    if (rightList) _buildDevcCheckboxes(rightList, 'devcR_', devcMeta);

  } catch (e) {
    console.error('Failed to load devices:', e);
    checkboxList.innerHTML = '<small class="text-danger">Error loading devices.</small>';
  }
}

function _buildDevcCheckboxes(container, prefix, devices) {
  container.innerHTML = '';
  devices.forEach((dv, i) => {
    const div = document.createElement('div');
    div.className = 'form-check';
    div.dataset.quantity = dv.quantity || '';
    div.innerHTML = `
      <input class="form-check-input" type="checkbox" id="${prefix}${i}" value="${dv.id}">
      <label class="form-check-label small" for="${prefix}${i}">
        ${dv.id} <span class="text-secondary">(${dv.quantity}${dv.unit ? ' [' + dv.unit + ']' : ''})</span>
      </label>`;
    container.appendChild(div);
  });
}

function filterDeviceList() {
  const qty = document.getElementById('devcQtyFilter')?.value || '';
  const containers = ['devcCheckboxList', 'devcRightCheckboxList'];
  containers.forEach(cid => {
    const container = document.getElementById(cid);
    if (!container) return;
    container.querySelectorAll('.form-check').forEach(div => {
      if (!qty || div.dataset.quantity === qty) {
        div.style.display = '';
      } else {
        div.style.display = 'none';
      }
    });
  });
}

function _getCheckedDeviceIds(prefix) {
  const ids = [];
  document.querySelectorAll(`[id^="${prefix}"]`).forEach(cb => {
    if (cb.checked && cb.type === 'checkbox') ids.push(cb.value);
  });
  return ids;
}

function devcSelectAll() {
  const qty = document.getElementById('devcQtyFilter')?.value || '';
  document.querySelectorAll('[id^="devc_"]').forEach(cb => {
    if (cb.type !== 'checkbox') return;
    const div = cb.closest('.form-check');
    if (!qty || (div && div.dataset.quantity === qty)) {
      cb.checked = true;
    }
  });
}

function devcSelectNone() {
  document.querySelectorAll('[id^="devc_"]').forEach(cb => {
    if (cb.type === 'checkbox') cb.checked = false;
  });
}

function onDevcViewModeChange() {
  const mode = document.getElementById('devcViewMode')?.value || 'plot';
  const plotPanel = document.getElementById('devcPanelPlot');
  const cmpPanel = document.getElementById('devcPanelCompare');
  if (plotPanel) plotPanel.classList.toggle('d-none', mode !== 'plot');
  if (cmpPanel) cmpPanel.classList.toggle('d-none', mode !== 'compare');
}

async function renderDevicePlot() {
  const ids = _getCheckedDeviceIds('devc_');
  if (ids.length === 0) return alert('Select at least one device');

  showLoading('Plotting device data...');
  try {
    const tStart = numOrNull('devcTStart');
    const tEnd = numOrNull('devcTEnd');
    const timeRange = (tStart !== null || tEnd !== null) ? [tStart || 0, tEnd || 99999] : null;
    const data = await apiPost('/api/device/render', {
      path: SIM_PATH,
      device_ids: ids,
      time_range: timeRange,
      show_grid: document.getElementById('devcShowGrid')?.checked !== false,
    });
    showPlot(data.image_b64, `Devices: ${ids.length} plotted`);
    hideLoading();
  } catch (e) { hideLoading(); alert(e.message); }
}

async function renderDeviceComparison() {
  const idsLeft = _getCheckedDeviceIds('devc_');
  const idsRight = _getCheckedDeviceIds('devcR_');
  if (idsLeft.length === 0) return alert('Select at least one left-axis device (main checkboxes)');
  if (idsRight.length === 0) return alert('Select at least one right-axis device');

  showLoading('Generating comparison plot...');
  try {
    const tStart = numOrNull('devcCmpTStart');
    const tEnd = numOrNull('devcCmpTEnd');
    const timeRange = (tStart !== null || tEnd !== null) ? [tStart || 0, tEnd || 99999] : null;
    const data = await apiPost('/api/device/compare', {
      path: SIM_PATH,
      device_ids_left: idsLeft,
      device_ids_right: idsRight,
      time_range: timeRange,
    });
    showPlot(data.image_b64, `Comparison: ${idsLeft.length} vs ${idsRight.length}`);
    hideLoading();
  } catch (e) { hideLoading(); alert(e.message); }
}


// ═══════════════════════════════════════════════════════════════════════════
//  HRR VIEWER
// ═══════════════════════════════════════════════════════════════════════════

async function fetchHRRMeta() {
  if (!SIM_PATH) return;
  updateNavStatus();
  const checkboxList = document.getElementById('hrrCheckboxList');
  if (!checkboxList) return;
  try {
    hrrMeta = await apiGet('/api/hrr?path=' + encodeURIComponent(SIM_PATH));
    const infoBox = document.getElementById('hrrInfoBox');

    if (!hrrMeta.columns || hrrMeta.columns.length === 0) {
      checkboxList.innerHTML = '<small class="text-secondary">No HRR data found in this simulation.</small>';
      if (infoBox) infoBox.classList.add('d-none');
      return;
    }

    // Show info box
    if (infoBox) {
      infoBox.classList.remove('d-none');
      infoBox.innerHTML =
        `<strong>Columns:</strong> ${hrrMeta.columns.length}<br>` +
        `<strong>Time:</strong> ${(hrrMeta.t_start || 0).toFixed(1)} – ${(hrrMeta.t_end || 0).toFixed(1)} s<br>` +
        `<strong>Data points:</strong> ${hrrMeta.n_points || 'N/A'}`;
    }

    // Build checkboxes
    checkboxList.innerHTML = '';
    hrrMeta.columns.forEach((col, i) => {
      const div = document.createElement('div');
      div.className = 'form-check';
      const checked = (col === 'HRR') ? 'checked' : '';
      div.innerHTML = `
        <input class="form-check-input" type="checkbox" id="hrr_${i}" value="${col}" ${checked}>
        <label class="form-check-label small" for="hrr_${i}">${col}</label>`;
      checkboxList.appendChild(div);
    });

  } catch (e) {
    console.error('Failed to load HRR metadata:', e);
    if (checkboxList) checkboxList.innerHTML = '<small class="text-danger">Error loading HRR data.</small>';
  }
}

function _getCheckedHRRColumns() {
  const cols = [];
  document.querySelectorAll('[id^="hrr_"]').forEach(cb => {
    if (cb.checked && cb.type === 'checkbox') cols.push(cb.value);
  });
  return cols;
}

function hrrSelectAll() {
  document.querySelectorAll('[id^="hrr_"]').forEach(cb => {
    if (cb.type === 'checkbox') cb.checked = true;
  });
}

function hrrSelectNone() {
  document.querySelectorAll('[id^="hrr_"]').forEach(cb => {
    if (cb.type === 'checkbox') cb.checked = false;
  });
}

function hrrPreset(preset) {
  // First clear all
  hrrSelectNone();
  if (preset === 'HRR') {
    document.querySelectorAll('[id^="hrr_"]').forEach(cb => {
      if (cb.value === 'HRR') cb.checked = true;
    });
  } else if (preset === 'all_Q') {
    document.querySelectorAll('[id^="hrr_"]').forEach(cb => {
      if (cb.value.startsWith('Q_') || cb.value === 'HRR') cb.checked = true;
    });
  } else if (preset === 'all') {
    hrrSelectAll();
  }
}

async function renderHRRPlot() {
  const cols = _getCheckedHRRColumns();
  if (cols.length === 0) return alert('Select at least one HRR column');

  showLoading('Plotting HRR data...');
  try {
    const tStart = numOrNull('hrrTStart');
    const tEnd = numOrNull('hrrTEnd');
    const timeRange = (tStart !== null || tEnd !== null) ? [tStart || 0, tEnd || 99999] : null;
    const data = await apiPost('/api/hrr/render', {
      path: SIM_PATH,
      columns: cols,
      time_range: timeRange,
      show_grid: document.getElementById('hrrShowGrid')?.checked !== false,
    });
    showPlot(data.image_b64, `HRR: ${cols.join(', ')}`);
    hideLoading();
  } catch (e) { hideLoading(); alert(e.message); }
}


// ═══════════════════════════════════════════════════════════════════════════
//  DIRECTORY BROWSER
// ═══════════════════════════════════════════════════════════════════════════

let _browseCurrentPath = '';

function openBrowseModal() {
  const modal = new bootstrap.Modal(document.getElementById('browseModal'));
  modal.show();
  const startPath = document.getElementById('simPath')?.value || '';
  browseTo(startPath);
}

async function browseTo(path) {
  try {
    const data = await apiGet('/api/browse?path=' + encodeURIComponent(path || ''));
    _browseCurrentPath = data.path || '';
    document.getElementById('browsePath').value = _browseCurrentPath;
    const smvBadge = document.getElementById('browseSmvBadge');
    if (smvBadge) smvBadge.classList.toggle('d-none', !data.has_smv);
    if (data.has_smv) smvBadge.className = 'badge bg-success align-self-center';

    const list = document.getElementById('browseDirList');
    list.innerHTML = '';
    if (data.dirs.length === 0) {
      list.innerHTML = '<small class="text-secondary p-2">No subdirectories</small>';
      return;
    }
    data.dirs.forEach(d => {
      const fullPath = _browseCurrentPath ? (_browseCurrentPath.replace(/[\\/]$/, '') + '/' + d) : d;
      const a = document.createElement('a');
      a.href = '#';
      a.className = 'list-group-item list-group-item-action bg-dark text-light border-secondary';
      a.innerHTML = `<i class="bi bi-folder text-warning me-2"></i>${d}`;
      a.onclick = (e) => { e.preventDefault(); browseTo(fullPath); };
      list.appendChild(a);
    });
  } catch (e) {
    document.getElementById('browseDirList').innerHTML = `<small class="text-danger p-2">${e.message}</small>`;
  }
}

function browseUp() {
  const parts = _browseCurrentPath.replace(/[\\/]$/, '').split(/[\\/]/);
  parts.pop();
  browseTo(parts.join('/') || '');
}

function browseSelect() {
  document.getElementById('simPath').value = _browseCurrentPath;
  bootstrap.Modal.getInstance(document.getElementById('browseModal')).hide();
}


// ═══════════════════════════════════════════════════════════════════════════
//  PLOT3D VIEWER
// ═══════════════════════════════════════════════════════════════════════════

let plot3dMeta = [];

async function fetchPlot3DList() {
  if (!SIM_PATH) return;
  updateNavStatus();
  const sel = document.getElementById('p3dSelect');
  if (!sel) return;
  try {
    plot3dMeta = await apiGet('/api/plot3d?path=' + encodeURIComponent(SIM_PATH));
    sel.innerHTML = '<option value="">-- Select dataset --</option>';
    plot3dMeta.forEach((p, i) => {
      const qNames = p.quantities.map(q => q.name).join(', ');
      const opt = document.createElement('option');
      opt.value = i;
      opt.textContent = `PL3D ${i} — ${qNames} (${p.n_timesteps} steps)`;
      sel.appendChild(opt);
    });
  } catch (e) { console.error('Plot3D load error:', e); }
}

function onP3dSelected() {
  const idx = parseInt(document.getElementById('p3dSelect').value);
  if (isNaN(idx)) return;
  const p = plot3dMeta[idx];
  const info = document.getElementById('p3dInfoBox');
  if (info) {
    info.classList.remove('d-none');
    info.innerHTML = `<strong>Quantities:</strong> ${p.quantities.map(q=>q.name).join(', ')}<br><strong>Timesteps:</strong> ${p.n_timesteps} | Meshes: ${p.n_meshes}`;
  }
  // Populate quantity dropdown
  const qtySel = document.getElementById('p3dQty');
  if (qtySel) {
    qtySel.innerHTML = '';
    p.quantities.forEach((q, qi) => {
      const opt = document.createElement('option');
      opt.value = qi; opt.textContent = `${q.name} (${q.unit})`;
      qtySel.appendChild(opt);
    });
  }
  // Populate timestep dropdown
  const tSel = document.getElementById('p3dTime');
  if (tSel) {
    tSel.innerHTML = '';
    p.times.forEach((t, ti) => {
      const opt = document.createElement('option');
      opt.value = ti; opt.textContent = `t = ${t.toFixed(1)} s`;
      tSel.appendChild(opt);
    });
  }
}

async function renderPlot3D() {
  const idx = parseInt(document.getElementById('p3dSelect').value);
  if (isNaN(idx)) return alert('Select a Plot3D dataset first');
  showLoading('Rendering Plot3D cut-plane...');
  try {
    const posVal = document.getElementById('p3dPos').value;
    const data = await apiPost('/api/plot3d/render', {
      path: SIM_PATH,
      p3d_index: idx,
      time_idx: parseInt(document.getElementById('p3dTime').value) || 0,
      quantity_idx: parseInt(document.getElementById('p3dQty').value) || 0,
      axis: document.getElementById('p3dAxis').value,
      position: posVal !== '' ? parseFloat(posVal) : null,
      cmap: document.getElementById('p3dCmap').value,
      vmin: numOrNull('p3dVMin'),
      vmax: numOrNull('p3dVMax'),
      show_colorbar: true,
    });
    showPlot(data.image_b64, 'Plot3D Cut-Plane');
    hideLoading();
  } catch (e) { hideLoading(); alert(e.message); }
}


// ═══════════════════════════════════════════════════════════════════════════
//  SMOKE3D VIEWER
// ═══════════════════════════════════════════════════════════════════════════

let smoke3dMeta = [];

async function fetchSmoke3DList() {
  if (!SIM_PATH) return;
  updateNavStatus();
  const sel = document.getElementById('s3dSelect');
  if (!sel) return;
  try {
    smoke3dMeta = await apiGet('/api/smoke3d?path=' + encodeURIComponent(SIM_PATH));
    sel.innerHTML = '<option value="">-- Select dataset --</option>';
    smoke3dMeta.forEach((s, i) => {
      const opt = document.createElement('option');
      opt.value = i;
      opt.textContent = `S3D ${i} — ${s.quantity} (${s.n_timesteps} steps)`;
      sel.appendChild(opt);
    });
  } catch (e) { console.error('Smoke3D load error:', e); }
}

function onS3dSelected() {
  const idx = parseInt(document.getElementById('s3dSelect').value);
  if (isNaN(idx)) return;
  const s = smoke3dMeta[idx];
  const info = document.getElementById('s3dInfoBox');
  if (info) {
    info.classList.remove('d-none');
    info.innerHTML = `<strong>${s.quantity}</strong> (${s.unit})<br>Timesteps: ${s.n_timesteps} | Meshes: ${s.n_meshes}`;
  }
  document.getElementById('s3dTimeIdx').max = s.n_timesteps - 1;
}

async function renderSmoke3D() {
  const idx = parseInt(document.getElementById('s3dSelect').value);
  if (isNaN(idx)) return alert('Select a Smoke3D dataset first');
  showLoading('Rendering Smoke3D cut-plane...');
  try {
    const posVal = document.getElementById('s3dPos').value;
    const data = await apiPost('/api/smoke3d/render', {
      path: SIM_PATH,
      smoke_index: idx,
      time_idx: parseInt(document.getElementById('s3dTimeIdx').value) || 0,
      axis: document.getElementById('s3dAxis').value,
      position: posVal !== '' ? parseFloat(posVal) : null,
      cmap: document.getElementById('s3dCmap').value,
      vmin: numOrNull('s3dVMin'),
      vmax: numOrNull('s3dVMax'),
      show_colorbar: true,
    });
    showPlot(data.image_b64, 'Smoke3D Cut-Plane');
    hideLoading();
  } catch (e) { hideLoading(); alert(e.message); }
}


// ═══════════════════════════════════════════════════════════════════════════
//  PARTICLE VIEWER
// ═══════════════════════════════════════════════════════════════════════════

let particleMeta = [];

async function fetchParticleList() {
  if (!SIM_PATH) return;
  updateNavStatus();
  const sel = document.getElementById('partClass');
  if (!sel) return;
  try {
    particleMeta = await apiGet('/api/particles?path=' + encodeURIComponent(SIM_PATH));
    sel.innerHTML = '<option value="">-- Select class --</option>';
    particleMeta.forEach((p, i) => {
      const opt = document.createElement('option');
      opt.value = i;
      opt.textContent = `${p.class_name} (${p.n_timesteps} steps, ${p.quantities.length} quantities)`;
      sel.appendChild(opt);
    });
  } catch (e) { console.error('Particle load error:', e); }
}

function onPartClassSelected() {
  const idx = parseInt(document.getElementById('partClass').value);
  if (isNaN(idx)) return;
  const p = particleMeta[idx];
  const info = document.getElementById('partInfoBox');
  if (info) {
    info.classList.remove('d-none');
    info.innerHTML = `<strong>${p.class_name}</strong><br>Quantities: ${p.quantities.map(q=>q.name).join(', ') || 'none'}<br>Timesteps: ${p.n_timesteps}`;
  }
  // Populate color quantity selectors
  const colorSel = document.getElementById('partColorQty');
  const histSel = document.getElementById('partHistQty');
  [colorSel, histSel].forEach(sel => {
    if (!sel) return;
    sel.innerHTML = sel === colorSel ? '<option value="">None (uniform)</option>' : '';
    p.quantities.forEach(q => {
      const opt = document.createElement('option');
      opt.value = q.name; opt.textContent = `${q.name} (${q.unit})`;
      sel.appendChild(opt);
    });
  });
}

function onPartViewModeChange() {
  const mode = document.getElementById('partViewMode')?.value || 'scatter';
  document.getElementById('partPanelScatter')?.classList.toggle('d-none', mode !== 'scatter');
  document.getElementById('partPanelHistogram')?.classList.toggle('d-none', mode !== 'histogram');
}

async function renderParticleScatter() {
  const idx = parseInt(document.getElementById('partClass').value);
  if (isNaN(idx)) return alert('Select a particle class first');
  showLoading('Rendering particle scatter...');
  try {
    const data = await apiPost('/api/particle/scatter', {
      path: SIM_PATH,
      class_index: idx,
      time_idx: parseInt(document.getElementById('partTimeIdx').value) || 0,
      plane: document.getElementById('partPlane').value,
      color_quantity: document.getElementById('partColorQty').value || null,
      cmap: document.getElementById('partCmap').value,
      show_colorbar: true,
    });
    showPlot(data.image_b64, `Particles: ${data.n_particles || '?'} points`);
    hideLoading();
  } catch (e) { hideLoading(); alert(e.message); }
}

async function renderParticleHistogram() {
  const idx = parseInt(document.getElementById('partClass').value);
  if (isNaN(idx)) return alert('Select a particle class first');
  const qty = document.getElementById('partHistQty').value;
  if (!qty) return alert('Select a quantity');
  showLoading('Rendering histogram...');
  try {
    const data = await apiPost('/api/particle/histogram', {
      path: SIM_PATH,
      class_index: idx,
      quantity: qty,
      time_idx: parseInt(document.getElementById('partHistTimeIdx').value) || 0,
      bins: parseInt(document.getElementById('partHistBins').value) || 50,
    });
    showPlot(data.image_b64, 'Particle Histogram');
    hideLoading();
  } catch (e) { hideLoading(); alert(e.message); }
}


// ═══════════════════════════════════════════════════════════════════════════
//  ISOSURFACE VIEWER
// ═══════════════════════════════════════════════════════════════════════════

let isoMeta = [];

async function fetchIsosurfaceList() {
  if (!SIM_PATH) return;
  updateNavStatus();
  const sel = document.getElementById('isoSelect');
  if (!sel) return;
  try {
    isoMeta = await apiGet('/api/isosurfaces?path=' + encodeURIComponent(SIM_PATH));
    sel.innerHTML = '<option value="">-- Select isosurface --</option>';
    isoMeta.forEach((iso, i) => {
      const opt = document.createElement('option');
      opt.value = i;
      opt.textContent = `Iso ${i} — ${iso.quantity} (${iso.n_timesteps} steps)`;
      sel.appendChild(opt);
    });
  } catch (e) { console.error('Isosurface load error:', e); }
}

function onIsoSelected() {
  const idx = parseInt(document.getElementById('isoSelect').value);
  if (isNaN(idx)) return;
  const iso = isoMeta[idx];
  const info = document.getElementById('isoInfoBox');
  if (info) {
    info.classList.remove('d-none');
    info.innerHTML = `<strong>${iso.quantity}</strong> (${iso.unit})<br>Timesteps: ${iso.n_timesteps}<br>Color data: ${iso.has_color_data ? 'Yes' : 'No'}`;
  }
  document.getElementById('isoTimeIdx').max = iso.n_timesteps - 1;
}

async function renderIsosurface() {
  const idx = parseInt(document.getElementById('isoSelect').value);
  if (isNaN(idx)) return alert('Select an isosurface first');
  showLoading('Rendering isosurface projection...');
  try {
    const data = await apiPost('/api/isosurface/render', {
      path: SIM_PATH,
      iso_index: idx,
      time_idx: parseInt(document.getElementById('isoTimeIdx').value) || 0,
      plane: document.getElementById('isoPlane').value,
      cmap: document.getElementById('isoCmap').value,
      show_colorbar: true,
    });
    showPlot(data.image_b64, `Isosurface: ${data.n_vertices || '?'} vertices`);
    hideLoading();
  } catch (e) { hideLoading(); alert(e.message); }
}


// ═══════════════════════════════════════════════════════════════════════════
//  EVACUATION VIEWER
// ═══════════════════════════════════════════════════════════════════════════

let evacMeta = {};

async function fetchEvacMeta() {
  if (!SIM_PATH) return;
  updateNavStatus();
  try {
    evacMeta = await apiGet('/api/evacuation?path=' + encodeURIComponent(SIM_PATH));
    const info = document.getElementById('evacInfoBox');
    if (info) {
      if (evacMeta.classes && evacMeta.classes.length > 0) {
        info.classList.remove('d-none');
        info.innerHTML = `<strong>Classes:</strong> ${evacMeta.classes.length}<br><strong>Timesteps:</strong> ${evacMeta.n_timesteps}`;
        // Populate class filter
        const classSel = document.getElementById('evacClassFilter');
        if (classSel) {
          classSel.innerHTML = '<option value="">All Classes</option>';
          evacMeta.classes.forEach((c, i) => {
            const opt = document.createElement('option');
            opt.value = i; opt.textContent = c.class_name;
            classSel.appendChild(opt);
          });
        }
      } else {
        info.classList.remove('d-none');
        info.innerHTML = '<span class="text-warning">No evacuation data found.</span>';
      }
    }
  } catch (e) { console.error('Evac load error:', e); }
}

function onEvacViewModeChange() {
  const mode = document.getElementById('evacViewMode')?.value || 'floorplan';
  document.getElementById('evacPanelFloorplan')?.classList.toggle('d-none', mode !== 'floorplan');
  document.getElementById('evacPanelTimeseries')?.classList.toggle('d-none', mode !== 'timeseries');
}

async function renderEvacFloorplan() {
  showLoading('Rendering evacuation floor plan...');
  try {
    const classVal = document.getElementById('evacClassFilter').value;
    const data = await apiPost('/api/evacuation/floorplan', {
      path: SIM_PATH,
      time_idx: parseInt(document.getElementById('evacTimeIdx').value) || 0,
      class_index: classVal !== '' ? parseInt(classVal) : null,
    });
    showPlot(data.image_b64, `Evacuation: ${data.n_agents || 0} agents`);
    hideLoading();
  } catch (e) { hideLoading(); alert(e.message); }
}

async function renderEvacTimeseries() {
  showLoading('Rendering evacuation time-series...');
  try {
    const data = await apiPost('/api/evacuation/timeseries', {
      path: SIM_PATH,
      metric: document.getElementById('evacMetric').value,
    });
    showPlot(data.image_b64, 'Evacuation Time-Series');
    hideLoading();
  } catch (e) { hideLoading(); alert(e.message); }
}


// ═══════════════════════════════════════════════════════════════════════════
//  PERFORMANCE VIEWER (CPU + STEPS)
// ═══════════════════════════════════════════════════════════════════════════

let perfCpuMeta = {};
let perfStepsMeta = {};

async function fetchPerfMeta() {
  if (!SIM_PATH) return;
  updateNavStatus();
  try {
    perfCpuMeta = await apiGet('/api/cpu?path=' + encodeURIComponent(SIM_PATH));
    perfStepsMeta = await apiGet('/api/steps?path=' + encodeURIComponent(SIM_PATH));
    onPerfSourceChange();
  } catch (e) { console.error('Performance load error:', e); }
}

function onPerfSourceChange() {
  const source = document.getElementById('perfSource')?.value || 'cpu';
  const meta = source === 'cpu' ? perfCpuMeta : perfStepsMeta;
  const info = document.getElementById('perfInfoBox');
  const list = document.getElementById('perfCheckboxList');
  if (!list) return;

  if (!meta.columns || meta.columns.length === 0) {
    list.innerHTML = `<small class="text-secondary">No ${source} data available.</small>`;
    if (info) { info.classList.remove('d-none'); info.innerHTML = `<span class="text-warning">No ${source} data found.</span>`; }
    return;
  }

  if (info) {
    info.classList.remove('d-none');
    info.innerHTML = `<strong>Columns:</strong> ${meta.columns.length}<br><strong>Time:</strong> ${(meta.t_start||0).toFixed(1)} – ${(meta.t_end||0).toFixed(1)} s`;
  }

  list.innerHTML = '';
  meta.columns.forEach((col, i) => {
    if (col === 'Time') return;
    const div = document.createElement('div');
    div.className = 'form-check';
    const checked = i < 4 ? 'checked' : '';
    div.innerHTML = `<input class="form-check-input" type="checkbox" id="perf_${i}" value="${col}" ${checked}><label class="form-check-label small" for="perf_${i}">${col}</label>`;
    list.appendChild(div);
  });
}

function _getCheckedPerfCols() {
  const cols = [];
  document.querySelectorAll('[id^="perf_"]').forEach(cb => {
    if (cb.checked && cb.type === 'checkbox') cols.push(cb.value);
  });
  return cols;
}

function perfSelectAll() {
  document.querySelectorAll('[id^="perf_"]').forEach(cb => { if (cb.type === 'checkbox') cb.checked = true; });
}
function perfSelectNone() {
  document.querySelectorAll('[id^="perf_"]').forEach(cb => { if (cb.type === 'checkbox') cb.checked = false; });
}

async function renderPerformance() {
  const cols = _getCheckedPerfCols();
  if (cols.length === 0) return alert('Select at least one column');
  const source = document.getElementById('perfSource').value;
  const apiUrl = source === 'cpu' ? '/api/cpu/render' : '/api/steps/render';

  showLoading(`Plotting ${source} data...`);
  try {
    const tStart = numOrNull('perfTStart');
    const tEnd = numOrNull('perfTEnd');
    const timeRange = (tStart !== null || tEnd !== null) ? [tStart || 0, tEnd || 99999] : null;
    const data = await apiPost(apiUrl, {
      path: SIM_PATH,
      columns: cols,
      time_range: timeRange,
      show_grid: document.getElementById('perfShowGrid')?.checked !== false,
    });
    showPlot(data.image_b64, `${source.toUpperCase()}: ${cols.length} columns`);
    hideLoading();
  } catch (e) { hideLoading(); alert(e.message); }
}


// ═══════════════════════════════════════════════════════════════════════════
//  DOWNLOAD HELPER
// ═══════════════════════════════════════════════════════════════════════════

function _downloadBlob(blob, filename) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(a.href);
}

function downloadCurrentImage() {
  const img = document.getElementById('plotImage');
  if (!img || !img.src || img.classList.contains('d-none')) return alert('No plot to download');
  const badge = document.getElementById('displayBadge');
  const title = badge ? badge.textContent : 'plot';
  const fname = title.replace(/[^a-zA-Z0-9_\-\.]/g, '_').replace(/_+/g, '_') + '.png';
  const a = document.createElement('a');
  a.href = img.src;
  a.download = fname;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}


// ═══════════════════════════════════════════════════════════════════════════
//  ON PAGE LOAD
// ═══════════════════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
  updateNavStatus();
  // Pre-fill sim path from localStorage
  const pathEl = document.getElementById('simPath');
  if (pathEl && SIM_PATH) pathEl.value = SIM_PATH;
});
