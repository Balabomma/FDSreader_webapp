/* ═══════════════════════════════════════════════════════════
   FDS Viewer — Frontend Application Logic
   ═══════════════════════════════════════════════════════════ */

let SIM_PATH = '';
let SIM_DATA = null;

// Slice data mode state
let sliceMode = 'global';  // 'global' or 'mesh'
let selectedMeshIndex = null;

// Animation state
let animFrames = [];
let animIdx = 0;
let animTimer = null;
let animPlaying = false;
let animType = ''; // 'slice' or 'boundary'

// ─── HELPERS ────────────────────────────────────────────────

function showLoader(msg) {
  document.getElementById('loaderText').textContent = msg || 'Loading...';
  document.getElementById('loaderOverlay').classList.add('active');
}
function hideLoader() {
  document.getElementById('loaderOverlay').classList.remove('active');
}

async function apiPost(url, body) {
  const resp = await fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  });
  const data = await resp.json();
  if (!resp.ok || data.error) throw new Error(data.error || 'API error');
  return data;
}

function showPlot(title, imgB64) {
  document.getElementById('welcomeScreen').style.display = 'none';
  document.getElementById('plotArea').style.display = 'block';
  document.getElementById('plotTitle').textContent = title;
  document.getElementById('plotImage').src = 'data:image/png;base64,' + imgB64;
  document.getElementById('animControls').style.display = 'none';
}

function numOrNull(id) {
  const v = document.getElementById(id).value;
  return v === '' ? null : parseFloat(v);
}

// ─── LOAD SIMULATION ────────────────────────────────────────

async function loadSimulation() {
  const path = document.getElementById('simPath').value.trim();
  if (!path) return alert('Enter a simulation directory path');

  showLoader('Loading simulation data...');
  try {
    const data = await apiPost('/api/load', {path});
    SIM_PATH = path;
    SIM_DATA = data;

    // Update status
    const info = data.info;
    document.getElementById('simInfo').innerHTML =
      `<span class="status-dot online"></span>` +
      `<span>${info.chid} — ${info.meshes} mesh, ${info.slices} slices, ${info.devices} devcs</span>`;

    // Show nav
    document.getElementById('sbNav').style.display = 'flex';

    // Populate panels
    populateDevices(data.devices);
    populateHRR(data.hrr_columns);
    populateSlices(data.slices);
    populateBoundaries(data.boundaries);

    switchTab('devices');
    hideLoader();
  } catch(e) {
    hideLoader();
    alert('Error loading simulation: ' + e.message);
  }
}

// ─── TAB SWITCHING ──────────────────────────────────────────

function switchTab(tab) {
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  document.querySelector(`.nav-btn[data-tab="${tab}"]`).classList.add('active');
  document.querySelectorAll('.tab-panel').forEach(p => p.style.display = 'none');
  const panel = document.getElementById('panel-' + tab);
  if (panel) panel.style.display = 'block';
}

// ─── POPULATE: DEVICES ──────────────────────────────────────

function populateDevices(devices) {
  const el = document.getElementById('deviceList');
  el.innerHTML = '';
  devices.forEach(d => {
    if (d.id === 'Time') return;
    const lbl = document.createElement('label');
    lbl.innerHTML = `<input type="checkbox" value="${d.id}" checked> ${d.id} <span style="color:#888;font-size:0.7rem">(${d.quantity})</span>`;
    el.appendChild(lbl);
  });
}

async function plotDevices() {
  const checks = document.querySelectorAll('#deviceList input:checked');
  const ids = Array.from(checks).map(c => c.value);
  if (!ids.length) return alert('Select at least one device');

  const tMin = numOrNull('devTMin'), tMax = numOrNull('devTMax');
  const tRange = (tMin !== null && tMax !== null) ? [tMin, tMax] : null;

  showLoader('Plotting devices...');
  try {
    const data = await apiPost('/api/plot/devices', {path: SIM_PATH, device_ids: ids, time_range: tRange});
    showPlot('Device Data (DEVC)', data.image);
    hideLoader();
  } catch(e) { hideLoader(); alert(e.message); }
}

// ─── POPULATE: HRR ──────────────────────────────────────────

function populateHRR(cols) {
  const el = document.getElementById('hrrList');
  el.innerHTML = '';
  cols.forEach(c => {
    if (c === 'Time') return;
    const lbl = document.createElement('label');
    const checked = ['HRR','Q_RADI','Q_CONV','Q_COND'].includes(c) ? 'checked' : '';
    lbl.innerHTML = `<input type="checkbox" value="${c}" ${checked}> ${c}`;
    el.appendChild(lbl);
  });
}

async function plotHRR() {
  const checks = document.querySelectorAll('#hrrList input:checked');
  const cols = Array.from(checks).map(c => c.value);
  if (!cols.length) return alert('Select at least one HRR column');

  const tMin = numOrNull('hrrTMin'), tMax = numOrNull('hrrTMax');
  const tRange = (tMin !== null && tMax !== null) ? [tMin, tMax] : null;

  showLoader('Plotting HRR...');
  try {
    const data = await apiPost('/api/plot/hrr', {path: SIM_PATH, columns: cols, time_range: tRange});
    showPlot('Heat Release Rate (HRR)', data.image);
    hideLoader();
  } catch(e) { hideLoader(); alert(e.message); }
}

// ─── POPULATE: SLICES ───────────────────────────────────────

function populateSlices(slices) {
  const sel = document.getElementById('sliceSelect');
  sel.innerHTML = '<option value="">— Select a slice —</option>';
  slices.forEach(s => {
    const opt = document.createElement('option');
    opt.value = s.index;
    const idLabel = s.id ? ` ID="${s.id}"` : '';
    opt.textContent = `[${s.index}]${idLabel} ${s.quantity} (${s.type}, orient=${s.orientation}, ${s.n_meshes} mesh)`;
    opt.dataset.quantity = s.quantity;
    sel.appendChild(opt);
  });

  // Populate quantity filter
  const qtyFilter = document.getElementById('sliceQtyFilter');
  qtyFilter.innerHTML = '<option value="">— All Quantities —</option>';
  if (SIM_DATA.slice_quantities) {
    SIM_DATA.slice_quantities.forEach(q => {
      const opt = document.createElement('option');
      opt.value = q; opt.textContent = q;
      qtyFilter.appendChild(opt);
    });
  }
}

function filterSlicesByQuantity() {
  const qty = document.getElementById('sliceQtyFilter').value;
  const sel = document.getElementById('sliceSelect');
  const options = sel.querySelectorAll('option');
  options.forEach(opt => {
    if (opt.value === '') { opt.style.display = ''; return; }
    if (!qty || opt.dataset.quantity === qty) {
      opt.style.display = '';
    } else {
      opt.style.display = 'none';
    }
  });
  sel.value = '';
  onSliceSelected();
}

function onSliceSelected() {
  const idx = document.getElementById('sliceSelect').value;
  if (idx === '') {
    document.getElementById('sliceControls').style.display = 'none';
    return;
  }
  const s = SIM_DATA.slices[parseInt(idx)];
  document.getElementById('sliceControls').style.display = 'block';

  // Info box
  const idLine = s.id ? `ID: <strong>${s.id}</strong> | ` : '';
  document.getElementById('sliceInfoBox').innerHTML =
    `${idLine}Quantity: <strong>${s.quantity}</strong> (${s.unit})<br>` +
    `Type: ${s.type} | Orient: ${s.orientation} | Cell-centered: ${s.cell_centered}<br>` +
    `Extent: [${s.extent.map(v=>v.toFixed(2)).join(', ')}]<br>` +
    `Dirs: ${s.extent_dirs.join(', ')} | Meshes: ${s.n_meshes}<br>` +
    `Times: ${s.n_times} steps (${s.time_min.toFixed(1)}–${s.time_max.toFixed(1)} s)`;

  // Slider range
  const slider = document.getElementById('sliceTimeSlider');
  slider.min = Math.floor(s.time_min);
  slider.max = Math.ceil(s.time_max);
  slider.value = Math.floor(s.time_min);
  document.getElementById('sliceTimeVal').textContent = slider.value;

  // Populate mesh selector
  const meshSel = document.getElementById('meshSelect');
  meshSel.innerHTML = '';
  if (s.meshes && s.meshes.length > 0) {
    s.meshes.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.mesh_index;
      const shp = m.shape.length ? ` (${m.shape.join('×')})` : '';
      opt.textContent = `Mesh ${m.mesh_index} — ${m.mesh_id}${shp}`;
      meshSel.appendChild(opt);
    });
  }

  // Reset to global mode
  setSliceMode('global');

  // Profile directions
  const dirSel = document.getElementById('profileDir');
  dirSel.innerHTML = '';
  s.extent_dirs.forEach(d => {
    const opt = document.createElement('option');
    opt.value = d; opt.textContent = d;
    dirSel.appendChild(opt);
  });

  // Coord placeholders
  if (s.extent_dirs.length >= 2) {
    document.getElementById('ptCoord0').placeholder = s.extent_dirs[0];
    document.getElementById('ptCoord1').placeholder = s.extent_dirs[1];
  }
}

function setSliceMode(mode) {
  sliceMode = mode;
  document.getElementById('btnGlobal').classList.toggle('active', mode === 'global');
  document.getElementById('btnMesh').classList.toggle('active', mode === 'mesh');
  document.getElementById('meshSelectGroup').style.display = mode === 'mesh' ? 'block' : 'none';
  if (mode === 'global') {
    selectedMeshIndex = null;
  } else {
    selectedMeshIndex = parseInt(document.getElementById('meshSelect').value) || 0;
  }
}

function onMeshChanged() {
  selectedMeshIndex = parseInt(document.getElementById('meshSelect').value) || 0;
}

function _slicePayload() {
  const idx = parseInt(document.getElementById('sliceSelect').value);
  const s = SIM_DATA.slices[idx];
  const payload = { path: SIM_PATH, slice_index: idx };
  if (s.id) payload.slice_id = s.id;
  payload.use_global = (sliceMode === 'global');
  if (sliceMode === 'mesh' && selectedMeshIndex !== null) {
    payload.mesh_index = selectedMeshIndex;
  }
  return payload;
}

async function plotSlice() {
  const idx = parseInt(document.getElementById('sliceSelect').value);
  if (isNaN(idx)) return alert('Select a slice');
  const t = parseFloat(document.getElementById('sliceTimeSlider').value);
  const s = SIM_DATA.slices[idx];
  const modeLabel = sliceMode === 'global' ? 'Global' : `Mesh ${selectedMeshIndex}`;

  showLoader('Rendering slice...');
  try {
    const payload = _slicePayload();
    payload.time = t;
    payload.colormap = document.getElementById('sliceCmap').value;
    payload.vmin = numOrNull('sliceVMin');
    payload.vmax = numOrNull('sliceVMax');
    const data = await apiPost('/api/plot/slice', payload);
    showPlot(`Slice [${idx}] ${s.quantity} [${modeLabel}] — t = ${data.actual_time.toFixed(1)} s`, data.image);
    hideLoader();
  } catch(e) { hideLoader(); alert(e.message); }
}

async function plotSliceMulti() {
  const idx = parseInt(document.getElementById('sliceSelect').value);
  if (isNaN(idx)) return alert('Select a slice');
  const s = SIM_DATA.slices[idx];
  const step = (s.time_max - s.time_min) / 5;
  const times = Array.from({length:6}, (_,i) => Math.round(s.time_min + i*step));

  showLoader('Generating multi-time view...');
  try {
    const payload = _slicePayload();
    payload.times = times;
    payload.colormap = document.getElementById('sliceCmap').value;
    payload.vmin = numOrNull('sliceVMin');
    payload.vmax = numOrNull('sliceVMax');
    const data = await apiPost('/api/plot/slice/multi', payload);
    showPlot(`Slice [${idx}] ${s.quantity} — Multi-Time Snapshots`, data.image);
    hideLoader();
  } catch(e) { hideLoader(); alert(e.message); }
}

// ─── SLICE ANIMATION ────────────────────────────────────────

async function animateSlice() {
  const idx = parseInt(document.getElementById('sliceSelect').value);
  if (isNaN(idx)) return alert('Select a slice');

  showLoader('Generating animation frames...');
  try {
    const payload = _slicePayload();
    payload.time_start = numOrNull('animTStart') || 0;
    payload.time_end = numOrNull('animTEnd');
    payload.n_frames = parseInt(document.getElementById('animFrames').value) || 20;
    payload.colormap = document.getElementById('sliceCmap').value;
    payload.vmin = numOrNull('sliceVMin');
    payload.vmax = numOrNull('sliceVMax');
    const data = await apiPost('/api/plot/slice/animate', payload);
    hideLoader();
    startAnimation(data.frames, 'slice', idx);
  } catch(e) { hideLoader(); alert(e.message); }
}

function startAnimation(frames, type, idx) {
  stopAnimation();
  animFrames = frames;
  animType = type;
  animIdx = 0;
  animPlaying = true;

  document.getElementById('welcomeScreen').style.display = 'none';
  document.getElementById('plotArea').style.display = 'block';
  document.getElementById('animControls').style.display = 'flex';
  document.getElementById('plotTitle').textContent = `${type === 'slice' ? 'Slice' : 'Boundary'} Animation`;

  const slider = document.getElementById('animSlider');
  slider.min = 0;
  slider.max = frames.length - 1;
  slider.value = 0;

  const ppBtn = type === 'slice' ? 'btnPlayPause' : 'btnBndfPlayPause';
  document.getElementById(ppBtn).style.display = 'inline-block';
  document.getElementById(ppBtn).textContent = '⏸ Pause';

  renderFrame(0);
  animTimer = setInterval(() => {
    if (!animPlaying) return;
    animIdx = (animIdx + 1) % animFrames.length;
    renderFrame(animIdx);
  }, 300);
}

function renderFrame(i) {
  const f = animFrames[i];
  document.getElementById('plotImage').src = 'data:image/png;base64,' + f.image;
  document.getElementById('animTimeLabel').textContent = `t = ${f.time.toFixed(1)} s`;
  document.getElementById('animSlider').value = i;
}

function seekAnimation(val) {
  animIdx = parseInt(val);
  renderFrame(animIdx);
}

function togglePlayback() {
  animPlaying = !animPlaying;
  document.getElementById('btnPlayPause').textContent = animPlaying ? '⏸ Pause' : '▶ Play';
}

function toggleBndfPlayback() {
  animPlaying = !animPlaying;
  document.getElementById('btnBndfPlayPause').textContent = animPlaying ? '⏸ Pause' : '▶ Play';
}

function stopAnimation() {
  if (animTimer) clearInterval(animTimer);
  animTimer = null;
  animPlaying = false;
  animFrames = [];
}

// ─── SLICE PROFILE / TIME-SERIES ────────────────────────────

async function plotProfile() {
  const idx = parseInt(document.getElementById('sliceSelect').value);
  if (isNaN(idx)) return alert('Select a slice');

  showLoader('Extracting profile...');
  try {
    const payload = _slicePayload();
    payload.direction = document.getElementById('profileDir').value;
    payload.position = parseFloat(document.getElementById('profilePos').value) || 0;
    payload.time = parseFloat(document.getElementById('profileTime').value) || 100;
    const data = await apiPost('/api/plot/slice/profile', payload);
    showPlot('Slice Line Profile', data.image);
    hideLoader();
  } catch(e) { hideLoader(); alert(e.message); }
}

async function plotSliceTS() {
  const idx = parseInt(document.getElementById('sliceSelect').value);
  if (isNaN(idx)) return alert('Select a slice');
  const s = SIM_DATA.slices[idx];
  const point = {};
  const v0 = parseFloat(document.getElementById('ptCoord0').value);
  const v1 = parseFloat(document.getElementById('ptCoord1').value);
  if (!isNaN(v0) && s.extent_dirs[0]) point[s.extent_dirs[0]] = v0;
  if (!isNaN(v1) && s.extent_dirs[1]) point[s.extent_dirs[1]] = v1;

  showLoader('Extracting time-series...');
  try {
    const payload = _slicePayload();
    payload.point = point;
    const data = await apiPost('/api/plot/slice/timeseries', payload);
    showPlot('Slice Point Time-Series', data.image);
    hideLoader();
  } catch(e) { hideLoader(); alert(e.message); }
}

// ─── POPULATE: BOUNDARIES ───────────────────────────────────

function populateBoundaries(bounds) {
  const sel = document.getElementById('bndfSelect');
  sel.innerHTML = '<option value="">— Select obstruction —</option>';
  bounds.forEach((b, arrIdx) => {
    // Server already filters to has_boundary_data=True, but guard just in case
    if (b.has_boundary_data === false) return;
    const opt = document.createElement('option');
    opt.value = b.id;                     // Use obstruction ID as value
    opt.dataset.arrIndex = arrIdx;        // Store array position for quick lookup
    opt.textContent = `ID="${b.id}" — ${b.quantities.join(', ')} (orient: ${b.orientations.join(',')})`;
    sel.appendChild(opt);
  });
}

function onBndfSelected() {
  const oid = document.getElementById('bndfSelect').value;
  if (oid === '') {
    document.getElementById('bndfControls').style.display = 'none';
    return;
  }
  const b = SIM_DATA.boundaries.find(x => String(x.id) === oid);
  if (!b) { document.getElementById('bndfControls').style.display = 'none'; return; }
  document.getElementById('bndfControls').style.display = 'block';

  // Info
  document.getElementById('bndfInfoBox').innerHTML =
    `ID: <strong>${b.id}</strong> | Index: ${b.index}<br>` +
    `Box: [${b.bounding_box.map(v=>v.toFixed(2)).join(', ')}]<br>` +
    `Quantities: ${b.quantities.join(', ')}<br>` +
    `Orientations: ${b.orientations.join(', ')}<br>` +
    `Meshes: ${b.n_meshes} | Times: ${b.n_times}`;

  // Qty select
  const qtySel = document.getElementById('bndfQty');
  qtySel.innerHTML = '';
  b.quantities.forEach(q => {
    const o = document.createElement('option'); o.value = q; o.textContent = q; qtySel.appendChild(o);
  });

  // Orient select
  const oSel = document.getElementById('bndfOrient');
  oSel.innerHTML = '';
  b.orientations.forEach(o => {
    const opt = document.createElement('option'); opt.value = o; opt.textContent = o; oSel.appendChild(opt);
  });

  // Slider
  if (b.time_max) {
    const sl = document.getElementById('bndfTimeSlider');
    sl.min = Math.floor(b.time_min);
    sl.max = Math.ceil(b.time_max);
    sl.value = Math.min(100, Math.ceil(b.time_max));
    document.getElementById('bndfTimeVal').textContent = sl.value;
  }
}

function _bndfPayload() {
  const oid = document.getElementById('bndfSelect').value;
  const b = SIM_DATA.boundaries.find(x => String(x.id) === oid);
  return {
    path: SIM_PATH,
    obstruction_id: oid,
    obstruction_index: b ? b.index : 0,
    quantity: document.getElementById('bndfQty').value,
    orientation: parseInt(document.getElementById('bndfOrient').value)
  };
}

async function plotBoundary() {
  const oid = document.getElementById('bndfSelect').value;
  if (!oid) return alert('Select an obstruction');

  showLoader('Rendering boundary...');
  try {
    const payload = _bndfPayload();
    payload.time = parseFloat(document.getElementById('bndfTimeSlider').value);
    payload.colormap = document.getElementById('bndfCmap').value;
    payload.vmin = numOrNull('bndfVMin');
    payload.vmax = numOrNull('bndfVMax');
    const data = await apiPost('/api/plot/boundary', payload);
    showPlot(`Boundary ID="${oid}" — t = ${data.actual_time.toFixed(1)} s`, data.image);
    hideLoader();
  } catch(e) { hideLoader(); alert(e.message); }
}

async function plotBndfTS() {
  const oid = document.getElementById('bndfSelect').value;
  if (!oid) return alert('Select an obstruction');

  showLoader('Extracting boundary time-series...');
  try {
    const data = await apiPost('/api/plot/boundary/timeseries', _bndfPayload());
    showPlot('Boundary Time-Series', data.image);
    hideLoader();
  } catch(e) { hideLoader(); alert(e.message); }
}

async function animateBoundary() {
  const oid = document.getElementById('bndfSelect').value;
  if (!oid) return alert('Select an obstruction');

  showLoader('Generating boundary animation...');
  try {
    const payload = _bndfPayload();
    payload.time_start = numOrNull('bndfAnimTStart') || 0;
    payload.time_end = numOrNull('bndfAnimTEnd');
    payload.n_frames = parseInt(document.getElementById('bndfAnimFrames').value) || 20;
    payload.colormap = document.getElementById('bndfCmap').value;
    payload.vmin = numOrNull('bndfVMin');
    payload.vmax = numOrNull('bndfVMax');
    const data = await apiPost('/api/plot/boundary/animate', payload);
    hideLoader();
    startAnimation(data.frames, 'boundary', oid);
  } catch(e) { hideLoader(); alert(e.message); }
}

// ─── DOWNLOADS ──────────────────────────────────────────────

function downloadCurrentPlot() {
  const img = document.getElementById('plotImage');
  if (!img.src || img.src === window.location.href) return alert('No plot to download');
  const title = document.getElementById('plotTitle').textContent || 'plot';
  const fname = title.replace(/[^a-zA-Z0-9_\-\.]/g, '_').replace(/_+/g, '_') + '.png';
  const a = document.createElement('a');
  a.href = img.src;
  a.download = fname;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

async function downloadSlicePNG() {
  const idx = parseInt(document.getElementById('sliceSelect').value);
  if (isNaN(idx)) return alert('Select a slice first');

  showLoader('Preparing high-res PNG...');
  try {
    const payload = _slicePayload();
    payload.time = parseFloat(document.getElementById('sliceTimeSlider').value);
    payload.colormap = document.getElementById('sliceCmap').value;
    payload.vmin = numOrNull('sliceVMin');
    payload.vmax = numOrNull('sliceVMax');
    const resp = await fetch('/api/download/slice', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    if (!resp.ok) { const err = await resp.json(); throw new Error(err.error); }
    const blob = await resp.blob();
    const fname = resp.headers.get('Content-Disposition')?.match(/filename="?(.+?)"?$/)?.[1] || 'slice.png';
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = fname;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(a.href);
    hideLoader();
  } catch(e) { hideLoader(); alert('Download error: ' + e.message); }
}

async function downloadSliceMultiZip() {
  const idx = parseInt(document.getElementById('sliceSelect').value);
  if (isNaN(idx)) return alert('Select a slice first');
  const s = SIM_DATA.slices[idx];

  // Parse custom times or generate even steps
  const raw = document.getElementById('dlSliceTimes').value.trim();
  let times;
  if (raw) {
    times = raw.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v));
    if (!times.length) return alert('Enter valid comma-separated time values');
  } else {
    const step = (s.time_max - s.time_min) / 5;
    times = Array.from({length:6}, (_,i) => Math.round(s.time_min + i*step));
  }

  showLoader(`Preparing ${times.length} snapshots (ZIP)...`);
  try {
    const payload = _slicePayload();
    payload.times = times;
    payload.colormap = document.getElementById('sliceCmap').value;
    payload.vmin = numOrNull('sliceVMin');
    payload.vmax = numOrNull('sliceVMax');
    const resp = await fetch('/api/download/slice/multi', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    if (!resp.ok) { const err = await resp.json(); throw new Error(err.error); }
    const blob = await resp.blob();
    const fname = resp.headers.get('Content-Disposition')?.match(/filename="?(.+?)"?$/)?.[1] || 'slices.zip';
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = fname;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(a.href);
    hideLoader();
  } catch(e) { hideLoader(); alert('Download error: ' + e.message); }
}

async function downloadSliceGIF() {
  const idx = parseInt(document.getElementById('sliceSelect').value);
  if (isNaN(idx)) return alert('Select a slice first');

  const nFrames = parseInt(document.getElementById('animFrames').value) || 20;
  const fps = parseInt(document.getElementById('animFPS').value) || 4;

  showLoader(`Generating GIF (${nFrames} frames)...`);
  try {
    const payload = _slicePayload();
    payload.time_start = numOrNull('animTStart') || 0;
    payload.time_end = numOrNull('animTEnd');
    payload.n_frames = nFrames;
    payload.fps = fps;
    payload.colormap = document.getElementById('sliceCmap').value;
    payload.vmin = numOrNull('sliceVMin');
    payload.vmax = numOrNull('sliceVMax');
    const resp = await fetch('/api/download/slice/gif', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    if (!resp.ok) { const err = await resp.json(); throw new Error(err.error); }
    const blob = await resp.blob();
    const fname = resp.headers.get('Content-Disposition')?.match(/filename="?(.+?)"?$/)?.[1] || 'slice_animation.gif';
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = fname;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(a.href);
    hideLoader();
  } catch(e) { hideLoader(); alert('Download error: ' + e.message); }
}

async function downloadBoundaryGIF() {
  const oid = document.getElementById('bndfSelect').value;
  if (!oid) return alert('Select an obstruction first');

  const nFrames = parseInt(document.getElementById('bndfAnimFrames').value) || 20;
  const fps = parseInt(document.getElementById('bndfAnimFPS').value) || 4;

  showLoader(`Generating boundary GIF (${nFrames} frames)...`);
  try {
    const payload = _bndfPayload();
    payload.time_start = numOrNull('bndfAnimTStart') || 0;
    payload.time_end = numOrNull('bndfAnimTEnd');
    payload.n_frames = nFrames;
    payload.fps = fps;
    payload.colormap = document.getElementById('bndfCmap').value;
    payload.vmin = numOrNull('bndfVMin');
    payload.vmax = numOrNull('bndfVMax');
    const resp = await fetch('/api/download/boundary/gif', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    if (!resp.ok) { const err = await resp.json(); throw new Error(err.error); }
    const blob = await resp.blob();
    const fname = resp.headers.get('Content-Disposition')?.match(/filename="?(.+?)"?$/)?.[1] || 'boundary_animation.gif';
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = fname;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(a.href);
    hideLoader();
  } catch(e) { hideLoader(); alert('Download error: ' + e.message); }
}

// ─── KEYBOARD SHORTCUTS ─────────────────────────────────────

document.addEventListener('keydown', e => {
  if (e.key === 'Enter' && document.activeElement.id === 'simPath') loadSimulation();
  if (animFrames.length && e.key === ' ') { e.preventDefault(); animType === 'slice' ? togglePlayback() : toggleBndfPlayback(); }
  if (animFrames.length && e.key === 'ArrowRight') { animIdx = Math.min(animIdx+1, animFrames.length-1); renderFrame(animIdx); }
  if (animFrames.length && e.key === 'ArrowLeft') { animIdx = Math.max(animIdx-1, 0); renderFrame(animIdx); }
});
