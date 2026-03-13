/* ═══════════════════════════════════════════════════════════
   FDS Viewer — Multi-Theme Styles (Bootstrap 5)
   Themes: dark (default), light, skyblue
   ═══════════════════════════════════════════════════════════ */

/* ── Dark Theme (default) ──────────────────────────────────── */
[data-theme="dark"], :root {
  --bg-primary: #0f0f1a;
  --bg-card: #1a1a2e;
  --bg-surface: #16213e;
  --border-color: #334466;
  --text-primary: #e0e0f0;
  --text-secondary: #8899aa;
  --accent-red: #e94560;
  --accent-blue: #53d8fb;
  --accent-orange: #f5a623;
  --navbar-bg: #1a1a2e;
  --navbar-text: #e0e0f0;
  --modal-bg: #1a1a2e;
  --loading-overlay: rgba(15, 15, 26, 0.85);
  --scrollbar-track: #0f0f1a;
  --scrollbar-thumb: #334466;
  --scrollbar-hover: #556688;
  --input-bg: #16213e;
  --card-header-bg: rgba(22, 33, 62, 0.7);
  --btn-close-filter: invert(1);
  --dropdown-bg: #1a1a2e;
  --link-color: #53d8fb;
}

/* ── Light Theme ───────────────────────────────────────────── */
[data-theme="light"] {
  --bg-primary: #f0f2f5;
  --bg-card: #ffffff;
  --bg-surface: #e9ecef;
  --border-color: #ced4da;
  --text-primary: #212529;
  --text-secondary: #6c757d;
  --accent-red: #dc3545;
  --accent-blue: #0d6efd;
  --accent-orange: #fd7e14;
  --navbar-bg: #ffffff;
  --navbar-text: #212529;
  --modal-bg: #ffffff;
  --loading-overlay: rgba(240, 242, 245, 0.9);
  --scrollbar-track: #f0f2f5;
  --scrollbar-thumb: #ced4da;
  --scrollbar-hover: #adb5bd;
  --input-bg: #ffffff;
  --card-header-bg: rgba(233, 236, 239, 0.8);
  --btn-close-filter: none;
  --dropdown-bg: #ffffff;
  --link-color: #0d6efd;
}

/* ── Sky Blue Theme ────────────────────────────────────────── */
[data-theme="skyblue"] {
  --bg-primary: #e3f2fd;
  --bg-card: #bbdefb;
  --bg-surface: #90caf9;
  --border-color: #64b5f6;
  --text-primary: #0d2137;
  --text-secondary: #37474f;
  --accent-red: #e53935;
  --accent-blue: #1565c0;
  --accent-orange: #ff8f00;
  --navbar-bg: #1565c0;
  --navbar-text: #ffffff;
  --modal-bg: #bbdefb;
  --loading-overlay: rgba(227, 242, 253, 0.9);
  --scrollbar-track: #e3f2fd;
  --scrollbar-thumb: #64b5f6;
  --scrollbar-hover: #42a5f5;
  --input-bg: #e3f2fd;
  --card-header-bg: rgba(144, 202, 249, 0.5);
  --btn-close-filter: none;
  --dropdown-bg: #bbdefb;
  --link-color: #1565c0;
}

/* ── Base ──────────────────────────────────────────────────── */
body {
  background-color: var(--bg-primary);
  color: var(--text-primary);
  font-family: 'Inter', 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

code {
  color: var(--accent-blue);
  background: rgba(83, 216, 251, 0.08);
  padding: 0.15em 0.4em;
  border-radius: 4px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.85em;
}

/* ── Navbar ────────────────────────────────────────────────── */
.navbar {
  background: var(--navbar-bg) !important;
  border-bottom: 1px solid var(--border-color) !important;
  font-family: 'Inter', sans-serif;
}
.navbar-brand {
  font-size: 1.2rem;
  letter-spacing: 0.02em;
  color: var(--navbar-text) !important;
}
.navbar .nav-link {
  color: var(--navbar-text) !important;
  opacity: 0.8;
}
.navbar .nav-link:hover,
.navbar .nav-link.active {
  opacity: 1;
}

/* ── Theme Switcher ────────────────────────────────────────── */
.theme-switcher {
  display: flex;
  align-items: center;
  gap: 4px;
  margin-right: 12px;
}
.theme-switcher .theme-btn {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  border: 2px solid transparent;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.75rem;
  transition: all 0.2s ease;
  padding: 0;
}
.theme-switcher .theme-btn:hover {
  transform: scale(1.15);
}
.theme-switcher .theme-btn.active {
  border-color: var(--accent-red);
  box-shadow: 0 0 6px var(--accent-red);
}
.theme-btn-dark {
  background: #1a1a2e;
  color: #e0e0f0;
}
.theme-btn-light {
  background: #f0f2f5;
  color: #212529;
}
.theme-btn-skyblue {
  background: #64b5f6;
  color: #0d2137;
}

/* ── Main layout ───────────────────────────────────────────── */
.main-wrapper {
  padding-top: 70px;
  min-height: 100vh;
}

/* ── Cards ─────────────────────────────────────────────────── */
.card {
  background: var(--bg-card) !important;
  border-color: var(--border-color) !important;
}
.card-header {
  background: var(--card-header-bg) !important;
  border-color: var(--border-color) !important;
}

/* ── Form controls ─────────────────────────────────────────── */
.form-control,
.form-select {
  background-color: var(--input-bg) !important;
  border-color: var(--border-color) !important;
  color: var(--text-primary) !important;
  font-size: 0.85rem;
}
.form-control:focus,
.form-select:focus {
  border-color: var(--accent-blue) !important;
  box-shadow: 0 0 0 0.15rem rgba(83, 216, 251, 0.2) !important;
}
.form-control::placeholder {
  color: var(--text-secondary) !important;
  opacity: 0.7;
}
.input-group-text {
  background-color: var(--input-bg) !important;
  border-color: var(--border-color) !important;
  color: var(--text-secondary) !important;
}

/* ── Alerts / Info boxes ───────────────────────────────────── */
.alert-dark {
  background: var(--bg-surface) !important;
  color: var(--text-primary) !important;
  border-color: var(--border-color) !important;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75rem;
  line-height: 1.5;
}

/* ── Buttons ───────────────────────────────────────────────── */
.btn-danger {
  background-color: var(--accent-red) !important;
  border-color: var(--accent-red) !important;
}
.btn-danger:hover {
  background-color: #d63350 !important;
}

/* ── Sidebar sticky ────────────────────────────────────────── */
.sticky-sidebar {
  position: sticky;
  top: 75px;
}

/* ── Loading overlay ───────────────────────────────────────── */
.loading-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: var(--loading-overlay);
  z-index: 9999;
  display: flex;
  align-items: center;
  justify-content: center;
  backdrop-filter: blur(4px);
}

/* ── Plot image ────────────────────────────────────────────── */
#plotImage {
  max-width: 100%;
  border-radius: 6px;
  border: 1px solid var(--border-color);
}

/* ── Placeholder message ───────────────────────────────────── */
.placeholder-message {
  padding: 80px 20px;
}

/* ── Timestep multi-select ─────────────────────────────────── */
select[multiple] {
  scrollbar-width: thin;
  scrollbar-color: var(--border-color) var(--bg-surface);
}
select[multiple] option {
  padding: 2px 6px;
  font-size: 0.8rem;
}
select[multiple] option:checked {
  background: var(--accent-blue);
  color: #000;
}

/* ── Range slider ──────────────────────────────────────────── */
.form-range::-webkit-slider-thumb {
  background: var(--accent-red);
}
.form-range::-moz-range-thumb {
  background: var(--accent-red);
}
.form-range::-webkit-slider-runnable-track {
  background: var(--border-color);
}

/* ── Scrollbar ─────────────────────────────────────────────── */
::-webkit-scrollbar {
  width: 6px;
  height: 6px;
}
::-webkit-scrollbar-track {
  background: var(--scrollbar-track);
}
::-webkit-scrollbar-thumb {
  background: var(--scrollbar-thumb);
  border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
  background: var(--scrollbar-hover);
}

/* ── Feature cards (index page) ────────────────────────────── */
.feature-card-grid .card:hover {
  border-color: var(--accent-blue) !important;
  transform: translateY(-2px);
  transition: all 0.2s ease;
}

/* ── Badge colours ─────────────────────────────────────────── */
.badge.bg-success {
  background-color: #198754 !important;
}

/* ── Responsive fixes ──────────────────────────────────────── */
@media (max-width: 991px) {
  .sticky-sidebar {
    position: static;
  }
  .main-wrapper {
    padding-top: 60px;
  }
}

/* ── Theme-aware dropdown menus ────────────────────────────── */
.dropdown-menu {
  background-color: var(--dropdown-bg) !important;
  border-color: var(--border-color) !important;
}
.dropdown-item {
  color: var(--text-primary) !important;
}
.dropdown-item:hover,
.dropdown-item:focus {
  background-color: var(--bg-surface) !important;
  color: var(--text-primary) !important;
}

/* ── Theme-aware modal ─────────────────────────────────────── */
.modal-content {
  background-color: var(--modal-bg) !important;
  border-color: var(--border-color) !important;
  color: var(--text-primary) !important;
}
.modal-header {
  border-color: var(--border-color) !important;
}
.modal-footer {
  border-color: var(--border-color) !important;
}
.btn-close {
  filter: var(--btn-close-filter);
}

/* ── Theme-aware links ─────────────────────────────────────── */
a.text-info {
  color: var(--link-color) !important;
}

/* ── Loading overlay text ──────────────────────────────────── */
#loadingText {
  color: var(--text-primary) !important;
}

/* ── Theme-aware list group ────────────────────────────────── */
.list-group-item {
  background-color: var(--bg-card);
  color: var(--text-primary);
  border-color: var(--border-color);
}

/* ── Light/skyblue theme: override hardcoded bg-dark in HTML ─ */
[data-theme="light"] .bg-dark,
[data-theme="skyblue"] .bg-dark {
  background-color: var(--bg-card) !important;
  color: var(--text-primary) !important;
}
[data-theme="light"] .text-light,
[data-theme="skyblue"] .text-light {
  color: var(--text-primary) !important;
}
[data-theme="light"] .border-secondary,
[data-theme="skyblue"] .border-secondary {
  border-color: var(--border-color) !important;
}
[data-theme="light"] .text-secondary,
[data-theme="skyblue"] .text-secondary {
  color: var(--text-secondary) !important;
}
[data-theme="light"] .btn-close-white,
[data-theme="skyblue"] .btn-close-white {
  filter: none;
}

/* ── Smooth theme transition ───────────────────────────────── */
body,
.navbar,
.card,
.card-header,
.form-control,
.form-select,
.input-group-text,
.modal-content,
.dropdown-menu {
  transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
}

