"""
FDS Viewer — fdsreader Helper Functions
========================================
All fdsreader data-access and matplotlib rendering logic lives here,
keeping app.py thin (routes only).
"""

from __future__ import annotations

import io
import base64
import traceback
from typing import Any, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fdsreader

# ── simulation cache ────────────────────────────────────────────────────────
_sim_cache: dict[str, fdsreader.Simulation] = {}


def load_simulation(path: str) -> fdsreader.Simulation:
    """Load (or return cached) fdsreader.Simulation.  Raises ValueError."""
    import os
    if not os.path.isdir(path):
        raise ValueError(f"Directory not found: {path}")
    if path not in _sim_cache:
        _sim_cache[path] = fdsreader.Simulation(path)
    return _sim_cache[path]


# ── dark theme ──────────────────────────────────────────────────────────────
def _apply_dark_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor": "#16213e",
        "axes.edgecolor": "#334466",
        "axes.labelcolor": "#d0d0e0",
        "text.color": "#d0d0e0",
        "xtick.color": "#8899aa",
        "ytick.color": "#8899aa",
        "grid.color": "#223355",
        "grid.alpha": 0.5,
        "legend.facecolor": "#16213e",
        "legend.edgecolor": "#445577",
        "legend.labelcolor": "#d0d0e0",
        "figure.figsize": (10, 7),
        "font.size": 11,
    })

def _apply_uniform_axes(ax, ext: list, xdir: str, ydir: str) -> None:
    """Force axes to real-world physical extents (metres) so all plots are uniform."""
    if ext and len(ext) >= 4:
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])

    # Equal aspect: 1 metre in x == 1 metre in y visually
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel(f'{xdir} (m)', fontsize=11)
    ax.set_ylabel(f'{ydir} (m)', fontsize=11)

    # Consistent tick formatting across all axes
    ax.tick_params(labelsize=9)
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))


def _resolve_extent_and_dirs(ext_full: list, xdir: str, ydir: str) -> list:
    """
    Given the full extent list and two axis direction labels (e.g. 'x', 'z'),
    extract the correct [min, max, min, max] for the 2D plot.

    fdsreader extent format: [x_min, x_max, y_min, y_max, z_min, z_max]
    Plane XY (normal Z): ext2d = [x_min, x_max, y_min, y_max]
    Plane XZ (normal Y): ext2d = [x_min, x_max, z_min, z_max]
    Plane YZ (normal X): ext2d = [y_min, y_max, z_min, z_max]
    """
    axis_map = {'x': (0, 1), 'y': (2, 3), 'z': (4, 5)}
    if ext_full and len(ext_full) >= 6:
        d1 = xdir.lower()
        d2 = ydir.lower()
        i0, i1 = axis_map.get(d1, (0, 1))
        i2, i3 = axis_map.get(d2, (4, 5))
        return [ext_full[i0], ext_full[i1], ext_full[i2], ext_full[i3]]
    # Fallback: return ext as-is if already 4-element
    return ext_full if ext_full and len(ext_full) == 4 else []


# Orientation → axis directions for boundary plots
_ORIENT_DIRS = {
    1: ('y', 'z'), -1: ('y', 'z'),
    2: ('x', 'z'), -2: ('x', 'z'),
    3: ('x', 'y'), -3: ('x', 'y'),
}


# ── figure → base64 ────────────────────────────────────────────────────────
def fig_to_base64(fig: plt.Figure, dpi: int = 120) -> str:
    """Convert a matplotlib Figure to a data-URI base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{b64}"


def _fig_to_raw_bytes(fig: plt.Figure, dpi: int = 150) -> io.BytesIO:
    """Render figure to PNG bytes (high-res for download)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    buf.seek(0)
    plt.close(fig)
    return buf


# ── helpers ─────────────────────────────────────────────────────────────────
def _safe(val: Any, default: Any = None) -> Any:
    return val if val is not None and val != "" else default


_ORIENT_MAP = {
    -3: "Z\u2212 (bottom)", 3: "Z+ (top)",
    -2: "Y\u2212 (front)",  2: "Y+ (back)",
    -1: "X\u2212 (left)",   1: "X+ (right)",
}


def _orient_label(o: int) -> str:
    return _ORIENT_MAP.get(o, str(o))


# Plot colour cycle for multi-line charts
COLORS = ['#e94560', '#53d8fb', '#f5a623', '#7ed321',
          '#bd10e0', '#50e3c2', '#ff6b6b', '#0f3460',
          '#ffd93d', '#6bcb77', '#4d96ff', '#ff6b81']


# ═════════════════════════════════════════════════════════════════════════════
#  DEVICE (DEVC) METADATA & RENDERING
# ═════════════════════════════════════════════════════════════════════════════

def get_device_metadata(sim: fdsreader.Simulation) -> list[dict]:
    """Return JSON-serialisable metadata for every device."""
    result: list[dict] = []
    devc = getattr(sim, "devices", None)
    if not devc:
        return result
    for dv in devc:
        did = getattr(dv, "id", "?")
        if did == "Time":
            continue
        result.append({
            "id": did,
            "quantity": getattr(dv, "quantity_name", str(getattr(dv, "quantity", ""))),
            "unit": getattr(dv, "unit", ""),
            "position": list(dv.position) if hasattr(dv, "position") else [],
            "n_points": len(dv.data) if hasattr(dv, "data") else 0,
        })
    return result


def render_devices(
    sim: fdsreader.Simulation,
    device_ids: list[str],
    time_range: Optional[list[float]] = None,
    show_grid: bool = True,
) -> dict:
    """Plot selected device time-series → { image_b64 }."""
    _apply_dark_style()
    devc = sim.devices
    time = devc["Time"].data
    mask = np.ones(len(time), dtype=bool)
    if time_range and len(time_range) == 2:
        mask = (time >= time_range[0]) & (time <= time_range[1])

    fig, ax = plt.subplots(figsize=(12, 6))
    n_plotted = 0
    unit_label = ""
    for i, did in enumerate(device_ids):
        if did == "Time":
            continue
        try:
            dv = devc[did]
            ax.plot(time[mask], dv.data[mask], label=dv.id,
                    color=COLORS[i % len(COLORS)], lw=1.5)
            if not unit_label:
                unit_label = f"{dv.quantity_name} ({dv.unit})"
            n_plotted += 1
        except Exception:
            continue
    if not n_plotted:
        plt.close(fig)
        raise ValueError("No valid devices found for the given IDs")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(unit_label)
    ax.legend(framealpha=0.7)
    if show_grid:
        ax.grid(True, alpha=0.3)
    ax.set_title("Device Data (DEVC)", fontsize=14, color="#e94560")
    return {"image_b64": fig_to_base64(fig)}


def render_device_comparison(
    sim: fdsreader.Simulation,
    device_ids_left: list[str],
    device_ids_right: list[str],
    time_range: Optional[list[float]] = None,
) -> dict:
    """Dual-axis plot: left-axis devices vs right-axis devices → { image_b64 }."""
    _apply_dark_style()
    devc = sim.devices
    time = devc["Time"].data
    mask = np.ones(len(time), dtype=bool)
    if time_range and len(time_range) == 2:
        mask = (time >= time_range[0]) & (time <= time_range[1])

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    unit_l = unit_r = ""
    for i, did in enumerate(device_ids_left):
        try:
            dv = devc[did]
            ax1.plot(time[mask], dv.data[mask], label=dv.id,
                     color=COLORS[i % len(COLORS)], lw=1.5)
            if not unit_l:
                unit_l = f"{dv.quantity_name} ({dv.unit})"
        except Exception:
            continue
    for i, did in enumerate(device_ids_right):
        try:
            dv = devc[did]
            ax2.plot(time[mask], dv.data[mask], label=dv.id,
                     color=COLORS[(i + len(device_ids_left)) % len(COLORS)],
                     lw=1.5, linestyle="--")
            if not unit_r:
                unit_r = f"{dv.quantity_name} ({dv.unit})"
        except Exception:
            continue

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel(unit_l, color=COLORS[0])
    ax2.set_ylabel(unit_r, color=COLORS[len(device_ids_left) % len(COLORS)])
    ax1.tick_params(axis="y", labelcolor=COLORS[0])
    ax2.tick_params(axis="y", labelcolor=COLORS[len(device_ids_left) % len(COLORS)])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, framealpha=0.7, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Device Comparison (dual-axis)", fontsize=14, color="#e94560")
    return {"image_b64": fig_to_base64(fig)}


# ═════════════════════════════════════════════════════════════════════════════
#  HRR METADATA & RENDERING
# ═════════════════════════════════════════════════════════════════════════════

def get_hrr_metadata(sim: fdsreader.Simulation) -> dict:
    """Return HRR column names + time range."""
    hrr = getattr(sim, "hrr", None)
    if not hrr:
        return {"columns": [], "t_start": 0, "t_end": 0, "n_points": 0}
    cols = list(hrr.keys())
    time = hrr.get("Time", np.array([]))
    return {
        "columns": cols,
        "t_start": float(time[0]) if len(time) > 0 else 0.0,
        "t_end": float(time[-1]) if len(time) > 0 else 0.0,
        "n_points": len(time),
    }


def render_hrr(
    sim: fdsreader.Simulation,
    columns: list[str],
    time_range: Optional[list[float]] = None,
    show_grid: bool = True,
) -> dict:
    """Plot selected HRR columns → { image_b64 }."""
    _apply_dark_style()
    hrr = sim.hrr
    time = hrr["Time"]
    mask = np.ones(len(time), dtype=bool)
    if time_range and len(time_range) == 2:
        mask = (time >= time_range[0]) & (time <= time_range[1])

    fig, ax = plt.subplots(figsize=(12, 6))
    n_plotted = 0
    for i, col in enumerate(columns):
        if col == "Time":
            continue
        if col in hrr:
            ax.plot(time[mask], hrr[col][mask], label=col,
                    color=COLORS[i % len(COLORS)], lw=1.5)
            n_plotted += 1
    if not n_plotted:
        plt.close(fig)
        raise ValueError("No valid HRR columns found")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("kW")
    ax.legend(framealpha=0.7)
    if show_grid:
        ax.grid(True, alpha=0.3)
    ax.set_title("Heat Release Rate (HRR)", fontsize=14, color="#e94560")
    return {"image_b64": fig_to_base64(fig)}


# ═════════════════════════════════════════════════════════════════════════════
#  SLICE METADATA
# ═════════════════════════════════════════════════════════════════════════════

def get_slice_metadata(sim: fdsreader.Simulation) -> list[dict]:
    """Return JSON-serialisable metadata for every slice."""
    result: list[dict] = []
    for idx, slc in enumerate(sim.slices or []):
        sid = getattr(slc, "id", "") or ""
        q_name = slc.quantity.name if hasattr(slc.quantity, "name") else str(slc.quantity)
        q_unit = slc.quantity.unit if hasattr(slc.quantity, "unit") else ""

        # per-mesh sub-slice info
        meshes_info: list[dict] = []
        for mi in range(len(slc)):
            try:
                ss = slc[mi]
                meshes_info.append({
                    "mesh_index": mi,
                    "shape": list(ss.data.shape[1:]) if hasattr(ss, "data") else [],
                    "extent": ss.extent.as_list() if hasattr(ss, "extent") and hasattr(ss.extent, "as_list") else [],
                })
            except Exception:
                meshes_info.append({"mesh_index": mi, "shape": [], "extent": []})

        times = slc.times if hasattr(slc, "times") else np.array([])
        result.append({
            "index": idx,
            "id": sid,
            "type": getattr(slc, "type", "?"),
            "quantity": q_name,
            "unit": q_unit,
            "orientation": getattr(slc, "orientation", 0),
            "cell_centered": getattr(slc, "cell_centered", False),
            "extent": slc.extent.as_list() if hasattr(slc.extent, "as_list") else [],
            "extent_dirs": list(getattr(slc, "extent_dirs", [])),
            "n_timesteps": len(times),
            "t_start": float(times[0]) if len(times) > 0 else 0.0,
            "t_end": float(times[-1]) if len(times) > 0 else 0.0,
            "times": [round(float(t), 3) for t in times],
            "n_meshes": len(slc),
            "meshes": meshes_info,
        })
    return result


# ═════════════════════════════════════════════════════════════════════════════
#  SLICE DATA ACCESS
# ═════════════════════════════════════════════════════════════════════════════

def _resolve_slice(sim: fdsreader.Simulation, slice_id: int | str) -> Any:
    """Resolve a slice by integer index or string ID."""
    if isinstance(slice_id, str) and not slice_id.isdigit():
        slc = sim.slices.get_by_id(slice_id)
        if slc is None:
            raise ValueError(f'Slice ID "{slice_id}" not found')
        return slc
    return sim.slices[int(slice_id)]


def _slice_frame(slc: Any, timestep_idx: int,
                 use_global: bool = True,
                 mesh_index: Optional[int] = None) -> tuple[np.ndarray, list, str, str]:
    """
    Extract a 2-D numpy array + extent for a given timestep.
    Returns (data_2d, extent_list, x_label, y_label).
    """
    dirs = getattr(slc, "extent_dirs", ("x", "z"))

    if use_global and mesh_index is None:
        # Merge all meshes
        try:
            gd = slc.to_global()
            if isinstance(gd, tuple):
                gd = gd[0]  # cell-centered dual-array
        except Exception:
            gd = None
        raw = gd[timestep_idx] if gd is not None else slc[0].data[timestep_idx]
        ext = slc.extent.as_list()
    else:
        mi = int(mesh_index) if mesh_index is not None else 0
        ss = slc[mi]
        raw = ss.data[timestep_idx]
        ext = ss.extent.as_list()

    if raw.ndim == 2:
        return raw, ext, dirs[0], dirs[1]

    # 3-D volumetric (orientation == 0): take mid-z cross-section
    dirs3 = getattr(slc, "extent_dirs", ("x", "y", "z"))
    mid = raw.shape[2] // 2
    return raw[:, :, mid], [ext[0], ext[1], ext[2], ext[3]], dirs3[0], dirs3[1]


# ═════════════════════════════════════════════════════════════════════════════
#  SLICE RENDERING
# ═════════════════════════════════════════════════════════════════════════════

def render_slice(
    sim: fdsreader.Simulation,
    slice_id: int | str,
    timestep_s: float,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "jet",
    show_colorbar: bool = True,
    show_labels: bool = True,
    use_global: bool = True,
    mesh_index: Optional[int] = None,
) -> dict:
    """Render a single slice timestep → { image_b64, actual_time }."""
    _apply_dark_style()
    slc = _resolve_slice(sim, slice_id)
    it = slc.get_nearest_timestep(timestep_s)
    t_actual = float(slc.times[it])

    pd, ext, xdir, ydir = _slice_frame(slc, it, use_global, mesh_index)
    ext2d = _resolve_extent_and_dirs(ext, xdir, ydir)
    pd = pd.T  # imshow expects (rows=y, cols=x)

    fig, ax = plt.subplots(figsize=(10, 7))
    kw: dict[str, Any] = dict(origin="lower", extent=ext2d, cmap=cmap, aspect="equal")
    if vmin is not None:
        kw["vmin"] = vmin
    if vmax is not None:
        kw["vmax"] = vmax
    im = ax.imshow(pd, **kw)

    q = slc.quantity.name if hasattr(slc.quantity, "name") else str(slc.quantity)
    u = slc.quantity.unit if hasattr(slc.quantity, "unit") else ""
    mode = "Global" if (use_global and mesh_index is None) else f"Mesh {mesh_index or 0}"

    if show_colorbar:
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.12, shrink=0.8)
        cb.set_label(f"{q} / {u}", color="#d0d0e0")
        cb.ax.tick_params(colors="#8899aa")
    _apply_uniform_axes(ax, ext2d, xdir, ydir)
    ax.set_title(f"{q} [{mode}] | t = {t_actual:.1f} s", fontsize=14, color="#e94560")

    return {"image_b64": fig_to_base64(fig), "actual_time": t_actual}


def render_slice_multi(
    sim: fdsreader.Simulation,
    slice_id: int | str,
    timesteps_s: list[float],
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "jet",
    use_global: bool = True,
    mesh_index: Optional[int] = None,
) -> dict:
    """Render a 2-column subplot grid for multiple timesteps → { image_b64 }."""
    _apply_dark_style()
    slc = _resolve_slice(sim, slice_id)
    n = len(timesteps_s)
    cols = min(2, n)
    rows = max(1, (n + cols - 1) // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows + 1.2),
                            squeeze=False,
                            gridspec_kw={"hspace": 0.50, "wspace": 0.40})
    q = slc.quantity.name if hasattr(slc.quantity, "name") else str(slc.quantity)
    u = slc.quantity.unit if hasattr(slc.quantity, "unit") else ""

    # Pre-compute frames and derive global colour limits when not user-supplied
    frames_data: list[tuple] = []
    for tv in timesteps_s:
        it = slc.get_nearest_timestep(tv)
        pd, ext, xdir, ydir = _slice_frame(slc, it, use_global, mesh_index)
        ext2d = _resolve_extent_and_dirs(ext, xdir, ydir)
        pd = pd.T
        frames_data.append((pd, ext2d, xdir, ydir, it))

    g_vmin = vmin if vmin is not None else float(min(np.nanmin(fd[0]) for fd in frames_data))
    g_vmax = vmax if vmax is not None else float(max(np.nanmax(fd[0]) for fd in frames_data))

    im = None
    for i, (pd, ext2d, xdir, ydir, it) in enumerate(frames_data):
        r, c = divmod(i, cols)
        ax = axs[r][c]
        kw: dict[str, Any] = dict(origin="lower", extent=ext2d, cmap=cmap, aspect="equal",
                                  vmin=g_vmin, vmax=g_vmax)
        im = ax.imshow(pd, **kw)
        ax.set_title(f"t = {slc.times[it]:.1f} s", fontsize=11, color="#53d8fb",
                     pad=10)
        _apply_uniform_axes(ax, ext2d, xdir, ydir)

    # hide unused axes
    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axs[r][c].set_visible(False)

    fig.suptitle(f"{q} \u2014 Multi-Time Snapshots", fontsize=15,
                 color="#e94560", y=0.98)

    # Shared colour bar with quantity name and units
    if im is not None:
        cbar = fig.colorbar(im, ax=axs.ravel().tolist(), orientation="horizontal",
                            fraction=0.05, pad=0.10, aspect=40)
        cbar_label = f"{q} ({u})" if u else q
        cbar.set_label(cbar_label, color="#d0d0e0", fontsize=11)
        cbar.ax.tick_params(colors="#8899aa", labelsize=9)

    return {"image_b64": fig_to_base64(fig, 110)}


def render_slice_animation_frames(
    sim: fdsreader.Simulation,
    slice_id: int | str,
    t_start: float = 0.0,
    t_end: Optional[float] = None,
    n_frames: int = 20,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "jet",
    use_global: bool = True,
    mesh_index: Optional[int] = None,
) -> list[dict]:
    """Return list of { time, image_b64 } dicts for animation."""
    _apply_dark_style()
    slc = _resolve_slice(sim, slice_id)
    if t_end is None:
        t_end = float(slc.times[-1])
    n_frames = min(n_frames, 60)
    frames: list[dict] = []

    for t in np.linspace(t_start, t_end, n_frames):
        it = slc.get_nearest_timestep(float(t))
        pd, ext, xdir, ydir = _slice_frame(slc, it, use_global, mesh_index)
        ext2d = _resolve_extent_and_dirs(ext, xdir, ydir)
        pd = pd.T
        fig, ax = plt.subplots(figsize=(10, 7))
        kw: dict[str, Any] = dict(origin="lower", extent=ext2d, cmap=cmap, aspect="equal")
        if vmin is not None:
            kw["vmin"] = vmin
        if vmax is not None:
            kw["vmax"] = vmax
        im = ax.imshow(pd, **kw)
        q = slc.quantity.name if hasattr(slc.quantity, "name") else str(slc.quantity)
        u = slc.quantity.unit if hasattr(slc.quantity, "unit") else ""
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.12, shrink=0.8)
        cb.set_label(f"{q} / {u}", color="#d0d0e0")
        cb.ax.tick_params(colors="#8899aa")
        _apply_uniform_axes(ax, ext2d, xdir, ydir)
        ax.set_title(f"{q} | t = {slc.times[it]:.1f} s", fontsize=14, color="#e94560")
        frames.append({"time": float(slc.times[it]), "image_b64": fig_to_base64(fig, 90)})
    return frames


def render_slice_profile(
    sim: fdsreader.Simulation,
    slice_id: int | str,
    direction: str,
    position: float,
    time_s: float,
    use_global: bool = True,
    mesh_index: Optional[int] = None,
) -> dict:
    """Extract and plot a 1-D line profile through a slice."""
    _apply_dark_style()
    slc = _resolve_slice(sim, slice_id)
    it = slc.get_nearest_timestep(time_s)

    # get data + coordinates
    if use_global and mesh_index is None:
        result = slc.to_global(return_coordinates=True)
        if isinstance(result[0], tuple):
            gdata, gc = result[0][0], result[1][0]
        else:
            gdata, gc = result
    else:
        mi = int(mesh_index) if mesh_index is not None else 0
        ss = slc[mi]
        gdata = ss.data
        gc = ss.get_coordinates()

    dirs = list(getattr(slc, "extent_dirs", ("x", "z")))
    frame = gdata[it]
    if frame.ndim == 3:
        frame = frame[:, :, frame.shape[2] // 2]
        dirs = dirs[:2]

    if direction == dirs[0]:
        idx = np.argmin(np.abs(gc[direction] - position))
        prof = frame[idx, :]
        coord = gc[dirs[1]]
        xlabel = f"{dirs[1]} (m)"
    else:
        idx = np.argmin(np.abs(gc[direction] - position))
        prof = frame[:, idx]
        coord = gc[dirs[0]]
        xlabel = f"{dirs[0]} (m)"

    q = slc.quantity.name if hasattr(slc.quantity, "name") else str(slc.quantity)
    u = slc.quantity.unit if hasattr(slc.quantity, "unit") else ""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(coord, prof, color="#53d8fb", lw=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{q} / {u}")
    ax.set_title(f"{q} profile at {direction}={position:.2f} m, t={slc.times[it]:.1f} s",
                 fontsize=14, color="#e94560")
    ax.grid(True, alpha=0.3)
    return {"image_b64": fig_to_base64(fig), "actual_time": float(slc.times[it])}


def render_slice_timeseries(
    sim: fdsreader.Simulation,
    slice_id: int | str,
    point: dict[str, float],
    use_global: bool = True,
    mesh_index: Optional[int] = None,
) -> dict:
    """Extract and plot a time-series at a point within a slice."""
    _apply_dark_style()
    slc = _resolve_slice(sim, slice_id)

    if use_global and mesh_index is None:
        result = slc.to_global(return_coordinates=True)
        if isinstance(result[0], tuple):
            gdata, gc = result[0][0], result[1][0]
        else:
            gdata, gc = result
    else:
        mi = int(mesh_index) if mesh_index is not None else 0
        ss = slc[mi]
        gdata = ss.data
        gc = ss.get_coordinates()

    dirs = list(getattr(slc, "extent_dirs", ("x", "z")))
    c0, c1 = gc[dirs[0]], gc[dirs[1]]
    v0 = point.get(dirs[0], float(c0[len(c0) // 2]))
    v1 = point.get(dirs[1], float(c1[len(c1) // 2]))
    i0 = np.argmin(np.abs(c0 - v0))
    i1 = np.argmin(np.abs(c1 - v1))

    if gdata.ndim == 4:
        ts = gdata[:, i0, i1, gdata.shape[3] // 2]
    else:
        ts = gdata[:, i0, i1]

    q = slc.quantity.name if hasattr(slc.quantity, "name") else str(slc.quantity)
    u = slc.quantity.unit if hasattr(slc.quantity, "unit") else ""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(slc.times, ts, color="#e94560", lw=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"{q} / {u}")
    ax.set_title(f"{q} at ({dirs[0]}={v0:.2f}, {dirs[1]}={v1:.2f})",
                 fontsize=14, color="#e94560")
    ax.grid(True, alpha=0.3)
    return {"image_b64": fig_to_base64(fig)}


# ═════════════════════════════════════════════════════════════════════════════
#  OBSTRUCTION / BOUNDARY METADATA
# ═════════════════════════════════════════════════════════════════════════════

def get_obstruction_metadata(sim: fdsreader.Simulation) -> list[dict]:
    """Return metadata for obstructions that have boundary data."""
    result: list[dict] = []
    for idx, o in enumerate(sim.obstructions or []):
        if not getattr(o, "has_boundary_data", False):
            continue
        times = o.times if hasattr(o, "times") else np.array([])
        result.append({
            "index": idx,
            "id": getattr(o, "id", "") or str(idx),
            "bounding_box": o.bounding_box.as_list() if hasattr(o, "bounding_box") and hasattr(o.bounding_box, "as_list") else [],
            "quantities": [q.name for q in getattr(o, "quantities", [])],
            "quantity_units": [q.unit for q in getattr(o, "quantities", [])],
            "orientations": list(getattr(o, "orientations", [])),
            "orientation_labels": [_orient_label(oi) for oi in getattr(o, "orientations", [])],
            "n_timesteps": len(times),
            "t_start": float(times[0]) if len(times) > 0 else 0.0,
            "t_end": float(times[-1]) if len(times) > 0 else 0.0,
            "times": [round(float(t), 3) for t in times],
            "n_meshes": len(getattr(o, "meshes", [])),
        })
    return result


# ═════════════════════════════════════════════════════════════════════════════
#  BOUNDARY DATA ACCESS
# ═════════════════════════════════════════════════════════════════════════════

def _resolve_obst(sim: fdsreader.Simulation, obst_id: int | str) -> Any:
    """Resolve an obstruction by integer index or string ID."""
    if isinstance(obst_id, str) and not obst_id.isdigit():
        obst = sim.obstructions.get_by_id(obst_id)
        if obst is None:
            raise ValueError(f'Obstruction ID "{obst_id}" not found')
    else:
        obst = sim.obstructions[int(obst_id)]
    if not obst.has_boundary_data:
        raise ValueError(f"Obstruction has no boundary data")
    return obst


def _get_bndf_arr(obst: Any, qty: str, orient: int) -> np.ndarray:
    """Get boundary data array, handling tuple returns at mesh borders."""
    bd = obst.get_global_boundary_data_arrays(qty)
    if orient not in bd:
        raise ValueError(f"Orientation {orient} not available for quantity '{qty}'")
    arr = bd[orient]
    if isinstance(arr, tuple):
        arr = arr[0]
    return arr


# ═════════════════════════════════════════════════════════════════════════════
#  BOUNDARY RENDERING
# ═════════════════════════════════════════════════════════════════════════════

def render_boundary(
    sim: fdsreader.Simulation,
    obst_id: int | str,
    quantity: str,
    orientation: int,
    timestep_s: float,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "hot",
    show_colorbar: bool = True,
) -> dict:
    """Render single boundary timestep → { image_b64, actual_time }."""
    _apply_dark_style()
    obst = _resolve_obst(sim, obst_id)
    it = obst.get_nearest_timestep(timestep_s)
    arr = _get_bndf_arr(obst, quantity, orientation)

    bbox_full = obst.bounding_box.as_list() if hasattr(obst, "bounding_box") and hasattr(obst.bounding_box, "as_list") else []
    xdir, ydir = _ORIENT_DIRS.get(orientation, ('y', 'z'))
    ext2d = _resolve_extent_and_dirs(bbox_full, xdir, ydir)

    fig, ax = plt.subplots(figsize=(10, 7))
    kw: dict[str, Any] = dict(origin="lower", cmap=cmap, aspect="equal", extent=ext2d if ext2d else None)
    if vmin is not None:
        kw["vmin"] = vmin
    if vmax is not None:
        kw["vmax"] = vmax
    # Remove None extent
    if kw.get("extent") is None:
        kw.pop("extent", None)

    im = ax.imshow(arr[it].T, **kw)
    if show_colorbar:
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.12, shrink=0.8)
        cb.set_label(quantity, color="#d0d0e0")
        cb.ax.tick_params(colors="#8899aa")
    _apply_uniform_axes(ax, ext2d, xdir, ydir)
    ax.set_title(f"BNDF: {quantity} | {_orient_label(orientation)} | t = {obst.times[it]:.1f} s",
                 fontsize=14, color="#e94560")
    return {"image_b64": fig_to_base64(fig, 120), "actual_time": float(obst.times[it])}


def render_boundary_multi(
    sim: fdsreader.Simulation,
    obst_id: int | str,
    quantity: str,
    orientation: int,
    timesteps_s: list[float],
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "hot",
) -> dict:
    """Render multi-timestep boundary subplot grid → { image_b64 }."""
    _apply_dark_style()
    obst = _resolve_obst(sim, obst_id)
    arr = _get_bndf_arr(obst, quantity, orientation)
    n = len(timesteps_s)
    cols = min(2, n)
    rows = max(1, (n + cols - 1) // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows + 1.2),
                            squeeze=False,
                            gridspec_kw={"hspace": 0.50, "wspace": 0.40})

    bbox_full = obst.bounding_box.as_list() if hasattr(obst, "bounding_box") and hasattr(obst.bounding_box, "as_list") else []
    xdir, ydir = _ORIENT_DIRS.get(orientation, ('y', 'z'))
    ext2d = _resolve_extent_and_dirs(bbox_full, xdir, ydir)

    # Pre-compute timestep indices and derive global colour limits
    frame_indices: list[int] = []
    for tv in timesteps_s:
        frame_indices.append(obst.get_nearest_timestep(tv))

    g_vmin = vmin if vmin is not None else float(min(np.nanmin(arr[it]) for it in frame_indices))
    g_vmax = vmax if vmax is not None else float(max(np.nanmax(arr[it]) for it in frame_indices))

    im = None
    for i, it in enumerate(frame_indices):
        r, c = divmod(i, cols)
        ax = axs[r][c]
        kw: dict[str, Any] = dict(origin="lower", cmap=cmap, aspect="equal",
                                  vmin=g_vmin, vmax=g_vmax)
        if ext2d:
            kw["extent"] = ext2d
        im = ax.imshow(arr[it].T, **kw)
        ax.set_title(f"t = {obst.times[it]:.1f} s", fontsize=11, color="#53d8fb",
                     pad=10)
        _apply_uniform_axes(ax, ext2d, xdir, ydir)

    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axs[r][c].set_visible(False)

    fig.suptitle(f"BNDF: {quantity} | {_orient_label(orientation)} \u2014 Multi-Time",
                 fontsize=15, color="#e94560", y=0.98)

    # Shared colour bar with quantity name
    if im is not None:
        cbar = fig.colorbar(im, ax=axs.ravel().tolist(), orientation="horizontal",
                            fraction=0.05, pad=0.10, aspect=40)
        cbar.set_label(quantity, color="#d0d0e0", fontsize=11)
        cbar.ax.tick_params(colors="#8899aa", labelsize=9)

    return {"image_b64": fig_to_base64(fig, 110)}


def render_boundary_animation_frames(
    sim: fdsreader.Simulation,
    obst_id: int | str,
    quantity: str,
    orientation: int,
    t_start: float = 0.0,
    t_end: Optional[float] = None,
    n_frames: int = 20,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "hot",
) -> list[dict]:
    """Return list of { time, image_b64 } dicts for boundary animation."""
    _apply_dark_style()
    obst = _resolve_obst(sim, obst_id)
    arr = _get_bndf_arr(obst, quantity, orientation)
    if t_end is None:
        t_end = float(obst.times[-1])
    n_frames = min(n_frames, 60)

    bbox_full = obst.bounding_box.as_list() if hasattr(obst, "bounding_box") and hasattr(obst.bounding_box, "as_list") else []
    xdir, ydir = _ORIENT_DIRS.get(orientation, ('y', 'z'))
    ext2d = _resolve_extent_and_dirs(bbox_full, xdir, ydir)

    frames: list[dict] = []
    for t in np.linspace(t_start, t_end, n_frames):
        it = obst.get_nearest_timestep(float(t))
        fig, ax = plt.subplots(figsize=(10, 7))
        kw: dict[str, Any] = dict(origin="lower", cmap=cmap, aspect="equal")
        if ext2d:
            kw["extent"] = ext2d
        if vmin is not None:
            kw["vmin"] = vmin
        if vmax is not None:
            kw["vmax"] = vmax
        im = ax.imshow(arr[it].T, **kw)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.12, shrink=0.8)
        cb.set_label(quantity, color="#d0d0e0")
        cb.ax.tick_params(colors="#8899aa")
        _apply_uniform_axes(ax, ext2d, xdir, ydir)
        ax.set_title(f"BNDF: {quantity} | {_orient_label(orientation)} | t = {obst.times[it]:.1f} s",
                     fontsize=14, color="#e94560")
        frames.append({"time": float(obst.times[it]), "image_b64": fig_to_base64(fig, 90)})
    return frames


def render_boundary_timeseries(
    sim: fdsreader.Simulation,
    obst_id: int | str,
    quantity: str,
    orientation: int,
) -> dict:
    """Plot boundary mean value over time → { image_b64 }."""
    _apply_dark_style()
    obst = _resolve_obst(sim, obst_id)
    arr = _get_bndf_arr(obst, quantity, orientation)
    # centre pixel
    ix, iy = arr.shape[1] // 2, arr.shape[2] // 2

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(obst.times, arr[:, ix, iy], color="#e94560", lw=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(quantity)
    ax.set_title(f"BNDF Time-Series \u2014 {quantity} (centre pixel)", fontsize=14, color="#e94560")
    ax.grid(True, alpha=0.3)
    return {"image_b64": fig_to_base64(fig)}


# ═════════════════════════════════════════════════════════════════════════════
#  GIF EXPORT HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _make_gif(pil_frames: list, fps: int = 4) -> io.BytesIO:
    """Build an animated GIF from a list of PIL Image objects."""
    buf = io.BytesIO()
    pil_frames[0].save(buf, format="GIF", save_all=True,
                       append_images=pil_frames[1:],
                       duration=int(1000 / fps), loop=0, optimize=True)
    buf.seek(0)
    return buf


def export_slice_gif(
    sim: fdsreader.Simulation,
    slice_id: int | str,
    t_start: float = 0.0,
    t_end: Optional[float] = None,
    n_frames: int = 20,
    fps: int = 4,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "jet",
    use_global: bool = True,
    mesh_index: Optional[int] = None,
) -> tuple[io.BytesIO, str]:
    """Generate slice animation GIF. Returns (gif_bytes, filename)."""
    from PIL import Image
    _apply_dark_style()
    slc = _resolve_slice(sim, slice_id)
    if t_end is None:
        t_end = float(slc.times[-1])
    n_frames = min(n_frames, 60)

    pil_frames = []
    for t in np.linspace(t_start, t_end, n_frames):
        it = slc.get_nearest_timestep(float(t))
        pd, ext, xdir, ydir = _slice_frame(slc, it, use_global, mesh_index)
        ext2d = _resolve_extent_and_dirs(ext, xdir, ydir)
        pd = pd.T
        fig, ax = plt.subplots(figsize=(10, 7))
        kw: dict = dict(origin="lower", extent=ext2d, cmap=cmap, aspect="equal")
        if vmin is not None:
            kw["vmin"] = vmin
        if vmax is not None:
            kw["vmax"] = vmax
        im = ax.imshow(pd, **kw)
        q = slc.quantity.name if hasattr(slc.quantity, "name") else str(slc.quantity)
        u = slc.quantity.unit if hasattr(slc.quantity, "unit") else ""
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.12, shrink=0.8)
        cb.set_label(f"{q} / {u}", color="#d0d0e0")
        cb.ax.tick_params(colors="#8899aa")
        _apply_uniform_axes(ax, ext2d, xdir, ydir)
        ax.set_title(f"{q} | t = {slc.times[it]:.1f} s", fontsize=14, color="#e94560")
        buf = _fig_to_raw_bytes(fig, 120)
        pil_frames.append(Image.open(buf).convert("RGB"))

    gif_buf = _make_gif(pil_frames, fps)
    q = slc.quantity.name if hasattr(slc.quantity, "name") else "slice"
    label = getattr(slc, "id", "") or slice_id
    return gif_buf, f"slice_{label}_{q}_animation.gif"


def export_boundary_gif(
    sim: fdsreader.Simulation,
    obst_id: int | str,
    quantity: str,
    orientation: int,
    t_start: float = 0.0,
    t_end: Optional[float] = None,
    n_frames: int = 20,
    fps: int = 4,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "hot",
) -> tuple[io.BytesIO, str]:
    """Generate boundary animation GIF. Returns (gif_bytes, filename)."""
    from PIL import Image
    _apply_dark_style()
    obst = _resolve_obst(sim, obst_id)
    arr = _get_bndf_arr(obst, quantity, orientation)
    if t_end is None:
        t_end = float(obst.times[-1])
    n_frames = min(n_frames, 60)

    bbox_full = obst.bounding_box.as_list() if hasattr(obst, "bounding_box") and hasattr(obst.bounding_box, "as_list") else []
    xdir, ydir = _ORIENT_DIRS.get(orientation, ('y', 'z'))
    ext2d = _resolve_extent_and_dirs(bbox_full, xdir, ydir)

    pil_frames = []
    for t in np.linspace(t_start, t_end, n_frames):
        it = obst.get_nearest_timestep(float(t))
        fig, ax = plt.subplots(figsize=(10, 7))
        kw: dict = dict(origin="lower", cmap=cmap, aspect="equal")
        if ext2d:
            kw["extent"] = ext2d
        if vmin is not None:
            kw["vmin"] = vmin
        if vmax is not None:
            kw["vmax"] = vmax
        im = ax.imshow(arr[it].T, **kw)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.12, shrink=0.8)
        cb.set_label(quantity, color="#d0d0e0")
        cb.ax.tick_params(colors="#8899aa")
        _apply_uniform_axes(ax, ext2d, xdir, ydir)
        ax.set_title(f"BNDF: {quantity} | t = {obst.times[it]:.1f} s", fontsize=14, color="#e94560")
        buf = _fig_to_raw_bytes(fig, 120)
        pil_frames.append(Image.open(buf).convert("RGB"))

    gif_buf = _make_gif(pil_frames, fps)
    oid_label = getattr(obst, "id", "") or obst_id
    return gif_buf, f"boundary_{oid_label}_{quantity}_animation.gif"


# ═════════════════════════════════════════════════════════════════════════════
#  DIRECTORY BROWSER
# ═════════════════════════════════════════════════════════════════════════════

def list_directory(path: str) -> dict:
    """List subdirectories and detect .smv files for the folder-picker UI."""
    import os
    path = path or ""
    if not path:
        # Return filesystem roots / home directory
        if os.name == "nt":
            import string
            drives = [f"{d}:\\" for d in string.ascii_uppercase
                      if os.path.exists(f"{d}:\\")]
            return {"path": "", "parent": "", "dirs": drives, "has_smv": False}
        else:
            path = os.path.expanduser("~")
    if not os.path.isdir(path):
        raise ValueError(f"Not a directory: {path}")

    dirs: list[str] = []
    has_smv = False
    try:
        for entry in sorted(os.scandir(path), key=lambda e: e.name.lower()):
            if entry.is_dir() and not entry.name.startswith("."):
                dirs.append(entry.name)
            elif entry.name.endswith(".smv"):
                has_smv = True
    except PermissionError:
        pass
    parent = os.path.dirname(path.rstrip(os.sep + "/"))
    return {"path": path, "parent": parent, "dirs": dirs, "has_smv": has_smv}


# ═════════════════════════════════════════════════════════════════════════════
#  PLOT3D (PL3D) — 3D VOLUMETRIC DATA
# ═════════════════════════════════════════════════════════════════════════════

def get_plot3d_metadata(sim: fdsreader.Simulation) -> list[dict]:
    """Return metadata for all Plot3D data objects."""
    result: list[dict] = []
    data_3d = getattr(sim, "data_3d", None)
    if not data_3d:
        return result
    for idx, p3d in enumerate(data_3d):
        times = list(getattr(p3d, "times", []))
        quantities = []
        for q in getattr(p3d, "quantities", []):
            qn = q.name if hasattr(q, "name") else str(q)
            qu = q.unit if hasattr(q, "unit") else ""
            quantities.append({"name": qn, "unit": qu})
        result.append({
            "index": idx,
            "n_timesteps": len(times),
            "times": [round(float(t), 3) for t in times],
            "quantities": quantities,
            "n_meshes": len(getattr(p3d, "subplots", [])),
        })
    return result


def render_plot3d_cutplane(
    sim: fdsreader.Simulation,
    p3d_index: int = 0,
    time_idx: int = 0,
    quantity_idx: int = 0,
    axis: str = "z",
    position: Optional[float] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "jet",
    show_colorbar: bool = True,
) -> dict:
    """Render a 2D cut-plane through Plot3D volumetric data."""
    _apply_dark_style()
    p3d = sim.data_3d[p3d_index]
    try:
        gd = p3d.to_global(return_coordinates=True)
        if isinstance(gd, tuple) and len(gd) == 2:
            vol, coords = gd
        else:
            vol = p3d.to_global()
            coords = None
    except Exception:
        sp = p3d.subplots[0]
        vol = sp.data
        coords = sp.mesh.coordinates if hasattr(sp, "mesh") else None

    # vol shape: (n_t, nx, ny, nz) or (n_t, n_quantities, nx, ny, nz)
    if vol.ndim == 5:
        frame = vol[time_idx, quantity_idx]
    elif vol.ndim == 4:
        frame = vol[time_idx]
    else:
        raise ValueError(f"Unexpected Plot3D data shape: {vol.shape}")

    # Determine slice axis and position
    axis_map = {"x": 0, "y": 1, "z": 2}
    ax_idx = axis_map.get(axis.lower(), 2)
    if position is not None and coords is not None:
        ax_label = axis.lower()
        if isinstance(coords, dict) and ax_label in coords:
            coord_arr = np.array(coords[ax_label])
            slice_idx = int(np.argmin(np.abs(coord_arr - position)))
        else:
            slice_idx = frame.shape[ax_idx] // 2
    else:
        slice_idx = frame.shape[ax_idx] // 2

    # Extract 2D slice
    if ax_idx == 0:
        data_2d = frame[slice_idx, :, :]
        xlabel, ylabel = "y (m)", "z (m)"
    elif ax_idx == 1:
        data_2d = frame[:, slice_idx, :]
        xlabel, ylabel = "x (m)", "z (m)"
    else:
        data_2d = frame[:, :, slice_idx]
        xlabel, ylabel = "x (m)", "y (m)"

    fig, ax = plt.subplots(figsize=(10, 7))
    kw: dict[str, Any] = dict(origin="lower", cmap=cmap, aspect="auto")
    if vmin is not None:
        kw["vmin"] = vmin
    if vmax is not None:
        kw["vmax"] = vmax
    im = ax.imshow(data_2d.T, **kw)

    q = p3d.quantities[quantity_idx] if quantity_idx < len(p3d.quantities) else None
    qname = q.name if q and hasattr(q, "name") else f"Qty {quantity_idx}"
    qunit = q.unit if q and hasattr(q, "unit") else ""
    t_val = p3d.times[time_idx] if time_idx < len(p3d.times) else 0

    if show_colorbar:
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.12, shrink=0.8)
        cb.set_label(f"{qname} ({qunit})", color="#d0d0e0")
        cb.ax.tick_params(colors="#8899aa")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Plot3D: {qname} | {axis}={position or 'mid'} | t={t_val:.1f} s",
                 fontsize=14, color="#e94560")
    return {"image_b64": fig_to_base64(fig)}


# ═════════════════════════════════════════════════════════════════════════════
#  SMOKE3D (S3D) — VOLUMETRIC SMOKE DATA
# ═════════════════════════════════════════════════════════════════════════════

def get_smoke3d_metadata(sim: fdsreader.Simulation) -> list[dict]:
    """Return metadata for all Smoke3D data objects."""
    result: list[dict] = []
    s3d = getattr(sim, "smoke_3d", None)
    if not s3d:
        return result
    for idx, smoke in enumerate(s3d):
        qn = smoke.quantity.name if hasattr(smoke.quantity, "name") else str(smoke.quantity)
        qu = smoke.quantity.unit if hasattr(smoke.quantity, "unit") else ""
        times = list(getattr(smoke, "times", []))
        result.append({
            "index": idx,
            "quantity": qn,
            "unit": qu,
            "n_timesteps": len(times),
            "times": [round(float(t), 3) for t in times],
            "n_meshes": len(getattr(smoke, "subsmokes", [])),
        })
    return result


def render_smoke3d_cutplane(
    sim: fdsreader.Simulation,
    smoke_index: int = 0,
    time_idx: int = 0,
    axis: str = "z",
    position: Optional[float] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "hot",
    show_colorbar: bool = True,
) -> dict:
    """Render a 2D cut-plane through Smoke3D data."""
    _apply_dark_style()
    smoke = sim.smoke_3d[smoke_index]
    try:
        gd = smoke.to_global(return_coordinates=True)
        if isinstance(gd, tuple) and len(gd) == 2:
            vol, coords = gd
        else:
            vol = smoke.to_global()
            coords = None
    except Exception:
        ss = smoke.subsmokes[0]
        vol = ss.data
        coords = ss.mesh.coordinates if hasattr(ss, "mesh") else None

    if vol.ndim < 4:
        raise ValueError(f"Smoke3D data has unexpected shape: {vol.shape}")
    frame = vol[time_idx]

    axis_map = {"x": 0, "y": 1, "z": 2}
    ax_idx = axis_map.get(axis.lower(), 2)
    if position is not None and coords is not None:
        ax_label = axis.lower()
        if isinstance(coords, dict) and ax_label in coords:
            coord_arr = np.array(coords[ax_label])
            slice_idx = int(np.argmin(np.abs(coord_arr - position)))
        else:
            slice_idx = frame.shape[ax_idx] // 2
    else:
        slice_idx = frame.shape[ax_idx] // 2

    if ax_idx == 0:
        data_2d = frame[slice_idx, :, :]
        xlabel, ylabel = "y (m)", "z (m)"
    elif ax_idx == 1:
        data_2d = frame[:, slice_idx, :]
        xlabel, ylabel = "x (m)", "z (m)"
    else:
        data_2d = frame[:, :, slice_idx]
        xlabel, ylabel = "x (m)", "y (m)"

    fig, ax = plt.subplots(figsize=(10, 7))
    kw: dict[str, Any] = dict(origin="lower", cmap=cmap, aspect="auto")
    if vmin is not None:
        kw["vmin"] = vmin
    if vmax is not None:
        kw["vmax"] = vmax
    im = ax.imshow(data_2d.T, **kw)

    qname = smoke.quantity.name if hasattr(smoke.quantity, "name") else "Smoke3D"
    qunit = smoke.quantity.unit if hasattr(smoke.quantity, "unit") else ""
    t_val = smoke.times[time_idx] if time_idx < len(smoke.times) else 0

    if show_colorbar:
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.12, shrink=0.8)
        cb.set_label(f"{qname} ({qunit})", color="#d0d0e0")
        cb.ax.tick_params(colors="#8899aa")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Smoke3D: {qname} | {axis}={position or 'mid'} | t={t_val:.1f} s",
                 fontsize=14, color="#e94560")
    return {"image_b64": fig_to_base64(fig)}


# ═════════════════════════════════════════════════════════════════════════════
#  PARTICLES (PART) — LAGRANGIAN PARTICLE DATA
# ═════════════════════════════════════════════════════════════════════════════

def get_particle_metadata(sim: fdsreader.Simulation) -> list[dict]:
    """Return metadata for all particle classes."""
    result: list[dict] = []
    parts = getattr(sim, "particles", None)
    if not parts:
        return result
    for idx, p in enumerate(parts):
        quantities = []
        for q in getattr(p, "quantities", []):
            qn = q.name if hasattr(q, "name") else str(q)
            qu = q.unit if hasattr(q, "unit") else ""
            quantities.append({"name": qn, "unit": qu})
        n_timesteps = len(getattr(p, "positions", []))
        result.append({
            "index": idx,
            "class_name": getattr(p, "class_name", f"Class_{idx}"),
            "quantities": quantities,
            "n_timesteps": n_timesteps,
            "color": list(getattr(p, "color", (1, 0, 0))),
        })
    return result


def render_particle_scatter(
    sim: fdsreader.Simulation,
    class_index: int = 0,
    time_idx: int = 0,
    plane: str = "xy",
    color_quantity: Optional[str] = None,
    cmap: str = "jet",
    show_colorbar: bool = True,
) -> dict:
    """Render 2D scatter of particle positions at a given timestep."""
    _apply_dark_style()
    part = sim.particles[class_index]
    positions = part.positions

    if time_idx >= len(positions):
        raise ValueError(f"Time index {time_idx} out of range (max {len(positions)-1})")

    pos = positions[time_idx]  # shape (n_particles, 3)
    if len(pos) == 0:
        _apply_dark_style()
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.text(0.5, 0.5, "No particles at this timestep", ha="center",
                va="center", color="#888", fontsize=16, transform=ax.transAxes)
        return {"image_b64": fig_to_base64(fig)}

    plane_map = {
        "xy": (0, 1, "x (m)", "y (m)"),
        "xz": (0, 2, "x (m)", "z (m)"),
        "yz": (1, 2, "y (m)", "z (m)"),
    }
    i0, i1, xlabel, ylabel = plane_map.get(plane.lower(), (0, 1, "x (m)", "y (m)"))

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter_kw: dict = dict(s=3, alpha=0.6)

    if color_quantity and hasattr(part, "data"):
        data = part.data
        if color_quantity in data and time_idx < len(data[color_quantity]):
            c_data = data[color_quantity][time_idx]
            sc = ax.scatter(pos[:, i0], pos[:, i1], c=c_data, cmap=cmap, **scatter_kw)
            if show_colorbar:
                cb = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.12, shrink=0.8)
                cb.set_label(color_quantity, color="#d0d0e0")
                cb.ax.tick_params(colors="#8899aa")
        else:
            ax.scatter(pos[:, i0], pos[:, i1], color="#53d8fb", **scatter_kw)
    else:
        ax.scatter(pos[:, i0], pos[:, i1], color="#53d8fb", **scatter_kw)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cname = getattr(part, "class_name", f"Class {class_index}")
    ax.set_title(f"Particles: {cname} | n={len(pos)} | t_idx={time_idx}",
                 fontsize=14, color="#e94560")
    ax.grid(True, alpha=0.2)
    return {"image_b64": fig_to_base64(fig), "n_particles": len(pos)}


def render_particle_histogram(
    sim: fdsreader.Simulation,
    class_index: int = 0,
    quantity: str = "",
    time_idx: int = 0,
    bins: int = 50,
) -> dict:
    """Render histogram of a particle quantity at a timestep."""
    _apply_dark_style()
    part = sim.particles[class_index]
    data = getattr(part, "data", {})
    if quantity not in data:
        raise ValueError(f"Quantity '{quantity}' not found in particle data")
    vals = data[quantity][time_idx]
    if len(vals) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No particles at this timestep", ha="center",
                va="center", color="#888", fontsize=16, transform=ax.transAxes)
        return {"image_b64": fig_to_base64(fig)}

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(vals, bins=bins, color="#53d8fb", edgecolor="#1a1a2e", alpha=0.8)
    ax.set_xlabel(quantity)
    ax.set_ylabel("Count")
    cname = getattr(part, "class_name", f"Class {class_index}")
    ax.set_title(f"Particle Histogram: {cname} — {quantity} | t_idx={time_idx}",
                 fontsize=14, color="#e94560")
    ax.grid(True, alpha=0.3)
    return {"image_b64": fig_to_base64(fig)}


# ═════════════════════════════════════════════════════════════════════════════
#  ISOSURFACES (ISOF) — ISO-CONTOUR DATA
# ═════════════════════════════════════════════════════════════════════════════

def get_isosurface_metadata(sim: fdsreader.Simulation) -> list[dict]:
    """Return metadata for all isosurface data objects."""
    result: list[dict] = []
    isos = getattr(sim, "isosurfaces", None)
    if not isos:
        return result
    for idx, iso in enumerate(isos):
        qn = iso.quantity.name if hasattr(iso, "quantity") and hasattr(iso.quantity, "name") else f"Iso {idx}"
        qu = iso.quantity.unit if hasattr(iso, "quantity") and hasattr(iso.quantity, "unit") else ""
        times = list(getattr(iso, "times", []))
        result.append({
            "index": idx,
            "quantity": qn,
            "unit": qu,
            "n_timesteps": len(times),
            "times": [round(float(t), 3) for t in times],
            "has_color_data": getattr(iso, "has_color_data", False),
        })
    return result


def render_isosurface_scatter(
    sim: fdsreader.Simulation,
    iso_index: int = 0,
    time_idx: int = 0,
    plane: str = "xy",
    cmap: str = "hot",
    show_colorbar: bool = True,
) -> dict:
    """Render a 2D projection of isosurface vertex positions."""
    _apply_dark_style()
    iso = sim.isosurfaces[iso_index]
    t = iso.times[time_idx] if time_idx < len(iso.times) else iso.times[-1]

    try:
        result = iso.to_global(t)
        if isinstance(result, tuple):
            vertices = result[0]  # global vertex array
            color_data = result[2] if len(result) > 2 else None
        else:
            vertices = result
            color_data = None
    except Exception as e:
        raise ValueError(f"Cannot extract isosurface data: {e}")

    if vertices is None or len(vertices) == 0:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.text(0.5, 0.5, "No isosurface data at this timestep", ha="center",
                va="center", color="#888", fontsize=16, transform=ax.transAxes)
        return {"image_b64": fig_to_base64(fig)}

    plane_map = {
        "xy": (0, 1, "x (m)", "y (m)"),
        "xz": (0, 2, "x (m)", "z (m)"),
        "yz": (1, 2, "y (m)", "z (m)"),
    }
    i0, i1, xlabel, ylabel = plane_map.get(plane.lower(), (0, 1, "x (m)", "y (m)"))

    fig, ax = plt.subplots(figsize=(10, 7))
    if color_data is not None and len(color_data) == len(vertices):
        sc = ax.scatter(vertices[:, i0], vertices[:, i1], c=color_data,
                        cmap=cmap, s=1, alpha=0.5)
        if show_colorbar:
            cb = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.12, shrink=0.8)
            cb.ax.tick_params(colors="#8899aa")
    else:
        ax.scatter(vertices[:, i0], vertices[:, i1], color="#f5a623", s=1, alpha=0.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    qn = iso.quantity.name if hasattr(iso.quantity, "name") else f"Iso {iso_index}"
    ax.set_title(f"Isosurface: {qn} | {len(vertices)} vertices | t={t:.1f} s",
                 fontsize=14, color="#e94560")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    return {"image_b64": fig_to_base64(fig), "n_vertices": len(vertices)}


# ═════════════════════════════════════════════════════════════════════════════
#  EVACUATION (EVAC) — AGENT / PEDESTRIAN DATA
# ═════════════════════════════════════════════════════════════════════════════

def get_evac_metadata(sim: fdsreader.Simulation) -> dict:
    """Return metadata for evacuation data."""
    evacs = getattr(sim, "evacs", None)
    if not evacs:
        return {"classes": [], "n_timesteps": 0}
    times = list(getattr(evacs, "times", []))
    classes = []
    for idx, ev in enumerate(evacs):
        quantities = []
        for q in getattr(ev, "quantities", []):
            qn = q.name if hasattr(q, "name") else str(q)
            qu = q.unit if hasattr(q, "unit") else ""
            quantities.append({"name": qn, "unit": qu})
        classes.append({
            "index": idx,
            "class_name": getattr(ev, "class_name", f"Class_{idx}"),
            "quantities": quantities,
            "color": list(getattr(ev, "color", (0, 1, 0))),
        })
    return {
        "classes": classes,
        "n_timesteps": len(times),
        "times": [round(float(t), 3) for t in times[:200]],  # cap for JSON
    }


def render_evac_floorplan(
    sim: fdsreader.Simulation,
    time_idx: int = 0,
    class_index: Optional[int] = None,
) -> dict:
    """Render 2D floor plan scatter of evacuation agent positions."""
    _apply_dark_style()
    evacs = sim.evacs
    fig, ax = plt.subplots(figsize=(10, 8))
    total_agents = 0

    if class_index is not None:
        classes_to_plot = [evacs[class_index]]
    else:
        classes_to_plot = list(evacs)

    for ci, ev in enumerate(classes_to_plot):
        positions = getattr(ev, "positions", [])
        if time_idx >= len(positions):
            continue
        pos = positions[time_idx]
        if len(pos) == 0:
            continue
        color = COLORS[ci % len(COLORS)]
        cname = getattr(ev, "class_name", f"Class {ci}")
        ax.scatter(pos[:, 0], pos[:, 1], s=8, alpha=0.7,
                   color=color, label=f"{cname} ({len(pos)})")
        total_agents += len(pos)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Evacuation Floor Plan | {total_agents} agents | t_idx={time_idx}",
                 fontsize=14, color="#e94560")
    ax.legend(framealpha=0.7)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    return {"image_b64": fig_to_base64(fig), "n_agents": total_agents}


def render_evac_timeseries(
    sim: fdsreader.Simulation,
    metric: str = "agents",
) -> dict:
    """Plot evacuation time-series: agent count, deaths, FED, or exit counters."""
    _apply_dark_style()
    evacs = sim.evacs
    times = np.array(getattr(evacs, "times", []))
    if len(times) == 0:
        raise ValueError("No evacuation time data available")

    fig, ax = plt.subplots(figsize=(12, 6))

    if metric == "agents":
        vals = np.array(getattr(evacs, "all_agents", []))
        ax.plot(times[:len(vals)], vals, color="#53d8fb", lw=1.5, label="Total agents")
        ax.set_ylabel("Agent count")
        title = "Evacuation — Agent Count Over Time"
    elif metric == "deaths":
        vals = np.array(getattr(evacs, "number_of_deads", []))
        ax.plot(times[:len(vals)], vals, color="#e94560", lw=1.5, label="Deaths")
        ax.set_ylabel("Count")
        title = "Evacuation — Fatalities Over Time"
    elif metric == "fed":
        vals = np.array(getattr(evacs, "fed_max", []))
        ax.plot(times[:len(vals)], vals, color="#f5a623", lw=1.5, label="FED_max")
        fed_alive = np.array(getattr(evacs, "fed_max_alive", []))
        if len(fed_alive) > 0:
            ax.plot(times[:len(fed_alive)], fed_alive, color="#7ed321",
                    lw=1.5, linestyle="--", label="FED_max (alive)")
        ax.set_ylabel("FED")
        title = "Evacuation — Fractional Effective Dose"
    elif metric == "exits":
        exits = getattr(evacs, "exit_counters", [])
        if hasattr(exits, '__len__') and len(exits) > 0:
            exits = np.array(exits)
            ax.plot(times[:len(exits)], exits, color="#50e3c2", lw=1.5, label="Exits")
        ax.set_ylabel("Count")
        title = "Evacuation — Exit Counters"
    else:
        raise ValueError(f"Unknown metric: {metric}")

    ax.set_xlabel("Time (s)")
    ax.legend(framealpha=0.7)
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=14, color="#e94560")
    return {"image_b64": fig_to_base64(fig)}


# ═════════════════════════════════════════════════════════════════════════════
#  PERFORMANCE — CPU, STEPS, PROFILES
# ═════════════════════════════════════════════════════════════════════════════

def get_cpu_metadata(sim: fdsreader.Simulation) -> dict:
    """Return CPU CSV column names and time range."""
    cpu = getattr(sim, "cpu", None)
    if not cpu:
        return {"columns": [], "t_start": 0, "t_end": 0, "n_points": 0}
    cols = list(cpu.keys())
    time = cpu.get("Time", np.array([]))
    return {
        "columns": cols,
        "t_start": float(time[0]) if len(time) > 0 else 0.0,
        "t_end": float(time[-1]) if len(time) > 0 else 0.0,
        "n_points": len(time),
    }


def render_cpu(
    sim: fdsreader.Simulation,
    columns: list[str],
    time_range: Optional[list[float]] = None,
    show_grid: bool = True,
) -> dict:
    """Plot selected CPU columns → { image_b64 }."""
    _apply_dark_style()
    cpu = sim.cpu
    time = cpu["Time"]
    mask = np.ones(len(time), dtype=bool)
    if time_range and len(time_range) == 2:
        mask = (time >= time_range[0]) & (time <= time_range[1])

    fig, ax = plt.subplots(figsize=(12, 6))
    n_plotted = 0
    for i, col in enumerate(columns):
        if col == "Time" or col not in cpu:
            continue
        ax.plot(time[mask], cpu[col][mask], label=col,
                color=COLORS[i % len(COLORS)], lw=1.5)
        n_plotted += 1
    if not n_plotted:
        plt.close(fig)
        raise ValueError("No valid CPU columns found")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("CPU Time (s)")
    ax.legend(framealpha=0.7)
    if show_grid:
        ax.grid(True, alpha=0.3)
    ax.set_title("CPU Performance", fontsize=14, color="#e94560")
    return {"image_b64": fig_to_base64(fig)}


def get_steps_metadata(sim: fdsreader.Simulation) -> dict:
    """Return Steps CSV column names and time range."""
    steps = getattr(sim, "steps", None)
    if not steps:
        return {"columns": [], "t_start": 0, "t_end": 0, "n_points": 0}
    cols = list(steps.keys())
    time = steps.get("Time", np.array([]))
    return {
        "columns": cols,
        "t_start": float(time[0]) if len(time) > 0 else 0.0,
        "t_end": float(time[-1]) if len(time) > 0 else 0.0,
        "n_points": len(time),
    }


def render_steps(
    sim: fdsreader.Simulation,
    columns: list[str],
    time_range: Optional[list[float]] = None,
    show_grid: bool = True,
) -> dict:
    """Plot selected Steps columns → { image_b64 }."""
    _apply_dark_style()
    steps = sim.steps
    time = steps["Time"]
    mask = np.ones(len(time), dtype=bool)
    if time_range and len(time_range) == 2:
        mask = (time >= time_range[0]) & (time <= time_range[1])

    fig, ax = plt.subplots(figsize=(12, 6))
    n_plotted = 0
    for i, col in enumerate(columns):
        if col == "Time" or col not in steps:
            continue
        ax.plot(time[mask], steps[col][mask], label=col,
                color=COLORS[i % len(COLORS)], lw=1.5)
        n_plotted += 1
    if not n_plotted:
        plt.close(fig)
        raise ValueError("No valid Steps columns found")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Value")
    ax.legend(framealpha=0.7)
    if show_grid:
        ax.grid(True, alpha=0.3)
    ax.set_title("Timestep Data", fontsize=14, color="#e94560")
    return {"image_b64": fig_to_base64(fig)}
