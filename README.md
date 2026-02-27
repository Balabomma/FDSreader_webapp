# FDSReader_webapp — Web Interface for fdsreader

A comprehensive Flask web application providing interactive visualization of FDS (Fire Dynamics Simulator) output data using the `fdsreader` Python module.

## Features

### Device Data (DEVC)
- Select and compare multiple device time-series
- Dual-axis comparison mode (left/right axes)
- Quantity-based filtering
- Customizable time range filtering
- Automatic quantity/unit labelling

### Heat Release Rate (HRR)
- Plot HRR and energy balance columns (Q_RADI, Q_CONV, Q_COND, etc.)
- Quick presets (HRR Only, All Q_*, All Columns)
- Customizable time range

### Slice Data (SLCF)
- **2D contour visualization** with time slider
- **Multi-time snapshot grids** (2-column layout)
- **Time-step animation** with play/pause/seek controls
- **Line profile extraction** along any spatial direction
- **Point time-series** at any (x,y,z) location
- **Multi-mesh support** via `to_global()` merging
- Customizable colormaps and value ranges

### Boundary Data (BNDF)
- **2D boundary field visualization** by obstruction
- **Orientation and quantity selection**
- **Multi-time snapshot grids** with shared colour scale
- **Time-step animation** with playback controls
- **Center-point time-series**
- Customizable colormaps and value ranges

### Plot3D Data (PL3D)
- 3D volumetric cut-plane visualization
- Axis selection (X, Y, Z) with position control
- Customizable colormaps and value ranges

### Smoke3D Data (S3D)
- Volumetric smoke cut-plane visualization
- Axis selection with position control
- Customizable colormaps and value ranges

### Particle Data (PART)
- 2D scatter plot of particle positions (XY, XZ, YZ planes)
- Colour-by-quantity support
- Histogram visualization of particle quantities

### Isosurface Data (ISOF)
- 2D projection of isosurface vertex positions
- Colour data support when available

### Evacuation Data (EVAC)
- Agent floor plan scatter at any timestep
- Time-series metrics: agent count, deaths, FED, exit counters
- Class-based filtering

### Performance (CPU & Steps)
- CPU time column plotting
- Timestep data visualization
- Customizable time range and column selection

### High-Resolution PNG Export
- **Single-frame HD PNG download** (300 DPI, 12×8 inches) for all data types
- **Multi-timestep separate PNGs** — one PNG per selected timestep with **shared colour scale** and embedded colorbar, packaged as a ZIP
- **Animation PNG sequences** — evenly-spaced frames as ZIP for slices, boundaries, smoke3d, and particles
- Dark and light (print-friendly) theme support
- Available from every viewer page via dedicated download buttons

### GIF Animation Export
- Animated GIF export for slice and boundary animations
- Configurable frame count and FPS

### General
- Dark theme optimized for fire engineering analysis
- Keyboard shortcuts (Space: play/pause, Arrow keys: step)
- Simulation caching for fast re-access
- Directory browser for simulation folder selection
- Responsive sidebar + main plot layout
- 11 dedicated viewer pages with separate HTML templates

## Installation

Follow the steps below for your operating system.

---

### Step 1 — Install Git

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install git
git --version
```

**Windows:**
1. Download Git from [https://git-scm.com/download/win](https://git-scm.com/install/windows)
2. Run the installer (keep default options)
3. Verify installation by opening **Command Prompt** or **Git Bash**:
```cmd
git --version
```

---

### Step 2 — Clone or Download the Repository

**Option A — Clone with Git (recommended):**

```bash
# Linux / Windows Git Bash / Windows Command Prompt
git clone https://github.com/balabomma/FDSreader_webapp.git
cd FDSreader_webapp
```

**Option B — Download ZIP:**
1. Go to the repository page on GitHub
2. Click **Code → Download ZIP**
3. Extract the ZIP to a folder of your choice
4. Open a terminal / Command Prompt and `cd` into that folder

---

### Step 3 — Install Python or Anaconda & Create a Virtual Environment

Choose **either** the standard Python approach **or** the Anaconda approach.

---

#### Option A — Standard Python (venv)

**Linux:**
```bash
# Install Python 3 if not already installed
sudo apt-get install python3 python3-pip python3-venv

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Your prompt will change to (venv) when active
```

**Windows (Command Prompt):**
```cmd
:: Download Python 3.8+ from https://www.python.org/downloads/
:: During installation, check "Add Python to PATH"

:: Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

:: Your prompt will change to (venv) when active
```

---

#### Option B — Anaconda / Miniconda

Download and install:
- **Anaconda** (full): https://www.anaconda.com/download
- **Miniconda** (lightweight): https://docs.conda.io/en/latest/miniconda.html

**Linux:**
```bash
# After installing Miniconda/Anaconda, create a new environment
conda create -n fdsreader python=3.10
conda activate fdsreader
```

**Windows (Anaconda Prompt):**
```cmd
conda create -n fdsreader python=3.12
conda activate fdsreader
```

---

### Step 4 — Install Dependencies

With your virtual environment **activated**, run:

**Linux & Windows:**
```bash
pip install -r requirements.txt
```

> **Tip:** If `pip` is slow, you can use `pip install -r requirements.txt --no-cache-dir` or install with conda:
> ```bash
> conda install --file requirements.txt
> ```

---

### Step 5 — Run the Flask App

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

### Quick Reference Summary

| Step | Linux | Windows |
|------|-------|---------|
| Install Git | `sudo apt-get install git` | Download from git-scm.com |
| Clone repo | `git clone <url>` | `git clone <url>` |
| Install Python | `sudo apt-get install python3` | Download from python.org |
| Create venv (Python) | `python3 -m venv venv` | `python -m venv venv` |
| Activate venv (Python) | `source venv/bin/activate` | `venv\Scripts\activate` |
| Create env (Conda) | `conda create -n fdsreader python=3.10` | same |
| Activate env (Conda) | `conda activate fdsreader` | same (Anaconda Prompt) |
| Install deps | `pip install -r requirements.txt` | same |
| Run app | `python app.py` | same |

## Usage

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

1. Enter the path to your FDS simulation output directory (the folder containing `.smv` and output files), or use the **Browse** button to navigate
2. Click **Load**
3. Use the navigation bar to switch between viewers:
   - **Slices** — 2D contour plots, profiles, time-series, animation
   - **Boundaries** — Obstruction surface data with orientation control
   - **Devices** — Time-series plots with dual-axis comparison
   - **HRR** — Heat release rate and energy balance
   - **Plot3D** — 3D volumetric cut-planes
   - **Smoke3D** — Smoke volumetric cut-planes
   - **Particles** — Lagrangian scatter and histograms
   - **Isosurfaces** — 2D projections
   - **Evacuation** — Agent floor plans and metrics
   - **Performance** — CPU and timestep data
4. Use the **Download HD PNG (300 DPI)** buttons to export publication-quality images
5. In Multi-Timestep mode (Slices/Boundaries), use **Download Separate PNGs (ZIP)** to get one PNG per timestep with a consistent colour scale across all frames

## Project Structure

```
FDSReader_Webapp/
├── app.py                          # Flask application (all API endpoints)
├── fds_utils.py                    # fdsreader data access + matplotlib rendering
├── requirements.txt                # Python dependencies
├── README.md
├── LICENSE
├── templates/
│   ├── base.html                   # Shared base template (navbar, layout)
│   ├── index.html                  # Landing / simulation loader page
│   ├── slice_viewer.html           # Slice data (SLCF) viewer
│   ├── boundary_viewer.html        # Boundary data (BNDF) viewer
│   ├── device_viewer.html          # Device data (DEVC) viewer
│   ├── hrr_viewer.html             # Heat Release Rate viewer
│   ├── plot3d_viewer.html          # Plot3D volumetric data viewer
│   ├── smoke3d_viewer.html         # Smoke3D volumetric data viewer
│   ├── particle_viewer.html        # Particle (PART) data viewer
│   ├── isosurface_viewer.html      # Isosurface (ISOF) data viewer
│   ├── evac_viewer.html            # Evacuation (EVAC) data viewer
│   └── performance_viewer.html     # CPU & Steps performance viewer
└── static/
    ├── css/
    │   └── style.css               # Dark fire-engineering theme
    └── js/
        ├── app.js                  # Shared frontend utilities
        └── viewer.js               # All viewer logic & download helpers
```

## API Endpoints

### Simulation & Navigation
| Endpoint | Method | Description |
|---|---|---|
| `/api/browse` | GET | Browse directories for simulation folder |
| `/api/load` | POST | Load simulation from directory path |
| `/api/colormaps` | GET | List available colormaps |

### Slice Data (SLCF)
| Endpoint | Method | Description |
|---|---|---|
| `/api/slices` | GET | List slice metadata |
| `/api/slice/render` | POST | Render single slice at time t |
| `/api/slice/render_multi` | POST | Multi-time snapshot grid |
| `/api/slice/animation_frames` | POST | Generate animation frames |
| `/api/slice/profile` | POST | Extract 1D line profile |
| `/api/slice/timeseries` | POST | Point time-series from slice |

### Boundary Data (BNDF)
| Endpoint | Method | Description |
|---|---|---|
| `/api/obstructions` | GET | List obstructions with boundary data |
| `/api/boundary/render` | POST | Render single boundary at time t |
| `/api/boundary/render_multi` | POST | Multi-time boundary grid |
| `/api/boundary/animation_frames` | POST | Boundary animation frames |
| `/api/boundary/timeseries` | POST | Boundary center time-series |

### Device Data (DEVC)
| Endpoint | Method | Description |
|---|---|---|
| `/api/devices` | GET | List device metadata |
| `/api/device/render` | POST | Plot selected device time-series |
| `/api/device/compare` | POST | Dual-axis device comparison |

### HRR, Plot3D, Smoke3D, Particle, Isosurface, Evacuation, Performance
| Endpoint | Method | Description |
|---|---|---|
| `/api/hrr` | GET | HRR column metadata |
| `/api/hrr/render` | POST | Plot HRR columns |
| `/api/plot3d` | GET | Plot3D metadata |
| `/api/plot3d/render` | POST | Plot3D cut-plane render |
| `/api/smoke3d` | GET | Smoke3D metadata |
| `/api/smoke3d/render` | POST | Smoke3D cut-plane render |
| `/api/particles` | GET | Particle class metadata |
| `/api/particle/scatter` | POST | Particle 2D scatter render |
| `/api/particle/histogram` | POST | Particle quantity histogram |
| `/api/isosurfaces` | GET | Isosurface metadata |
| `/api/isosurface/render` | POST | Isosurface 2D projection |
| `/api/evacuation` | GET | Evacuation metadata |
| `/api/evacuation/floorplan` | POST | Evacuation floor plan render |
| `/api/evacuation/timeseries` | POST | Evacuation metric time-series |
| `/api/cpu` | GET | CPU performance metadata |
| `/api/cpu/render` | POST | Plot CPU columns |
| `/api/steps` | GET | Steps data metadata |
| `/api/steps/render` | POST | Plot Steps columns |

### Download / Export
| Endpoint | Method | Description |
|---|---|---|
| `/api/download/slice/gif` | POST | Download slice animation as GIF |
| `/api/download/boundary/gif` | POST | Download boundary animation as GIF |
| `/api/download/slice/multi_pngs` | POST | Download separate HD PNGs per timestep (ZIP) — shared colour scale |
| `/api/download/boundary/multi_pngs` | POST | Download separate HD PNGs per timestep (ZIP) — shared colour scale |
| `/api/download/png` | POST | Universal single-frame HD PNG (300 DPI) |
| `/api/download/png/sequence` | POST | Multi-timestep PNG sequence as ZIP |

## fdsreader Capabilities Covered

This application implements all core capabilities demonstrated in the [fdsreader tutorial](https://firedynamics.github.io/LectureFireSimulation/content/tools/03_analysis/02_fdsreader.html):

- `fdsreader.Simulation()` — loading simulation data
- `sim.devices` — DEVC data access and plotting
- `sim.hrr` — HRR file columns
- `sim.slices` — SLCF data with `filter_by_quantity`, `get_nearest`, `to_global`
- Slice sub-slice access per mesh
- Slice coordinate extraction via `return_coordinates=True`
- `sim.obstructions` — BNDF data with `get_global_boundary_data_arrays`
- `sim.data_3d` — Plot3D volumetric data with cut-plane extraction
- `sim.smoke_3d` — Smoke3D volumetric data with cut-plane extraction
- `sim.particles` — Lagrangian particle positions and quantity data
- `sim.isosurfaces` — Isosurface vertex extraction and 2D projection
- `sim.evacs` — Evacuation agent positions, deaths, FED metrics
- `sim.cpu` — CPU performance data
- `sim.steps` — Timestep data
- Multi-mesh handling and global array merging
- High-resolution PNG export with shared colour scales across timesteps

## Requirements

- Python 3.8+
- FDS simulation output data (completed simulation directory)

## Dependencies

| Package | Minimum Version | Purpose |
|---|---|---|
| Flask | 3.0.0 | Web framework |
| fdsreader | 1.9.0 | FDS data parsing |
| NumPy | 1.24.0 | Numerical computation |
| Matplotlib | 3.7.0 | Plot rendering (Agg backend) |
| Pillow | 9.0.0 | GIF export, image processing |
## Acknowledgments

This web application is built on top of the [`fdsreader`](https://github.com/FireDynamics/fdsreader) Python module for reading and processing FDS (Fire Dynamics Simulator) output data. We gratefully acknowledge the authors and maintainers of `fdsreader`:

- **Jan Vogelsang** — [j.vogelsang@fz-juelich.de](mailto:j.vogelsang@fz-juelich.de)
- **Prof. Dr. Lukas Arnold** — [l.arnold@fz-juelich.de](mailto:l.arnold@fz-juelich.de)

*Forschungszentrum Jülich GmbH*

For more information about `fdsreader`, see:
- GitHub: [https://github.com/FireDynamics/fdsreader](https://github.com/FireDynamics/fdsreader)
- Tutorial: [https://firedynamics.github.io/LectureFireSimulation/content/tools/03_analysis/02_fdsreader.html](https://firedynamics.github.io/LectureFireSimulation/content/tools/03_analysis/02_fdsreader.html)

