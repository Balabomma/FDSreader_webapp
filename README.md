# FDSReader_webapp — Web Interface for fdsreader

A comprehensive Flask web application providing interactive visualization of FDS (Fire Dynamics Simulator) output data using the `fdsreader` Python module.

## Features

### Device Data (DEVC)
- Select and compare multiple device time-series
- Customizable time range filtering
- Automatic quantity/unit labelling

### Heat Release Rate (HRR)
- Plot HRR and energy balance columns (Q_RADI, Q_CONV, Q_COND, etc.)
- Customizable time range

### Slice Data (SLCF)
- **2D contour visualization** with time slider
- **Multi-time snapshot grids** (6-panel view)
- **Time-step animation** with play/pause/seek controls
- **Line profile extraction** along any spatial direction
- **Point time-series** at any (x,y,z) location
- **Multi-mesh support** via `to_global()` merging
- Customizable colormaps and value ranges

### Boundary Data (BNDF)
- **2D boundary field visualization** by obstruction
- **Orientation and quantity selection**
- **Time-step animation** with playback controls
- **Center-point time-series**
- Customizable colormaps and value ranges

### General
- Dark theme optimized for fire engineering analysis
- Keyboard shortcuts (Space: play/pause, Arrow keys: step)
- Simulation caching for fast re-access
- Responsive sidebar + main plot layout

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

1. Enter the path to your FDS simulation output directory (the folder containing `.smv` and output files)
2. Click **Load**
3. Use the sidebar tabs to navigate between DEVC, HRR, SLCF, and BNDF visualization

## Project Structure

```
fds-viewer/
├── app.py                  # Flask application (all API endpoints)
├── requirements.txt        # Python dependencies
├── README.md
├── templates/
│   └── index.html          # Single-page dashboard template
└── static/
    ├── css/
    │   └── style.css       # Dark fire-engineering theme
    └── js/
        └── app.js          # Frontend application logic
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/load` | POST | Load simulation from directory path |
| `/api/plot/devices` | POST | Plot selected DEVC time-series |
| `/api/plot/hrr` | POST | Plot HRR columns |
| `/api/plot/slice` | POST | Render 2D slice at time t |
| `/api/plot/slice/animate` | POST | Generate slice animation frames |
| `/api/plot/slice/multi` | POST | Multi-time snapshot grid |
| `/api/plot/slice/profile` | POST | Extract 1D line profile |
| `/api/plot/slice/timeseries` | POST | Point time-series from slice |
| `/api/plot/boundary` | POST | Render boundary field |
| `/api/plot/boundary/timeseries` | POST | Boundary center time-series |
| `/api/plot/boundary/animate` | POST | Boundary animation frames |
| `/api/colormaps` | GET | List available colormaps |

## fdsreader Capabilities Covered

This application implements all core capabilities demonstrated in the [fdsreader tutorial](https://firedynamics.github.io/LectureFireSimulation/content/tools/03_analysis/02_fdsreader.html):

- `fdsreader.Simulation()` — loading simulation data
- `sim.devices` — DEVC data access and plotting
- `sim.hrr` — HRR file columns
- `sim.slices` — SLCF data with `filter_by_quantity`, `get_nearest`, `to_global`
- Slice sub-slice access per mesh
- Slice coordinate extraction via `return_coordinates=True`
- `sim.obstructions` — BNDF data with `get_global_boundary_data_arrays`
- Multi-mesh handling and global array merging

## Requirements

- Python 3.8+
- FDS simulation output data (completed simulation directory)

