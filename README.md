# Subaru PFS Observation Scheduler & Reporting Pipeline

This project provides a complete automated pipeline to fetch observing allocations, pre-calculate target visibility grids, optimize target scheduling using Simulated Annealing, and compile the results into a high-quality PDF report with visualization plots.

## Overview

The scheduling pipeline consists of several key steps:
1. **Fetch & Parse Observing Dates**: Connects to the NAOJ telescope schedule, parses allocations for `"SSP PFS"`, calculates twilight times, and creates the observing night definitions.
2. **Pre-compute Visibility Map**: Uses astropy/astroplan to calculate the visibility of targets at 20-minute slots during observing nights.
3. **Simulated Annealing Optimization**: An extremely fast JIT-compiled (Numba) SA scheduler optimizes target allocations to maximize scientific priority and effective exposure times ($t_{\mathrm{eff}}$) while respecting physical telescope limits.
4. **Generate Visualization Plots**: Computes altitude-vs-time, rotator angle-vs-time, sky coverage maps, and statistics charts.
5. **Generate PDF Report**: Compiles a ReportLab PDF document presenting cover plots, statistics, sky coverage maps, and night-by-night detailed tabular schedules.

---

## Installation & Prerequisites

This pipeline is built for Python 3.12 and depends on standard scientific and astronomy libraries.

Within the virtual environment, install the required packages:
```bash
# Standard dependency packages
./bin/python3 -m pip install numpy astropy astroplan numba matplotlib pyyaml reportlab pandas tqdm
```

---

## Quick Start

You can run the entire pipeline from end-to-end with a single command by passing the year and month of the observing run:

```bash
# Automate the entire workflow
./run_pipeline.py 2026 May
```

This will run the following individual scripts sequentially:
1. `get_obsdates.py 2026 May`
2. `make_visibility_map.py -o obsdates_2026May.txt`
3. `sa_scheduler.py -v vis_map.npz`
4. `plot_schedule_v3.py -o obsdates_2026May.txt`
5. `create_pdf_report_v3.py -s schedule_result.json -v vis_map.npz`

---

## SA Scheduler Algorithm (`sa_scheduler.py`)

### Population-based Simulated Annealing

The scheduler uses a **Population-based SA** approach with multiple optimization strategies to find high-quality observation schedules with low variance across runs.

#### Execution Flow

```
Phase 1: N workers start from randomized greedy initial schedules
    ↓ SA optimization (iter_per_phase iterations each)
    ↓ Sort by score → top 50% = "elite"
Phase 2: Elite keep their schedules, others restart from elite solutions
    ↓ SA optimization (lower T0)
    ↓ Sort → select new elite
Phase 3-4: Repeat with progressively lower temperatures
    ↓
Final: Best schedule across all workers selected
```

- **Phase 1**: All workers independently build randomized greedy initial schedules (same-priority targets are shuffled) and run SA optimization at full temperature (`T0`).
- **Phase 2+**: The top `elite_fraction` (default 50%) of workers continue from their best schedules. The remaining workers restart from a random elite schedule. Temperature decreases each phase (`T0 × reheat_factor^(phase-1)`).
- This propagates good structural solutions while maintaining diversity through different random seeds.

#### Initial Schedule Construction (Randomized Greedy)

Each worker generates a unique initial schedule by:
1. Adding small noise (±0.4) to target priorities to randomize ordering within same-priority groups
2. Greedily assigning targets night-by-night in priority order
3. Checking visibility, rotator limits, ordering constraints, and GA/GE sequencing rules

#### SA Neighborhood Operations

The SA explorer uses the following move types with assigned probabilities:

| Probability | Move Type | Description |
|-------------|-----------|-------------|
| 4% | **Multi-night R&R** | Destroy 2-3 random nights, rebuild with greedy insertion |
| 3% | **Category R&R** | Remove all targets of a random category (CO/GA/GE), reinsert |
| 6% | **Single-night R&R** | Destroy 1 random night, rebuild with greedy insertion |
| 4% | **Random swap** | Swap two random slots |
| 18% | **Block swap** | Swap two consecutive 2-slot blocks |
| 13% | **Target relocation** | Move a target to a new valid position |
| 12% | **Target replacement** | Replace an observed target with an unobserved one of same slot size |
| 10% | **Insert unobserved** | Place an unobserved target into empty slots |
| 5% | **Remove target** | Remove a random observed target |
| 5% | **Replace G→2×CO** | Replace a 2-slot target with two 1-slot CO targets |
| 5% | **Replace 2×CO→G** | Replace two adjacent CO targets with a 2-slot target |
| 10% | **Insert G in gap** | Insert a 2-slot target into a CO+empty or empty+CO gap |
| 5% | **Replace G→CO split** | Replace a 2-slot target, put one CO in each original slot |

#### Reheating

Periodic temperature resets prevent the search from getting trapped in local optima:
- Every `sa_reheat_interval` iterations (default: 250,000), temperature is reset to `T0 × sa_reheat_factor` (default: 30% of T0)
- Reheating stops after `sa_reheat_cutoff` fraction (default: 80%) of total iterations to allow final convergence

#### Score Function

The objective function combines multiple weighted components:

| Component | Weight | Description |
|-----------|--------|-------------|
| Hard violations | -1,000,000 | Altitude/rotator limit violations, ordering violations |
| Priority bonus | +100 × (max_pri - pri + 1) | Reward for observing high-priority targets |
| t_eff sum | +100 × Σt_eff | Reward for effective exposure time quality |
| Split penalty | -1,000 | Penalty for non-contiguous observation of same target |
| Empty slots | -5,000 | Penalty per unused slot |
| Slew time | -5 × total_slew | Penalty for telescope slew overhead |
| Overtime | -1,000 × minutes | Penalty for exceeding night length |

### Usage

```bash
# Basic run (4 jobs, 4 seeds, 1M iterations)
bin/python sa_scheduler.py

# Recommended production run (~80 seconds)
bin/python sa_scheduler.py -j 8 --total-seeds 16 --iter 4000000

# Larger run for maximum quality (~5 minutes)
bin/python sa_scheduler.py -j 8 --total-seeds 32 --iter 8000000

# Custom output paths
bin/python sa_scheduler.py -j 8 --total-seeds 16 --iter 4000000 \
  --output-seeds-csv results.csv \
  -o schedule.txt \
  --output-json schedule.json \
  --output-plot schedule.png
```

#### Command-line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-j`, `--jobs` | 4 | Number of parallel worker processes |
| `--seed` | 42 | Base random seed |
| `--total-seeds` | same as jobs | Total number of workers (across all phases) |
| `--iter` | 1,000,000 | Total SA iterations (split across phases) |
| `-v`, `--vis-map` | `vis_map.npz` | Input visibility map file |
| `-o`, `--output-txt` | `schedule_result.txt` | Output text schedule |
| `--output-json` | `schedule_result.json` | Output JSON schedule |
| `--output-plot` | `schedule_plot.png` | Output Gantt chart plot |
| `--output-seeds-csv` | `seeds_scores.csv` | CSV with per-phase per-seed scores |

### Performance Benchmarks

Comparison of old (independent seeds) vs new (population-based SA) approach:

| Metric | Old: 1024 seeds × 1M iter | New: 16 workers × 4M iter (4 phases) |
|--------|---------------------------|---------------------------------------|
| Best score | 18,986 | **19,199** (+1.1%) |
| Worst score | 8,838 | **18,296** |
| Score std | ~1,700 | **241** (86% reduction) |
| Wall time | Hours (1024 sequential) | **~80 seconds** |

---

## GUI Schedule Editor (`schedule_editor.py`)

A modern, web-based graphical interface for viewing, manually modifying, and validating observing schedules.

### Features
- **Interactive Drag/Swap Slots**: Select any time slot on the grid and swap it with another slot, empty it, or replace it with a scientific target from the catalog.
- **Real-time Hard Limit Validation**: When changes are made, the tool automatically re-evaluates all physical and scheduling constraints in the background:
  - *Altitude Limit*: 32.5° to 75.0°
  - *Rotator Limit*: -174° to 174° (and rotator 180° cross wrap checks)
  - *Overtime Limit*: Target observation ending time within night limits (+10m tolerance)
  - *Ordering Constraints*: Enforces required sequence of identical target coordinate steps
  - *GA-then-GE sequencing*: Prevents scheduling GE targets after GA targets on the same night
- **Live Score Tracking**: Displays the re-calculated objective score and total $t_{\mathrm{eff}}$ instantly on edit.
- **Automatic Backups**: Saving changes creates timestamped backups of `schedule_result.json` and `schedule_result.txt` in the `backups/` directory (e.g., `schedule_result_YYYYMMDD_HHMMSS.json`) before overwriting.

### Usage
Run the Flask-based utility local server:
```bash
# Start the server on a port of your choice (e.g., 8085)
bin/python schedule_editor.py --port 8085
```

Open `http://127.0.0.1:8085/` in your browser. (If using VS Code Remote-SSH, the port is forwarded automatically).

---

## Core Components

- **`obs_config.yaml`**: The unified configuration file where you can adjust telescope details, twilight limits, program switching margins, scheduling slot durations, hardware limits, and SA scheduler weights.
- **`run_pipeline.py`**: The main automation wrapper. It automatically detects and calls the local virtual environment Python interpreter (`./bin/python3`) if available.
- **`get_obsdates.py`**: Queries the NAOJ schedule CGI and formats observing intervals.
- **`make_visibility_map.py`**: Builds the visibility maps (`vis_map.npz`) containing the target coordinates, visibility grid, and effective exposure factor.
- **`sa_scheduler.py`**: Evaluates and schedules targets using population-based simulated annealing.
- **`plot_schedule_v3.py`**: Plots altitude profiles, rotator tracking, Mollweide sky coverage projection, and cumulative scheduling progress.
- **`create_pdf_report_v3.py`**: Reconstructs the schedule data dynamically on the fly to generate the PDF report, ensuring clean code without database or pandas dependency.
- **`obs_utils.py`**: Houses common utility functions like `load_config()`, `setup_observer()`, and time/target parsers.

---

## Configuration Settings (`obs_config.yaml`)

Adjust parameters globally under these main sections:
- **`location`**: Coordinates, timezone name, and UTC offset hours.
- **`twilight`**: Astronomical twilight horizon altitude (e.g. `-18.0` degrees).
- **`scheduling`**: Switching split margins (e.g. `5` minutes) and slot duration (`20` minutes).
- **`constraints`**: Hardware/physical safety limits (such as `max_airmass`, `max_altitude`, `min_altitude`, and `rotator_min/max`). Highlights violations in the generated PDF report.
- **`slew`**: Telescope azimuth, elevation, and instrument rotator speed limits.
- **`scheduler`**: SA hyperparameters and scoring weights:
  - `sa_t0`, `sa_alpha`, `sa_iterations`, `sa_t_min`: Core SA cooling schedule
  - `sa_reheat_interval`, `sa_reheat_factor`, `sa_reheat_cutoff`: Reheating parameters
  - `population_phases`, `elite_fraction`: Population-based SA parameters
  - `weight_*`: Scoring function weights

---

## Output Files

Upon successful execution, the following files will be generated in the root directory:
- **`obsdates_<year><month>.txt`**: Target night boundaries and twilight times.
- **`vis_map.npz`**: Binary visibility map matrix.
- **`schedule_result.json` & `schedule_result.txt`**: Plaintext and structured optimized schedule.
- **`seeds_scores.csv`**: Per-phase, per-seed score records for analysis.
- **`obsplan_<year><month>.<version>.pdf`**: The final landscape A4 PDF report.
