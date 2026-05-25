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
./bin/python3 -m pip install numpy astropy astroplan numba matplotlib pyyaml reportlab pandas
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

## Core Components

- **`obs_config.yaml`**: The unified configuration file where you can adjust telescope details, twilight limits, program switching margins, scheduling slot durations, hardware limits, and SA scheduler weights.
- **`run_pipeline.py`**: The main automation wrapper. It automatically detects and calls the local virtual environment Python interpreter (`./bin/python3`) if available.
- **`get_obsdates.py`**: Queries the NAOJ schedule CGI and formats observing intervals.
- **`make_visibility_map.py`**: Builds the visibility maps (`vis_map.npz`) containing the target coordinates, visibility grid, and effective exposure factor.
- **`sa_scheduler.py`**: Evaluates and schedules targets using simulated annealing.
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
- **`scheduler`**: SA cooling schedule variables and scoring weights for scientific priority, constraints, slew times, and consecutive blocks.

---

## Output Files

Upon successful execution, the following files will be generated in the root directory:
- **`obsdates_<year><month>.txt`**: Target night boundaries and twilight times.
- **`vis_map.npz`**: Binary visibility map matrix.
- **`schedule_result.json` & `schedule_result.txt`**: Plaintext and structured optimized schedule.
- **`obsplan_<year><month>.<version>.pdf`**: The final landscape A4 PDF report.
