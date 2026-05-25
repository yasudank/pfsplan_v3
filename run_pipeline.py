#!/usr/bin/env python3
"""
run_pipeline.py
===============
Automates the full PFS scheduling and reporting pipeline in one shot.

Usage:
  ./run_pipeline.py <year> <month>
Example:
  ./run_pipeline.py 2026 May
"""

import argparse
import os
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="PFS Observation Scheduling & PDF Report Pipeline")
    parser.add_argument("year", type=int, help="Year of observing run (e.g. 2026)")
    parser.add_argument("month", type=str, help="Month name of observing run (e.g. May)")
    args = parser.parse_args()

    # Automatically detect local virtualenv python interpreter in the workspace
    local_python = os.path.join(os.getcwd(), "bin", "python3")
    if os.path.exists(local_python):
        python_exe = local_python
    else:
        python_exe = sys.executable

    year = args.year
    month = args.month
    obsdates_file = f"obsdates_{year}{month}.txt"

    print("=" * 60)
    print(f"Starting PFS pipeline for {year} {month}")
    print(f"Using python: {python_exe}")
    print("=" * 60)

    # 1. get_obsdates.py
    print(f"\n>>> [1/5] Running get_obsdates.py for {year} {month}...")
    cmd1 = [python_exe, "get_obsdates.py", str(year), month]
    res1 = subprocess.run(cmd1)
    if res1.returncode != 0:
        print("Error: get_obsdates.py failed.")
        sys.exit(1)
        
    if not os.path.exists(obsdates_file):
        print(f"Error: Expected file '{obsdates_file}' was not created.")
        sys.exit(1)

    # 2. make_visibility_map.py
    print(f"\n>>> [2/5] Running make_visibility_map.py using {obsdates_file}...")
    cmd2 = [python_exe, "make_visibility_map.py", "-o", obsdates_file]
    res2 = subprocess.run(cmd2)
    if res2.returncode != 0:
        print("Error: make_visibility_map.py failed.")
        sys.exit(1)

    # 3. sa_scheduler.py
    print("\n>>> [3/5] Running sa_scheduler.py (Simulated Annealing)...")
    cmd3 = [python_exe, "sa_scheduler.py", "-v", "vis_map.npz"]
    res3 = subprocess.run(cmd3)
    if res3.returncode != 0:
        print("Error: sa_scheduler.py failed.")
        sys.exit(1)

    # 4. plot_schedule_v3.py
    print(f"\n>>> [4/5] Running plot_schedule_v3.py with {obsdates_file}...")
    cmd4 = [python_exe, "plot_schedule_v3.py", "-o", obsdates_file]
    res4 = subprocess.run(cmd4)
    if res4.returncode != 0:
        print("Error: plot_schedule_v3.py failed.")
        sys.exit(1)

    # 5. create_pdf_report_v3.py
    print("\n>>> [5/5] Running create_pdf_report_v3.py to generate PDF report...")
    cmd5 = [python_exe, "create_pdf_report_v3.py", "-s", "schedule_result.json", "-v", "vis_map.npz"]
    res5 = subprocess.run(cmd5)
    if res5.returncode != 0:
        print("Error: create_pdf_report_v3.py failed.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("PFS Pipeline completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
