import os
import json
import csv
import shutil
from datetime import datetime, timedelta

def main():
    json_path = "schedule_result.json"
    co_csv_path = "pfs_designs/CO_summary_reconfigure.csv"
    ge_csv_path = "pfs_designs/GE_summary_reconfigure.csv"
    ga_csv_path = "pfs_designs/GA_summary_reconfigure.csv"
    
    # 1. Load schedule result
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found. Please run the scheduler first.")
        return
        
    print(f"Loading schedule from {json_path}...")
    with open(json_path, 'r') as f:
        sched_data = json.load(f)
        
    schedule = sched_data["schedule"]
    target_codes = sched_data["target_codes"]
    slot_times_iso = sched_data["slot_times_iso"]
    
    # Map target code to its first scheduled slot time (UTC)
    code_to_slot_time = {}
    for s_idx, ti in enumerate(schedule):
        if ti >= 0:
            code = target_codes[ti]
            if code not in code_to_slot_time:
                code_to_slot_time[code] = slot_times_iso[s_idx]
                
    print(f"Found {len(code_to_slot_time)} scheduled targets in the schedule.")

    # 2. Get last night of observation period and calculate target date (HST) for unscheduled targets
    night_dates = []
    for t_str in slot_times_iso:
        t_utc = datetime.strptime(t_str.split(".")[0], "%Y-%m-%dT%H:%M:%S")
        t_hst = t_utc - timedelta(hours=10)
        if t_hst.hour < 12:
            night_date = (t_hst - timedelta(days=1)).date()
        else:
            night_date = t_hst.date()
        night_dates.append(night_date)
    last_night = max(night_dates)
    target_date_hst = last_night + timedelta(days=1)
    print(f"Observation period last night (HST): {last_night}")
    print(f"Setting unscheduled targets to (HST): {target_date_hst}")

    # Helper to parse and map unscheduled target observation time to the target day
    def map_to_target_day(orig_obstime_str, has_z=False):
        clean_str = orig_obstime_str.split(".")[0].replace("Z", "")
        if has_z:
            dt_utc = datetime.strptime(clean_str, "%Y-%m-%dT%H:%M:%S")
            dt_hst = dt_utc - timedelta(hours=10)
        else:
            dt_hst = datetime.strptime(clean_str, "%Y-%m-%dT%H:%M:%S")

        # If observation time is past midnight (hour < 12), it belongs to the morning following target_date_hst
        if dt_hst.hour < 12:
            assigned_date = target_date_hst + timedelta(days=1)
        else:
            assigned_date = target_date_hst

        new_dt_hst = datetime(
            year=assigned_date.year,
            month=assigned_date.month,
            day=assigned_date.day,
            hour=dt_hst.hour,
            minute=dt_hst.minute,
            second=dt_hst.second
        )
        if has_z:
            new_dt_utc = new_dt_hst + timedelta(hours=10)
            return new_dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            return new_dt_hst.strftime("%Y-%m-%dT%H:%M:%S")

    # Helper to format scheduled time
    def format_scheduled_times(slot_time_str):
        # E.g. "2026-07-13T06:27:10.000"
        dt_utc = datetime.strptime(slot_time_str.split(".")[0], "%Y-%m-%dT%H:%M:%S")
        dt_local = dt_utc - timedelta(hours=10) # HST is UTC - 10 hours
        
        obstime_utc = dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        obstime = dt_local.strftime("%Y-%m-%dT%H:%M:%S")
        return obstime_utc, obstime

    # 2. Process CSV files
    for csv_path, is_co in [(co_csv_path, True), (ge_csv_path, False), (ga_csv_path, False)]:
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found. Skipping.")
            continue
            
        backup_path = csv_path + ".bak"
        print(f"Creating backup: {backup_path}")
        shutil.copy2(csv_path, backup_path)
        
        updated_rows = []
        header = None
        scheduled_count = 0
        unscheduled_count = 0
        
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            for row in reader:
                code = row["ppc_code"]
                
                # Check if target is scheduled
                if code in code_to_slot_time:
                    # Scheduled target
                    slot_time_str = code_to_slot_time[code]
                    obstime_utc, obstime = format_scheduled_times(slot_time_str)
                    row["ppc_obstime_utc"] = obstime_utc
                    row["ppc_obstime"] = obstime
                    scheduled_count += 1
                else:
                    # Unscheduled target -> Map observation time to the day after the last night of the period
                    row["ppc_obstime_utc"] = map_to_target_day(row["ppc_obstime_utc"], has_z=True)
                    row["ppc_obstime"] = map_to_target_day(row["ppc_obstime"], has_z=False)
                    unscheduled_count += 1
                    
                # For CO targets, ensure ppc_exptime = 900 and ppc_nframes = 2
                if is_co:
                    row["ppc_exptime"] = "900"
                    row["ppc_nframes"] = "2"
                    
                updated_rows.append(row)
                
        # Write back updated rows
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(updated_rows)
            
        print(f"Updated {csv_path}: {scheduled_count} scheduled, {unscheduled_count} unscheduled targets updated.")

if __name__ == "__main__":
    main()
