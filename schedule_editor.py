#!/usr/bin/env python3
"""
schedule_editor.py
======================
PFS観測スケジュール GUIエディタ＆バリデータ (Flaskベース)
"""

import os
import sys
import math
import json
import shutil
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from flask import Flask, jsonify, request, send_from_directory, render_template_string

# Import functions and variables from sa_scheduler
OBSDIR = Path(__file__).parent
sys.path.append(str(OBSDIR))
import sa_scheduler
from obs_utils import load_config

app = Flask(__name__)

CONFIG = load_config()
BACKUP_DIR = OBSDIR / "backups"
BACKUP_DIR.mkdir(exist_ok=True)

VIS_MAP_FILE = OBSDIR / "vis_map.npz"
SCHEDULE_JSON_FILE = OBSDIR / "schedule_result.json"
SCHEDULE_TXT_FILE = OBSDIR / "schedule_result.txt"

# Data cache
npz_data = None

def load_npz_data():
    global npz_data
    if npz_data is None:
        if not VIS_MAP_FILE.exists():
            raise FileNotFoundError(f"vis_map.npz が見つかりません。{VIS_MAP_FILE}")
        npz_data = np.load(VIS_MAP_FILE, allow_pickle=True)
    return npz_data

def get_order_pairs(ra, dec):
    coords = [(round(float(r), 5), round(float(d), 5)) for r, d in zip(ra, dec)]
    from collections import defaultdict
    coord_to_indices = defaultdict(list)
    for i, c in enumerate(coords):
        coord_to_indices[c].append(i)
    
    order_pairs = []
    for idxs in coord_to_indices.values():
        if len(idxs) > 1:
            for i in range(len(idxs) - 1):
                for j in range(i + 1, len(idxs)):
                    order_pairs.append((idxs[i], idxs[j]))
    return order_pairs

def validate_schedule_details(schedule_arr):
    data = load_npz_data()
    n_targets = len(data["target_priority"])
    n_slots = len(schedule_arr)
    n_nights = len(data["night_names"])
    
    fine_alt = data["fine_alt"]
    fine_az = data["fine_az"]
    fine_rot = data["fine_rot"]
    fine_night_minutes = data["fine_night_minutes"]
    slot_night_idx = data["slot_night_idx"].astype(int)
    
    # Get start/end slots for each night
    night_slots_start = np.zeros(n_nights, dtype=np.int32)
    night_slots_end = np.zeros(n_nights, dtype=np.int32)
    for d in range(n_nights):
        slots = np.where(slot_night_idx == d)[0]
        if len(slots) > 0:
            night_slots_start[d] = slots[0]
            night_slots_end[d] = slots[-1] + 1

    ra = data["target_ra"]
    dec = data["target_dec"]
    order_pairs = get_order_pairs(ra, dec)
    
    start_slot = np.full(n_targets, -1, dtype=np.int32)
    slot_success = np.zeros(n_slots, dtype=np.bool_)
    
    slot_errors = {s: [] for s in range(n_slots)}
    actual_times = {}
    
    # 1. Physical constraints check
    for d in range(n_nights):
        s_start = night_slots_start[d]
        s_end = night_slots_end[d]
        if s_start == s_end:
            continue

        # Get start time of the night (HST)
        t_utc = datetime.strptime(str(data["slot_times_iso"][s_start])[:16], "%Y-%m-%dT%H:%M")
        night_start_hst = t_utc - timedelta(hours=10)

        accumulated_slew_delay = 0.0
        last_target = -1

        for s in range(s_start, s_end):
            ti = schedule_arr[s]
            if ti < 0:
                accumulated_slew_delay = max(0.0, accumulated_slew_delay - 1200.0)
                last_target = -1
                
                # Store actual time for empty slot
                act_start_hst = night_start_hst + timedelta(seconds=accumulated_slew_delay + (s - s_start) * 1200.0)
                act_end_hst = act_start_hst + timedelta(minutes=20)
                actual_times[s] = {
                    "start": act_start_hst.strftime("%Y-%m-%dT%H:%M:%S"),
                    "end": act_end_hst.strftime("%Y-%m-%dT%H:%M:%S"),
                    "slew_delay": float(accumulated_slew_delay)
                }
                continue

            # Calculate Slew time
            slew_time = 0.0
            if last_target >= 0 and ti != last_target:
                k = s - s_start
                m1 = (k - 1) * 20
                m2 = k * 20
                
                alt1 = fine_alt[last_target, d, m1]
                az1 = fine_az[last_target, d, m1]
                rot1 = fine_rot[last_target, d, m1]
                
                alt2 = fine_alt[ti, d, m2]
                az2 = fine_az[ti, d, m2]
                rot2 = fine_rot[ti, d, m2]
                
                slew_time = sa_scheduler.calculate_slew_time(alt1, az1, rot1, alt2, az2, rot2)
                accumulated_slew_delay += slew_time

            # Store actual time for occupied slot
            act_start_hst = night_start_hst + timedelta(seconds=accumulated_slew_delay + (s - s_start) * 1200.0)
            act_end_hst = act_start_hst + timedelta(minutes=15)
            actual_times[s] = {
                "start": act_start_hst.strftime("%Y-%m-%dT%H:%M:%S"),
                "end": act_end_hst.strftime("%Y-%m-%dT%H:%M:%S"),
                "slew_delay": float(accumulated_slew_delay)
            }

            start_sec = (s - s_start) * 1200.0 + accumulated_slew_delay
            start_min = int(start_sec // 60)
            exp_end_min = int((start_sec + 900.0) // 60)

            night_len = fine_night_minutes[d]
            overtime = 0
            if exp_end_min >= night_len:
                overtime = exp_end_min - night_len + 1

            is_valid_slot = True
            
            # Overtime check
            if overtime > sa_scheduler.MAX_OVERTIME_MIN:
                end_time = night_start_hst + timedelta(minutes=exp_end_min)
                limit_time = night_start_hst + timedelta(minutes=night_len)
                slot_errors[s].append({
                    "type": "error",
                    "msg": f"Overtime limit exceeded: observation ends at {end_time.strftime('%H:%M')} ({exp_end_min}m) (night ends at {limit_time.strftime('%H:%M')} ({night_len}m) + max {sa_scheduler.MAX_OVERTIME_MIN}m limit)."
                })
                is_valid_slot = False
            else:
                # Altitude check
                for m in range(start_min, exp_end_min + 1):
                    m_clamped = min(m, night_len - 1)
                    val = fine_alt[ti, d, m_clamped]
                    if val < 32.5 or val > 75.0:
                        err_time = night_start_hst + timedelta(minutes=m)
                        time_str = err_time.strftime("%H:%M")
                        slot_errors[s].append({
                            "type": "error",
                            "msg": f"Altitude limit violated: reached {val:.1f}° at {time_str} ({m}m from night start) (must be between 32.5° and 75.0°)."
                        })
                        is_valid_slot = False
                        break
                
                # Rotator check
                if is_valid_slot:
                    for m in range(start_min, exp_end_min + 1):
                        m_clamped = min(m, night_len - 1)
                        val = fine_rot[ti, d, m_clamped]
                        if val < -174.0 or val > 174.0:
                            err_time = night_start_hst + timedelta(minutes=m)
                            time_str = err_time.strftime("%H:%M")
                            slot_errors[s].append({
                                "type": "error",
                                "msg": f"Rotator limit violated: reached {val:.1f}° at {time_str} ({m}m from night start) (must be between -174.0° and 174.0°)."
                            })
                            is_valid_slot = False
                            break
                            
                # Rotator Wrap check
                if is_valid_slot:
                    r_start = fine_rot[ti, d, min(start_min, night_len - 1)]
                    r_end = fine_rot[ti, d, min(exp_end_min, night_len - 1)]
                    if (r_start * r_end < 0) and (abs(r_start) + abs(r_end) > 180.0):
                        t_start = (night_start_hst + timedelta(minutes=start_min)).strftime("%H:%M")
                        t_end = (night_start_hst + timedelta(minutes=exp_end_min)).strftime("%H:%M")
                        slot_errors[s].append({
                            "type": "error",
                            "msg": f"Rotator wrapping error: crossed 180° boundary ({r_start:.1f}° to {r_end:.1f}°) between {t_start} and {t_end}."
                        })
                        is_valid_slot = False

            if is_valid_slot:
                slot_success[s] = True
                if start_slot[ti] < 0:
                    start_slot[ti] = s
            
            last_target = ti

    # 2. Ordering constraints check
    target_codes = data["target_codes"]
    for ta, tb in order_pairs:
        tb_start = start_slot[tb]
        if tb_start >= 0:
            ta_start = start_slot[ta]
            if ta_start < 0 or ta_start >= tb_start:
                # Add error to all slots occupied by tb
                for s in range(n_slots):
                    if schedule_arr[s] == tb:
                        slot_errors[s].append({
                            "type": "error",
                            "msg": f"Ordering error: target '{target_codes[tb]}' must be observed AFTER '{target_codes[ta]}' (index {ta})."
                        })

    # 3. GA-then-GE Night Sequencing check
    target_category_code = np.zeros(n_targets, dtype=np.int32)
    target_category = data["target_category"]
    for i in range(n_targets):
        cat = str(target_category[i])
        if cat == "CO":
            target_category_code[i] = 0
        elif cat == "GA":
            target_category_code[i] = 1
        elif cat == "GE":
            target_category_code[i] = 2

    for d in range(n_nights):
        s_start = night_slots_start[d]
        s_end = night_slots_end[d]
        
        first_ga_slot = -1
        for s in range(s_start, s_end):
            ti = schedule_arr[s]
            if ti >= 0 and target_category_code[ti] == 1 and slot_success[s]:
                first_ga_slot = s
                break
        
        if first_ga_slot >= 0:
            for s in range(first_ga_slot + 1, s_end):
                ti = schedule_arr[s]
                if ti >= 0 and target_category_code[ti] == 2 and slot_success[s]:
                    slot_errors[s].append({
                        "type": "error",
                        "msg": "Sequencing error: GE target cannot be observed after a GA target on the same night."
                    })

    # 4. Target required slots check (warning)
    slot_duration = CONFIG['scheduling']['slot_duration_minutes'] * 60
    target_exptime = data["target_exptime"].astype(np.int32)
    target_n_slots = np.ceil(target_exptime / slot_duration).astype(np.int32)
    target_n_slots = np.maximum(target_n_slots, 1)

    for ti in range(n_targets):
        slots_occupied = np.where(schedule_arr == ti)[0]
        if len(slots_occupied) > 0:
            required = target_n_slots[ti]
            is_consecutive = True
            for idx in range(len(slots_occupied) - 1):
                if slots_occupied[idx+1] - slots_occupied[idx] != 1 or slot_night_idx[slots_occupied[idx]] != slot_night_idx[slots_occupied[idx+1]]:
                    is_consecutive = False
                    break
            
            if len(slots_occupied) < required:
                for s in slots_occupied:
                    slot_errors[s].append({
                        "type": "warning",
                        "msg": f"Duration warning: target requires {required} slots, but only {len(slots_occupied)} slots are allocated."
                    })
            elif not is_consecutive:
                for s in slots_occupied:
                    slot_errors[s].append({
                        "type": "warning",
                        "msg": "Split warning: target observation is split across non-consecutive slots or nights."
                    })

    # 5. Recalculate score & teff
    co_indices = [i for i in range(n_targets) if str(target_category[i]) == "CO"]
    co_indices_arr = np.array(co_indices, dtype=np.int32)
    
    co_adj_matrix = np.zeros((n_targets, n_targets), dtype=np.bool_)
    for i in co_indices:
        for j in co_indices:
            if i != j:
                r1, d1 = math.radians(ra[i]), math.radians(dec[i])
                r2, d2 = math.radians(ra[j]), math.radians(dec[j])
                val = math.sin(d1)*math.sin(d2) + math.cos(d1)*math.cos(d2)*math.cos(r1 - r2)
                val = min(1.0, max(-1.0, val))
                if math.degrees(math.acos(val)) <= 1.4:
                    co_adj_matrix[i, j] = True

    score = sa_scheduler.compute_score(
        schedule_arr, data["target_priority"].astype(int), target_category_code,
        slot_night_idx, np.array(order_pairs, dtype=np.int32) if len(order_pairs) > 0 else np.zeros((0,2), dtype=np.int32),
        co_adj_matrix, n_nights,
        fine_alt, data["fine_az"], fine_rot, data["fine_teff"], fine_night_minutes,
        night_slots_start, night_slots_end, co_indices_arr
    )
    
    teff = sa_scheduler.calculate_schedule_teff(
        schedule_arr, n_nights, night_slots_start, night_slots_end,
        fine_alt, data["fine_az"], fine_rot, data["fine_teff"], fine_night_minutes
    )
    
    return {
        "slot_errors": slot_errors,
        "score": float(score),
        "total_teff": float(teff),
        "violations_count": sum(1 for errors in slot_errors.values() if any(err["type"] == "error" for err in errors)),
        "actual_times": actual_times
    }


# ============================================================
# API Routes
# ============================================================

@app.route('/')
def index():
    # editor.html をレンダリング
    try:
        with open(OBSDIR / 'templates' / 'editor.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        return render_template_string(html_content)
    except FileNotFoundError:
        return "editor.html is missing. Place it inside PFS/work/pfsplan_v3/templates/", 404

@app.route('/api/data')
def get_data():
    try:
        data = load_npz_data()
        
        # Load current schedule
        if not SCHEDULE_JSON_FILE.exists():
            return jsonify({"error": "schedule_result.json が見つかりません。先に最適化を実行してください。"}), 404
            
        with open(SCHEDULE_JSON_FILE, "r") as f:
            sched_json = json.load(f)
            
        schedule = sched_json["schedule"]
        
        # Pack catalog data
        target_n_slots = np.ceil(data["target_exptime"] / (CONFIG['scheduling']['slot_duration_minutes'] * 60)).astype(np.int32)
        target_n_slots = np.maximum(target_n_slots, 1)
        
        catalog = []
        for i in range(len(data["target_priority"])):
            catalog.append({
                "index": i,
                "code": str(data["target_codes"][i]),
                "category": str(data["target_category"][i]),
                "priority": int(data["target_priority"][i]),
                "duration_slots": int(target_n_slots[i]),
                "ra": float(data["target_ra"][i]),
                "dec": float(data["target_dec"][i])
            })
            
        slots = []
        slot_night_idx = data["slot_night_idx"].astype(int)
        for s in range(len(schedule)):
            slots.append({
                "slot_index": s,
                "night_idx": int(slot_night_idx[s]),
                "time_hst": str(data["slot_times_iso"][s])
            })
            
        response_data = {
            "catalog": catalog,
            "slots": slots,
            "schedule": schedule,
            "night_names": list(data["night_names"]),
            "night_n_slots": [int(x) for x in data["night_n_slots"]],
            "score": sched_json.get("score", 0),
            "total_teff": sched_json.get("total_teff", 0)
        }
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/validate', methods=['POST'])
def api_validate():
    try:
        req_data = request.get_json()
        schedule_arr = np.array(req_data["schedule"], dtype=np.int32)
        
        report = validate_schedule_details(schedule_arr)
        return jsonify(report)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/save', methods=['POST'])
def api_save():
    try:
        req_data = request.get_json()
        schedule_list = req_data["schedule"]
        schedule_arr = np.array(schedule_list, dtype=np.int32)
        
        # 1. バリデーション実行
        report = validate_schedule_details(schedule_arr)
        
        # 2. 前のバージョンのバックアップを作成 (JSON & TXT)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if SCHEDULE_JSON_FILE.exists():
            backup_json = BACKUP_DIR / f"schedule_result_{timestamp}.json"
            shutil.copy(SCHEDULE_JSON_FILE, backup_json)
            
        if SCHEDULE_TXT_FILE.exists():
            backup_txt = BACKUP_DIR / f"schedule_result_{timestamp}.txt"
            shutil.copy(SCHEDULE_TXT_FILE, backup_txt)

        # 3. JSON の上書き
        data = load_npz_data()
        result_json = {
            "score": float(report["score"]),
            "n_targets": len(data["target_priority"]),
            "n_slots": len(schedule_arr),
            "total_teff": float(report["total_teff"]),
            "schedule": [int(ti) for ti in schedule_arr],
            "target_codes": [str(c) for c in data["target_codes"]],
            "target_category": [str(c) for c in data["target_category"]],
            "target_priority": [int(p) for p in data["target_priority"]],
            "slot_times_iso": [str(t) for t in data["slot_times_iso"]],
            "score_history": [[0, float(report["score"])]] # ダミー
        }
        
        with open(SCHEDULE_JSON_FILE, "w") as f:
            json.dump(result_json, f, indent=2)
            
        # 4. TXT スケジュールの更新 (sa_scheduler のテキストフォーマッターを使用)
        text_content = sa_scheduler.format_schedule_text(schedule_arr, data)
        with open(SCHEDULE_TXT_FILE, "w") as f:
            f.write(text_content)
            
        return jsonify({
            "status": "success",
            "score": report["score"],
            "total_teff": report["total_teff"],
            "backup_json": backup_json.name if SCHEDULE_JSON_FILE.exists() else None,
            "backup_txt": backup_txt.name if SCHEDULE_TXT_FILE.exists() else None
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ============================================================
# メイン
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PFS Schedule GUI Editor Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to run Flask server (default: 8080)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to run (default: 127.0.0.1)")
    args = parser.parse_args()
    
    # 開始時に NPZ のロードとキャッシュを確認
    print("Loading NPZ visibility and metadata...")
    load_npz_data()
    print("NPZ data loaded successfully.")
    
    app.run(host=args.host, port=args.port, debug=True)
