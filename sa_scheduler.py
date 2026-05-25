#!/usr/bin/env python3
"""
sa_scheduler.py
======================
Step 2: PFS観測スケジューリング シミュレーテッドアニーリング（SA）[Numba高速化版]
"""

import sys
import math
import json
import random
import argparse
import multiprocessing
import numpy as np
import matplotlib
matplotlib.use("Agg")  # ヘッドレス環境対応
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime, timedelta
from numba import njit
from obs_utils import load_config


OBSDIR = Path(__file__).parent
VIS_MAP_FILE = OBSDIR / "vis_map.npz"
OUTPUT_TXT = OBSDIR / "schedule_result.txt"
OUTPUT_JSON = OBSDIR / "schedule_result.json"
OUTPUT_PLOT = OBSDIR / "schedule_plot.png"

# ============================================================
# 設定読み込みと定数
# ============================================================
config = load_config()

# SA ハイパーパラメーター
T0 = config['scheduler']['sa_t0']
ALPHA = config['scheduler']['sa_alpha']
N_ITER = config['scheduler']['sa_iterations']
T_MIN = config['scheduler']['sa_t_min']

# スコア重み (Numbaコンパイラに定数として伝えるためグローバルに定義)
W_HARD = config['scheduler']['weight_hard']
W_SPLIT = config['scheduler']['weight_split']
W_PRIORITY_BASE = config['scheduler']['weight_priority_base']
W_TEFF = config['scheduler']['weight_teff']
W_CONN = config['scheduler']['weight_conn']
W_EMPTY = config['scheduler']['weight_empty']
W_SLEW = config['scheduler']['weight_slew']  # スルー時間に対する直接的なペナルティ
MAX_PRIORITY = config['scheduler']['max_priority']

# 望遠鏡スルー速度
SLEW_SPEED_AZ = config['slew']['speed_az']
SLEW_SPEED_EL = config['slew']['speed_el']
SLEW_SPEED_ROT = config['slew']['speed_rot']


# ============================================================
# データ読み込み
# ============================================================
def load_vis_map(filepath: Path) -> dict:
    if not filepath.exists():
        raise FileNotFoundError(
            f"{filepath} が見つかりません。\n"
            "先に make_visibility_map.py を実行してください。"
        )
    data = np.load(filepath, allow_pickle=True)
    return {k: data[k] for k in data.files}

# ============================================================
# JITコンパイル対象のヘルパー・スコア関数・SAエンジン
# ============================================================

@njit
def calculate_slew_time(alt1: float, az1: float, rot1: float, alt2: float, az2: float, rot2: float) -> float:
    az_diff = (az2 - az1 + 180.0) % 360.0 - 180.0
    alt_diff = alt2 - alt1
    rot_diff = rot2 - rot1
    return max(abs(az_diff) / SLEW_SPEED_AZ, abs(alt_diff) / SLEW_SPEED_EL, abs(rot_diff) / SLEW_SPEED_ROT)


@njit
def compute_co_components(
    co_observed_night: np.ndarray,
    co_indices: np.ndarray,
    co_adj_matrix: np.ndarray,
    d: int,
    visited: np.ndarray,
) -> int:
    for idx in co_indices:
        visited[idx] = False
        
    components_count = 0
    queue = np.zeros(len(co_indices), dtype=np.int32)
    
    for ti in co_indices:
        if co_observed_night[ti] <= d and not visited[ti]:
            components_count += 1
            head = 0
            tail = 0
            queue[tail] = ti
            tail += 1
            visited[ti] = True
            
            while head < tail:
                curr = queue[head]
                head += 1
                for nbr in co_indices:
                    if co_adj_matrix[curr, nbr] and co_observed_night[nbr] <= d and not visited[nbr]:
                        visited[nbr] = True
                        queue[tail] = nbr
                        tail += 1
    return components_count

@njit
def compute_score(
    schedule: np.ndarray,
    target_priority: np.ndarray,
    target_category_code: np.ndarray,
    slot_night_idx: np.ndarray,
    order_pairs_arr: np.ndarray,
    co_adj_matrix: np.ndarray,
    n_nights: int,
    fine_alt: np.ndarray,
    fine_az: np.ndarray,
    fine_rot: np.ndarray,
    fine_teff: np.ndarray,
    fine_night_minutes: np.ndarray,
    night_slots_start: np.ndarray,
    night_slots_end: np.ndarray,
    co_indices: np.ndarray,
) -> float:
    n_targets = len(target_priority)
    n_slots = len(schedule)
    
    start_slot = np.full(n_targets, -1, dtype=np.int32)
    slot_success = np.zeros(n_slots, dtype=np.bool_)
    
    hard_violations = 0
    teff_sum = 0.0
    total_slew_time = 0.0

    for d in range(n_nights):
        s_start = night_slots_start[d]
        s_end = night_slots_end[d]
        if s_start == s_end:
            continue

        accumulated_slew_delay = 0.0
        last_target = -1

        for s in range(s_start, s_end):
            k = s - s_start
            ti = schedule[s]
            if ti < 0:
                accumulated_slew_delay = max(0.0, accumulated_slew_delay - 1200.0)
                last_target = -1
                continue

            if last_target >= 0 and ti != last_target:
                m1 = (k - 1) * 20
                m2 = k * 20
                
                alt1 = fine_alt[last_target, d, m1]
                az1 = fine_az[last_target, d, m1]
                rot1 = fine_rot[last_target, d, m1]
                
                alt2 = fine_alt[ti, d, m2]
                az2 = fine_az[ti, d, m2]
                rot2 = fine_rot[ti, d, m2]
                
                az_diff = (az2 - az1 + 180.0) % 360.0 - 180.0
                alt_diff = alt2 - alt1
                rot_diff = rot2 - rot1
                t_slew = max(abs(az_diff) / 0.5, abs(alt_diff) / 0.5, abs(rot_diff) / 1.5)
                
                accumulated_slew_delay += t_slew
                total_slew_time += t_slew

            start_sec = k * 1200.0 + accumulated_slew_delay
            start_min = int(start_sec // 60)
            exp_end_min = int((start_sec + 900.0) // 60) # 15分露出

            night_len = fine_night_minutes[d]
            if exp_end_min >= night_len:
                # 夜の終了時間を超過
                hard_violations += 1
            else:
                # 露出時間帯での高度・ロテーター制約チェック (純粋Pythonループで早期リターン)
                is_visible = True
                
                # 高度チェック
                for m in range(start_min, exp_end_min + 1):
                    val = fine_alt[ti, d, m]
                    if val < 32.5 or val > 75.0:
                        is_visible = False
                        break
                
                if is_visible:
                    # ロテーターチェック
                    for m in range(start_min, exp_end_min + 1):
                        val = fine_rot[ti, d, m]
                        if val < -174.0 or val > 174.0:
                            is_visible = False
                            break
                            
                if is_visible:
                    # 180度またぎチェック
                    r_start = fine_rot[ti, d, start_min]
                    r_end = fine_rot[ti, d, exp_end_min]
                    if (r_start * r_end < 0) and (abs(r_start) + abs(r_end) > 180.0):
                        is_visible = False

                if is_visible:
                    # 正常観測: teff の平均
                    teff_sum_val = 0.0
                    for m in range(start_min, exp_end_min + 1):
                        teff_sum_val += fine_teff[ti, d, m]
                    teff_val = teff_sum_val / (exp_end_min - start_min + 1)
                    
                    teff_sum += teff_val
                    slot_success[s] = True
                    if start_slot[ti] < 0:
                        start_slot[ti] = s
                else:
                    hard_violations += 1

            last_target = ti

    # 2. 順序制約
    order_violations = 0
    for idx in range(len(order_pairs_arr)):
        ta = order_pairs_arr[idx, 0]
        tb = order_pairs_arr[idx, 1]
        tb_start = start_slot[tb]
        if tb_start >= 0:
            ta_start = start_slot[ta]
            if ta_start < 0 or ta_start >= tb_start:
                order_violations += 1

    # 3. GAの後にGEを観測しない制約
    ga_ge_violations = 0
    for d in range(n_nights):
        s_start = night_slots_start[d]
        s_end = night_slots_end[d]
        
        first_ga_slot = -1
        for s in range(s_start, s_end):
            ti = schedule[s]
            if ti >= 0 and target_category_code[ti] == 1:
                if slot_success[s]:
                    first_ga_slot = s
                    break
        if first_ga_slot >= 0:
            for s in range(first_ga_slot + 1, s_end):
                ti = schedule[s]
                if ti >= 0 and target_category_code[ti] == 2:
                    if slot_success[s]:
                        ga_ge_violations += 1

    score = -W_HARD * (hard_violations + order_violations + ga_ge_violations)

    # 4. 優先度ボーナスと分割ペナルティ
    split_penalties = 0
    observed_count = 0
    for ti in range(n_targets):
        prev_s = -1
        has_obs = False
        for s in range(n_slots):
            if schedule[s] == ti and slot_success[s]:
                if not has_obs:
                    has_obs = True
                    observed_count += 1
                    pri = target_priority[ti]
                    score += W_PRIORITY_BASE * (MAX_PRIORITY - pri + 1)
                
                if prev_s >= 0:
                    if (s - prev_s != 1) or (slot_night_idx[s] != slot_night_idx[prev_s]):
                        split_penalties += 1
                prev_s = s
                
    score -= W_SPLIT * split_penalties
    score += W_TEFF * teff_sum
    score -= W_SLEW * total_slew_time  # 合計スルー時間を減点

    # 空きスロットペナルティ
    empty_slots_count = 0
    for s in range(n_slots):
        if schedule[s] < 0:
            empty_slots_count += 1
    score -= W_EMPTY * empty_slots_count

    # 5. CO領域連結成分ペナルティ
    if W_CONN > 0.0 and len(co_indices) > 0:
        co_observed_night = np.full(n_targets, 999, dtype=np.int32)
        for s in range(n_slots):
            ti = schedule[s]
            if ti >= 0 and slot_success[s] and target_category_code[ti] == 0:
                night = slot_night_idx[s]
                if night < co_observed_night[ti]:
                    co_observed_night[ti] = night

        visited = np.zeros(n_targets, dtype=np.bool_)
        total_components = 0
        for d in range(n_nights):
            total_components += compute_co_components(co_observed_night, co_indices, co_adj_matrix, d, visited)
        score -= W_CONN * total_components

    return score

@njit
def greedy_initial_schedule(
    target_priority: np.ndarray,
    target_category_code: np.ndarray,
    target_n_slots: np.ndarray,
    slot_night_idx: np.ndarray,
    order_pairs_arr: np.ndarray,
    fine_alt: np.ndarray,
    fine_az: np.ndarray,
    fine_rot: np.ndarray,
    fine_night_minutes: np.ndarray,
    night_slots_start: np.ndarray,
    night_slots_end: np.ndarray,
    priority_order: np.ndarray,
) -> np.ndarray:
    n_targets = len(target_priority)
    n_slots = len(slot_night_idx)
    schedule = np.full(n_slots, -1, dtype=np.int32)
    assigned = np.zeros(n_targets, dtype=np.bool_)

    prior_matrix = np.zeros((n_targets, n_targets), dtype=np.bool_)
    for idx in range(len(order_pairs_arr)):
        ta = order_pairs_arr[idx, 0]
        tb = order_pairs_arr[idx, 1]
        prior_matrix[ta, tb] = True

    n_nights = len(fine_night_minutes)

    for d in range(n_nights):
        s_start = night_slots_start[d]
        s_end = night_slots_end[d]
        n_slots_in_night = s_end - s_start
        if n_slots_in_night <= 0:
            continue

        accumulated_slew_delay = 0.0
        last_target = -1

        k = 0
        while k < n_slots_in_night:
            placed = False
            for ti in priority_order:
                if assigned[ti]:
                    continue

                has_unassigned_prior = False
                for ta in range(n_targets):
                    if prior_matrix[ta, ti] and not assigned[ta]:
                        has_unassigned_prior = True
                        break
                if has_unassigned_prior:
                    continue

                L = target_n_slots[ti]
                if k + L > n_slots_in_night:
                    continue

                if target_category_code[ti] == 2:
                    has_ga = False
                    for prev_k in range(k):
                        prev_s = s_start + prev_k
                        prev_ti = schedule[prev_s]
                        if prev_ti >= 0 and target_category_code[prev_ti] == 1:
                            has_ga = True
                            break
                    if has_ga:
                        continue

                temp_delay = accumulated_slew_delay
                if last_target >= 0 and ti != last_target:
                    m1 = (k - 1) * 20
                    m2 = k * 20
                    alt1 = fine_alt[last_target, d, m1]
                    az1 = fine_az[last_target, d, m1]
                    rot1 = fine_rot[last_target, d, m1]

                    alt2 = fine_alt[ti, d, m2]
                    az2 = fine_az[ti, d, m2]
                    rot2 = fine_rot[ti, d, m2]

                    az_diff = (az2 - az1 + 180.0) % 360.0 - 180.0
                    alt_diff = alt2 - alt1
                    rot_diff = rot2 - rot1
                    t_slew = max(abs(az_diff) / 0.5, abs(alt_diff) / 0.5, abs(rot_diff) / 1.5)
                    temp_delay += t_slew

                start_sec = k * 1200.0 + temp_delay
                start_min = int(start_sec // 60)
                exp_end_min = int((start_sec + L * 1200.0 - 300.0) // 60)

                night_len = fine_night_minutes[d]
                if exp_end_min >= night_len:
                    continue

                is_visible = True
                for m in range(start_min, exp_end_min + 1):
                    val = fine_alt[ti, d, m]
                    if val < 32.5 or val > 75.0:
                        is_visible = False
                        break
                
                if is_visible:
                    for m in range(start_min, exp_end_min + 1):
                        val = fine_rot[ti, d, m]
                        if val < -174.0 or val > 174.0:
                            is_visible = False
                            break
                            
                if is_visible:
                    r_start = fine_rot[ti, d, start_min]
                    r_end = fine_rot[ti, d, exp_end_min]
                    if (r_start * r_end < 0) and (abs(r_start) + abs(r_end) > 180.0):
                        is_visible = False

                if is_visible:
                    for l_idx in range(L):
                        schedule[s_start + k + l_idx] = ti
                    assigned[ti] = True
                    accumulated_slew_delay = temp_delay
                    last_target = ti
                    k += L
                    placed = True
                    break

            if not placed:
                accumulated_slew_delay = max(0.0, accumulated_slew_delay - 1200.0)
                last_target = -1
                k += 1

    return schedule

@njit
def sa_optimize(
    schedule: np.ndarray,
    vis_map: np.ndarray,
    teff_map: np.ndarray,
    target_priority: np.ndarray,
    target_category_code: np.ndarray,
    target_n_slots: np.ndarray,
    slot_night_idx: np.ndarray,
    order_pairs_arr: np.ndarray,
    co_adj_matrix: np.ndarray,
    n_nights: int,
    fine_alt: np.ndarray,
    fine_az: np.ndarray,
    fine_rot: np.ndarray,
    fine_teff: np.ndarray,
    fine_night_minutes: np.ndarray,
    night_slots_start: np.ndarray,
    night_slots_end: np.ndarray,
    co_indices: np.ndarray,
    T0: float,
    ALPHA: float,
    N_ITER: int,
    T_MIN: float,
    seed: int,
    worker_id: int,
):
    np.random.seed(seed)
    n_slots = len(schedule)
    n_targets = len(target_priority)
    
    current_score = compute_score(
        schedule, target_priority, target_category_code, slot_night_idx, order_pairs_arr, co_adj_matrix, n_nights,
        fine_alt, fine_az, fine_rot, fine_teff, fine_night_minutes, night_slots_start, night_slots_end, co_indices
    )
    best_schedule = schedule.copy()
    best_score = current_score

    T = T0

    valid_blocks = np.zeros(n_slots, dtype=np.int32)
    valid_blocks_count = 0
    for s in range(n_slots - 1):
        if slot_night_idx[s] == slot_night_idx[s + 1]:
            valid_blocks[valid_blocks_count] = s
            valid_blocks_count += 1
            
    for iteration in range(1, N_ITER + 1):
        move_type = np.random.random()
        new_schedule = schedule.copy()
        
        if move_type < 0.05:
            s1 = np.random.randint(0, n_slots)
            s2 = np.random.randint(0, n_slots)
            if s1 != s2:
                new_schedule[s1], new_schedule[s2] = schedule[s2], schedule[s1]

        elif move_type < 0.30:
            if valid_blocks_count >= 2:
                b1_idx = np.random.randint(0, valid_blocks_count)
                b2_idx = np.random.randint(0, valid_blocks_count)
                if b1_idx != b2_idx:
                    b1 = valid_blocks[b1_idx]
                    b2 = valid_blocks[b2_idx]
                    if abs(b1 - b2) >= 2:
                        new_schedule[b1], new_schedule[b2] = schedule[b2], schedule[b1]
                        new_schedule[b1+1], new_schedule[b2+1] = schedule[b2+1], schedule[b1+1]

        elif move_type < 0.45:
            occupied = np.zeros(n_slots, dtype=np.int32)
            occ_count = 0
            for s in range(n_slots):
                if schedule[s] >= 0:
                    occupied[occ_count] = s
                    occ_count += 1
            if occ_count > 0:
                s_start = occupied[np.random.randint(0, occ_count)]
                ti = schedule[s_start]
                L = target_n_slots[ti]
                
                ti_slots = np.zeros(L, dtype=np.int32)
                ti_slots_count = 0
                for s in range(n_slots):
                    if schedule[s] == ti:
                        if ti_slots_count < L:
                            ti_slots[ti_slots_count] = s
                            ti_slots_count += 1
                            
                if ti_slots_count == L:
                    is_valid = True
                    for idx in range(L - 1):
                        if (ti_slots[idx+1] - ti_slots[idx] != 1) or (slot_night_idx[ti_slots[idx]] != slot_night_idx[ti_slots[idx+1]]):
                            is_valid = False
                            break
                    if is_valid:
                        night_idx = slot_night_idx[ti_slots[0]]
                        empty_blocks = np.zeros(n_slots, dtype=np.int32)
                        empty_count = 0
                        s_night_start = night_slots_start[night_idx]
                        s_night_end = night_slots_end[night_idx]
                        
                        for s in range(s_night_start, s_night_end - L + 1):
                            block_ok = True
                            for idx in range(L):
                                if schedule[s + idx] >= 0:
                                    block_ok = False
                                    break
                            if block_ok:
                                empty_blocks[empty_count] = s
                                empty_count += 1
                                    
                        if empty_count > 0:
                            new_start = empty_blocks[np.random.randint(0, empty_count)]
                            for s_idx in range(L):
                                new_schedule[ti_slots[s_idx]] = -1
                            for k in range(L):
                                new_schedule[new_start + k] = ti

        elif move_type < 0.60:
            observed_mask = np.zeros(n_targets, dtype=np.bool_)
            for s in range(n_slots):
                if schedule[s] >= 0:
                    observed_mask[schedule[s]] = True
                    
            observed = np.zeros(n_targets, dtype=np.int32)
            obs_count = 0
            for ti in range(n_targets):
                if observed_mask[ti]:
                    observed[obs_count] = ti
                    obs_count += 1
                    
            if obs_count > 0:
                ti_old = observed[np.random.randint(0, obs_count)]
                L = target_n_slots[ti_old]
                ti_old_slots = np.zeros(L, dtype=np.int32)
                ti_old_count = 0
                for s in range(n_slots):
                    if schedule[s] == ti_old:
                        if ti_old_count < L:
                            ti_old_slots[ti_old_count] = s
                            ti_old_count += 1
                            
                unobserved_valid = np.zeros(n_targets, dtype=np.int32)
                unobs_valid_count = 0
                for ti in range(n_targets):
                    if not observed_mask[ti] and target_n_slots[ti] == L:
                        is_ok = True
                        for idx in range(L):
                            s = ti_old_slots[idx]
                            if vis_map[ti, s] == 0:
                                is_ok = False
                                break
                        if is_ok:
                            unobserved_valid[unobs_valid_count] = ti
                            unobs_valid_count += 1
                            
                if unobs_valid_count > 0:
                    ti_new = unobserved_valid[np.random.randint(0, unobs_valid_count)]
                    for idx in range(L):
                        new_schedule[ti_old_slots[idx]] = ti_new

        elif move_type < 0.70:
            observed_mask = np.zeros(n_targets, dtype=np.bool_)
            for s in range(n_slots):
                if schedule[s] >= 0:
                    observed_mask[schedule[s]] = True
                    
            unobserved = np.zeros(n_targets, dtype=np.int32)
            unobs_count = 0
            for ti in range(n_targets):
                if not observed_mask[ti]:
                    unobserved[unobs_count] = ti
                    unobs_count += 1
                    
            if unobs_count > 0:
                ti_new = unobserved[np.random.randint(0, unobs_count)]
                L = target_n_slots[ti_new]
                
                valid_starts = np.zeros(n_slots, dtype=np.int32)
                valid_starts_count = 0
                for s in range(n_slots - L + 1):
                    is_ok = True
                    for k in range(L):
                        if schedule[s+k] >= 0:
                            is_ok = False
                            break
                    if is_ok:
                        if slot_night_idx[s] == slot_night_idx[s + L - 1]:
                            is_ok_vis = True
                            for k in range(L):
                                if vis_map[ti_new, s+k] == 0:
                                    is_ok_vis = False
                                    break
                            if is_ok_vis:
                                valid_starts[valid_starts_count] = s
                                valid_starts_count += 1
                                
                if valid_starts_count > 0:
                    s_start = valid_starts[np.random.randint(0, valid_starts_count)]
                    for k in range(L):
                        new_schedule[s_start + k] = ti_new

        elif move_type < 0.75:
            observed_mask = np.zeros(n_targets, dtype=np.bool_)
            for s in range(n_slots):
                if schedule[s] >= 0:
                    observed_mask[schedule[s]] = True
            observed = np.zeros(n_targets, dtype=np.int32)
            obs_count = 0
            for ti in range(n_targets):
                if observed_mask[ti]:
                    observed[obs_count] = ti
                    obs_count += 1
            if obs_count > 0:
                ti_old = observed[np.random.randint(0, obs_count)]
                for s in range(n_slots):
                    if schedule[s] == ti_old:
                        new_schedule[s] = -1

        elif move_type < 0.80:
            observed_mask = np.zeros(n_targets, dtype=np.bool_)
            for s in range(n_slots):
                if schedule[s] >= 0:
                    observed_mask[schedule[s]] = True
            observed_g = np.zeros(n_targets, dtype=np.int32)
            obs_g_count = 0
            for ti in range(n_targets):
                if observed_mask[ti] and target_n_slots[ti] == 2:
                    observed_g[obs_g_count] = ti
                    obs_g_count += 1
            if obs_g_count > 0:
                ti_old = observed_g[np.random.randint(0, obs_g_count)]
                ti_old_slots = np.zeros(2, dtype=np.int32)
                ti_old_count = 0
                for s in range(n_slots):
                    if schedule[s] == ti_old:
                        if ti_old_count < 2:
                            ti_old_slots[ti_old_count] = s
                            ti_old_count += 1
                if ti_old_count == 2:
                    s0 = ti_old_slots[0]
                    s1 = ti_old_slots[1]
                    unobserved_co = np.zeros(n_targets, dtype=np.int32)
                    unobs_co_count = 0
                    for ti in range(n_targets):
                        if not observed_mask[ti] and target_n_slots[ti] == 1:
                            unobserved_co[unobs_co_count] = ti
                            unobs_co_count += 1
                    if unobs_co_count >= 2:
                        valid_co0 = np.zeros(unobs_co_count, dtype=np.int32)
                        valid_co0_count = 0
                        valid_co1 = np.zeros(unobs_co_count, dtype=np.int32)
                        valid_co1_count = 0
                        for idx in range(unobs_co_count):
                            ti = unobserved_co[idx]
                            if vis_map[ti, s0] == 1:
                                valid_co0[valid_co0_count] = ti
                                valid_co0_count += 1
                            if vis_map[ti, s1] == 1:
                                valid_co1[valid_co1_count] = ti
                                valid_co1_count += 1
                        if valid_co0_count > 0 and valid_co1_count > 0:
                            co0 = valid_co0[np.random.randint(0, valid_co0_count)]
                            valid_co1_filtered = np.zeros(valid_co1_count, dtype=np.int32)
                            valid_co1_fil_count = 0
                            for idx in range(valid_co1_count):
                                ti = valid_co1[idx]
                                if ti != co0:
                                    valid_co1_filtered[valid_co1_fil_count] = ti
                                    valid_co1_fil_count += 1
                            if valid_co1_fil_count > 0:
                                co1 = valid_co1_filtered[np.random.randint(0, valid_co1_fil_count)]
                                new_schedule[s0] = co0
                                new_schedule[s1] = co1

        elif move_type < 0.85:
            co_pairs = np.zeros((n_slots, 3), dtype=np.int32)
            co_pairs_count = 0
            for s in range(n_slots - 1):
                if slot_night_idx[s] == slot_night_idx[s+1]:
                    ti0 = schedule[s]
                    ti1 = schedule[s+1]
                    if ti0 >= 0 and ti1 >= 0 and ti0 != ti1:
                        if target_n_slots[ti0] == 1 and target_n_slots[ti1] == 1:
                            co_pairs[co_pairs_count, 0] = s
                            co_pairs[co_pairs_count, 1] = ti0
                            co_pairs[co_pairs_count, 2] = ti1
                            co_pairs_count += 1
            if co_pairs_count > 0:
                pair_idx = np.random.randint(0, co_pairs_count)
                s = co_pairs[pair_idx, 0]
                observed_mask = np.zeros(n_targets, dtype=np.bool_)
                for slot in range(n_slots):
                    if schedule[slot] >= 0:
                        observed_mask[schedule[slot]] = True
                unobserved_g_valid = np.zeros(n_targets, dtype=np.int32)
                unobs_g_valid_count = 0
                for ti in range(n_targets):
                    if not observed_mask[ti] and target_n_slots[ti] == 2:
                        if vis_map[ti, s] == 1 and vis_map[ti, s+1] == 1:
                            unobserved_g_valid[unobs_g_valid_count] = ti
                            unobs_g_valid_count += 1
                if unobs_g_valid_count > 0:
                    ti_new = unobserved_g_valid[np.random.randint(0, unobs_g_valid_count)]
                    new_schedule[s] = ti_new
                    new_schedule[s+1] = ti_new

        elif move_type < 0.95:
            candidates = np.zeros((n_slots, 2), dtype=np.int32)
            cand_count = 0
            for s in range(n_slots - 1):
                if slot_night_idx[s] == slot_night_idx[s+1]:
                    ti0 = schedule[s]
                    ti1 = schedule[s+1]
                    if (ti0 >= 0 and target_n_slots[ti0] == 1 and ti1 < 0) or \
                       (ti1 >= 0 and target_n_slots[ti1] == 1 and ti0 < 0):
                        candidates[cand_count, 0] = s
                        candidates[cand_count, 1] = s+1
                        cand_count += 1
            if cand_count > 0:
                cand_idx = np.random.randint(0, cand_count)
                s0 = candidates[cand_idx, 0]
                s1 = candidates[cand_idx, 1]
                observed_mask = np.zeros(n_targets, dtype=np.bool_)
                for slot in range(n_slots):
                    if schedule[slot] >= 0:
                        observed_mask[schedule[slot]] = True
                unobserved_g_valid = np.zeros(n_targets, dtype=np.int32)
                unobs_g_valid_count = 0
                for ti in range(n_targets):
                    if not observed_mask[ti] and target_n_slots[ti] == 2:
                        if vis_map[ti, s0] == 1 and vis_map[ti, s1] == 1:
                            unobserved_g_valid[unobs_g_valid_count] = ti
                            unobs_g_valid_count += 1
                if unobs_g_valid_count > 0:
                    ti_new = unobserved_g_valid[np.random.randint(0, unobs_g_valid_count)]
                    new_schedule[s0] = ti_new
                    new_schedule[s1] = ti_new

        else:
            observed_mask = np.zeros(n_targets, dtype=np.bool_)
            for s in range(n_slots):
                if schedule[s] >= 0:
                    observed_mask[schedule[s]] = True
            observed_g = np.zeros(n_targets, dtype=np.int32)
            obs_g_count = 0
            for ti in range(n_targets):
                if observed_mask[ti] and target_n_slots[ti] == 2:
                    observed_g[obs_g_count] = ti
                    obs_g_count += 1
            if obs_g_count > 0:
                ti_old = observed_g[np.random.randint(0, obs_g_count)]
                ti_old_slots = np.zeros(2, dtype=np.int32)
                ti_old_count = 0
                for s in range(n_slots):
                    if schedule[s] == ti_old:
                        if ti_old_count < 2:
                            ti_old_slots[ti_old_count] = s
                            ti_old_count += 1
                if ti_old_count == 2:
                    s0 = ti_old_slots[0]
                    s1 = ti_old_slots[1]
                    unobserved_co = np.zeros(n_targets, dtype=np.int32)
                    unobs_co_count = 0
                    for ti in range(n_targets):
                        if not observed_mask[ti] and target_n_slots[ti] == 1:
                            unobserved_co[unobs_co_count] = ti
                            unobs_co_count += 1
                    put_in_first = np.random.random() < 0.5
                    if put_in_first:
                        valid_co = np.zeros(unobs_co_count, dtype=np.int32)
                        valid_co_count = 0
                        for idx in range(unobs_co_count):
                            ti = unobserved_co[idx]
                            if vis_map[ti, s0] == 1:
                                valid_co[valid_co_count] = ti
                                valid_co_count += 1
                        if valid_co_count > 0:
                            co = valid_co[np.random.randint(0, valid_co_count)]
                            new_schedule[s0] = co
                            new_schedule[s1] = -1
                    else:
                        valid_co = np.zeros(unobs_co_count, dtype=np.int32)
                        valid_co_count = 0
                        for idx in range(unobs_co_count):
                            ti = unobserved_co[idx]
                            if vis_map[ti, s1] == 1:
                                valid_co[valid_co_count] = ti
                                valid_co_count += 1
                        if valid_co_count > 0:
                            co = valid_co[np.random.randint(0, valid_co_count)]
                            new_schedule[s0] = -1
                            new_schedule[s1] = co

        new_score = compute_score(
            new_schedule, target_priority, target_category_code, slot_night_idx, order_pairs_arr, co_adj_matrix, n_nights,
            fine_alt, fine_az, fine_rot, fine_teff, fine_night_minutes, night_slots_start, night_slots_end, co_indices
        )
        delta = new_score - current_score

        if delta >= 0 or np.random.random() < math.exp(delta / T):
            schedule = new_schedule
            current_score = new_score

            if current_score > best_score:
                best_score = current_score
                best_schedule = schedule.copy()

        T = max(T * ALPHA, T_MIN)
        
        # JIT内での進捗表示
        if iteration % 200_000 == 0:
            if worker_id >= 0:
                print("  [Worker", worker_id, "] iter=", iteration, "  T=", T, "  score=", current_score, "  best=", best_score)
            else:
                print("  [Warmup] iter=", iteration, "  T=", T, "  score=", current_score, "  best=", best_score)
            
    return best_schedule, best_score

# ============================================================
# テキスト出力 (非JIT)
# ============================================================
def format_schedule_text(schedule: np.ndarray, data: dict) -> str:
    lines = []
    sep = "=" * 80

    lines.append(sep)
    lines.append("  PFS Observation Schedule  (SA Optimized, Dynamic Slew Time Sim)")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(sep)

    night_names = data["night_names"]
    night_n_slots = data["night_n_slots"]
    slot_times_iso = data["slot_times_iso"]
    target_codes = data["target_codes"]
    target_category = data["target_category"]
    target_priority = data["target_priority"]
    
    fine_alt = data["fine_alt"]
    fine_az = data["fine_az"]
    fine_rot = data["fine_rot"]
    fine_teff = data["fine_teff"]
    fine_night_minutes = data["fine_night_minutes"]

    slot_start = 0
    actual_teff_total = 0.0
    hard_violations_count = 0

    for ni, (night, n_slots_n) in enumerate(zip(night_names, night_n_slots)):
        lines.append(f"\n{'─'*80}")
        lines.append(f"  Night {ni+1:2d}: {night}")
        lines.append(f"{'─'*80}")
        col_hdr = f"  {'#':>3}  {'Time(HST)':>11}  {'Category':^8}  {'Priority':>8}  {'t_eff':>8}  {'Slew':>5}  {'Status':^8}  {'Target Code':<30}"
        lines.append(col_hdr)
        lines.append(f"  {'─'*3}  {'─'*11}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*5}  {'─'*8}  {'─'*30}")

        # Precompute slew times and accumulate slew delay
        slew_secs = np.zeros(n_slots_n, dtype=np.float64)
        accumulated_slew_delays = np.zeros(n_slots_n, dtype=np.float64)

        accum_delay = 0.0
        last_tgt = -1
        for s in range(n_slots_n):
            si = slot_start + s
            ti = schedule[si]
            if ti >= 0:
                if last_tgt >= 0 and ti != last_tgt:
                    m1 = (s - 1) * 20
                    m2 = s * 20
                    alt1 = fine_alt[last_tgt, ni, m1]
                    az1 = fine_az[last_tgt, ni, m1]
                    rot1 = fine_rot[last_tgt, ni, m1]

                    alt2 = fine_alt[ti, ni, m2]
                    az2 = fine_az[ti, ni, m2]
                    rot2 = fine_rot[ti, ni, m2]

                    slew_val = calculate_slew_time(alt1, az1, rot1, alt2, az2, rot2)
                    accum_delay += slew_val
                    slew_secs[s - 1] = slew_val
                accumulated_slew_delays[s] = accum_delay
                last_tgt = ti
            else:
                accum_delay = max(0.0, accum_delay - 1200.0)
                accumulated_slew_delays[s] = accum_delay
                last_tgt = -1

        for s in range(n_slots_n):
            si = slot_start + s
            ti = schedule[si]

            t_utc = datetime.strptime(str(slot_times_iso[si])[:16], "%Y-%m-%dT%H:%M")
            t_hst_base = t_utc - timedelta(hours=10)

            accumulated_slew_delay = accumulated_slew_delays[s]
            slew_sec = slew_secs[s]

            if ti < 0:
                time_str = t_hst_base.strftime("%H:%M")
                slew_str = f"{slew_sec:.0f}s" if slew_sec > 0 else "---"
                lines.append(
                    f"  {s+1:3d}  {time_str:>11}  {'---':^8}  {'---':>8}  {'---':>8}  {slew_str:>5}  {'---':^8}  {'(empty)':<30}"
                )
            else:
                actual_start_hst = t_hst_base + timedelta(seconds=float(accumulated_slew_delay))
                actual_end_hst = actual_start_hst + timedelta(minutes=15)
                time_str = f"{actual_start_hst.strftime('%H:%M')}-{actual_end_hst.strftime('%H:%M')}"

                start_sec = s * 1200.0 + accumulated_slew_delay
                start_min = int(start_sec // 60)
                exp_end_min = int((start_sec + 900.0) // 60)

                night_len = fine_night_minutes[ni]
                status = "OK"
                teff_val = 0.0

                if exp_end_min >= night_len:
                    status = "EXPIRED"
                    hard_violations_count += 1
                else:
                    alts = fine_alt[ti, ni, start_min : exp_end_min + 1]
                    rots = fine_rot[ti, ni, start_min : exp_end_min + 1]

                    is_visible_alt = np.all(alts >= 32.5) and np.all(alts <= 75.0)
                    is_visible_rot = np.all(rots >= -174.0) and np.all(rots <= 174.0)
                    r_start = rots[0]
                    r_end = rots[-1]
                    cross_180 = (r_start * r_end < 0) and (abs(r_start) + abs(r_end) > 180.0)

                    if not is_visible_alt:
                        status = "ALT_ERR"
                        hard_violations_count += 1
                    elif not is_visible_rot:
                        status = "ROT_ERR"
                        hard_violations_count += 1
                    elif cross_180:
                        status = "WRAP_ERR"
                        hard_violations_count += 1
                    else:
                        teff_val = np.mean(fine_teff[ti, ni, start_min : exp_end_min + 1])
                        actual_teff_total += teff_val

                code = str(target_codes[ti])
                cat = str(target_category[ti])
                pri = int(target_priority[ti])
                slew_str = f"{slew_sec:.0f}s" if slew_sec > 0 else "---"

                lines.append(
                    f"  {s+1:3d}  {time_str:>11}  {cat:^8}  {pri:>8}  {teff_val:>8.2f}  {slew_str:>5}  {status:^8}  {code:<30}"
                )

        slot_start += n_slots_n

    # サマリー
    lines.append(f"\n{sep}")
    lines.append("  SUMMARY")
    lines.append(sep)
    
    obs_ti_success = set()
    slot_start_temp = 0
    for n_idx, n_slots_n in enumerate(night_n_slots):
        accum_delay = 0.0
        last_tgt = -1
        for s in range(n_slots_n):
            si = slot_start_temp + s
            ti = schedule[si]
            if ti >= 0:
                if last_tgt >= 0 and ti != last_tgt:
                    m1 = (s - 1) * 20
                    m2 = s * 20
                    t_s = calculate_slew_time(
                        fine_alt[last_tgt, n_idx, m1], fine_az[last_tgt, n_idx, m1], fine_rot[last_tgt, n_idx, m1],
                        fine_alt[ti, n_idx, m2], fine_az[ti, n_idx, m2], fine_rot[ti, n_idx, m2]
                    )
                    accum_delay += t_s
                st_sec = s * 1200.0 + accum_delay
                st_min = int(st_sec // 60)
                ed_min = int((st_sec + 900.0) // 60)
                n_len = fine_night_minutes[n_idx]
                if ed_min < n_len:
                    alts = fine_alt[ti, n_idx, st_min : ed_min + 1]
                    rots = fine_rot[ti, n_idx, st_min : ed_min + 1]
                    is_visible_alt = np.all(alts >= 32.5) and np.all(alts <= 75.0)
                    is_visible_rot = np.all(rots >= -174.0) and np.all(rots <= 174.0)
                    r_start = rots[0]
                    r_end = rots[-1]
                    cross_180 = (r_start * r_end < 0) and (abs(r_start) + abs(r_end) > 180.0)
                    if is_visible_alt and is_visible_rot and not cross_180:
                        obs_ti_success.add(ti)
                last_tgt = ti
        slot_start_temp += n_slots_n

    total_slots = len(schedule)
    occupied = int(np.sum(schedule >= 0))

    lines.append(f"  Total slots       : {total_slots}")
    lines.append(f"  Occupied slots    : {occupied}  ({occupied/total_slots*100:.1f}%)")
    lines.append(f"  Empty slots       : {total_slots - occupied}")
    lines.append(f"  Hard Violations   : {hard_violations_count}")
    lines.append(f"  Targets observed  : {len(obs_ti_success)} / {len(target_codes)}")
    lines.append(f"  Total t_eff       : {actual_teff_total:.2f}")
    lines.append("")
    for cat in ["CO", "GA", "GE"]:
        cat_total = sum(1 for c in target_category if c == cat)
        cat_obs = sum(1 for ti in obs_ti_success if str(target_category[ti]) == cat)
        if cat_total:
            lines.append(f"    {cat}: {cat_obs:3d} / {cat_total:3d} targets observed ({cat_obs/cat_total*100:.1f}%)")

    lines.append(sep)
    return "\n".join(lines)

# ============================================================
# 可視化: ガントチャート (非JIT)
# ============================================================
def plot_schedule(
    schedule: np.ndarray,
    data: dict,
    score_history: list,
    output_path: Path,
) -> None:
    night_names = data["night_names"]
    night_n_slots = data["night_n_slots"]
    slot_times_iso = data["slot_times_iso"]
    target_codes = data["target_codes"]
    target_category = data["target_category"]
    target_priority = data["target_priority"]

    cat_colors = {"CO": "#4fc3f7", "GA": "#81c784", "GE": "#ffb74d"}
    pri_alpha = {1: 1.0, 2: 0.82, 3: 0.64, 4: 0.46}

    n_nights = len(night_names)

    fig = plt.figure(figsize=(18, 3.2 * n_nights + 3.5))
    fig.patch.set_facecolor("#0f0f1a")

    gs = fig.add_gridspec(
        n_nights + 1, 1,
        height_ratios=[1.0] * n_nights + [1.5],
        hspace=0.45,
    )

    slot_start = 0
    for ni, (night, n_slots_n) in enumerate(zip(night_names, night_n_slots)):
        ax = fig.add_subplot(gs[ni])
        ax.set_facecolor("#141428")

        ax.set_xlim(-0.5, n_slots_n - 0.5)
        ax.set_ylim(-0.6, 0.6)

        ax.set_ylabel(night, fontsize=9, color="#e0e0e0", fontweight="bold", rotation=0, labelpad=35, va="center")
        ax.set_xticks(range(n_slots_n))
        
        x_labels = []
        for s in range(n_slots_n):
            si = slot_start + s
            t_utc = datetime.strptime(str(slot_times_iso[si])[:16], "%Y-%m-%dT%H:%M")
            t_hst = t_utc - timedelta(hours=10)
            x_labels.append(t_hst.strftime("%H:%M"))
        
        ax.set_xticklabels(x_labels, fontsize=6.5, color="#8888aa", rotation=30)
        ax.set_yticks([])
        ax.spines[:].set_color("#222244")
        ax.spines[:].set_linewidth(0.8)

        night_sched = schedule[slot_start : slot_start + n_slots_n]
        occ = np.sum(night_sched >= 0)
        n_obs = len(set(int(x) for x in night_sched if x >= 0))

        # ガントバー
        si = 0
        while si < n_slots_n:
            ti = night_sched[si]
            if ti < 0:
                ax.barh(0, 1, left=si, height=0.55,
                        color="#2a2a3a", edgecolor="#1a1a28", linewidth=0.3)
                si += 1
                continue

            start_si = si
            while si < n_slots_n and night_sched[si] == ti:
                si += 1
            width = si - start_si

            cat = str(target_category[ti])
            pri = int(target_priority[ti])
            color = cat_colors.get(cat, "#aaaaaa")
            alpha = pri_alpha.get(pri, 0.5)

            ax.barh(0, width, left=start_si, height=0.55,
                    color=color, alpha=alpha,
                    edgecolor="white", linewidth=0.4)

            if width >= 1:
                code = str(target_codes[ti])
                parts = code.split("_")
                short = "_".join(parts[-3:]) if len(parts) >= 3 else code[-15:]
                ax.text(
                    start_si + width / 2, 0,
                    f"[P{pri}]{short[:14]}",
                    ha="center", va="center",
                    fontsize=5.5, color="black", fontweight="bold",
                )

        ax.text(
            n_slots_n - 0.5, 0.45,
            f"{n_obs} targets / {occ}/{n_slots_n} slots",
            ha="right", va="top", fontsize=7, color="#cccccc",
        )
        ax.grid(axis="x", color="#222244", linewidth=0.5, alpha=0.7)
        slot_start += n_slots_n

    # スコア収束グラフ
    ax_score = fig.add_subplot(gs[n_nights])
    ax_score.set_facecolor("#141428")
    if len(score_history) > 1:
        iters = [h[0] for h in score_history]
        scores = [h[1] for h in score_history]
        ax_score.plot(iters, scores, color="#7c4dff", linewidth=1.8, label="Best Score")
        ax_score.fill_between(iters, scores, alpha=0.15, color="#7c4dff")
        ax_score.set_xlabel("SA Iteration", fontsize=9, color="#aaaaaa")
        ax_score.set_ylabel("Score", fontsize=9, color="#aaaaaa")
        ax_score.set_title("SA Convergence", fontsize=10, color="#e0e0e0", pad=8)
        ax_score.tick_params(colors="#aaaaaa")
        ax_score.spines[:].set_color("#333355")
        ax_score.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="#e0e0e0")
        ax_score.grid(color="#222244", linewidth=0.5, alpha=0.7)

    patches = [
        mpatches.Patch(color=c, label=f"{cat}  (Priority →)")
        for cat, c in cat_colors.items()
    ]
    fig.legend(
        handles=patches, loc="upper right",
        fontsize=8, facecolor="#1a1a2e", edgecolor="#444466", labelcolor="#e0e0e0",
        bbox_to_anchor=(0.99, 0.99),
    )
    fig.suptitle(
        "PFS Observation Schedule — SA Optimized (Numba Speedup)\n"
        "(color=category, opacity=priority: darker=higher)",
        fontsize=13, fontweight="bold", color="#e0e0e0", y=1.005,
    )

    plt.savefig(output_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Schedule plot saved: {output_path}")
    plt.close()

# ============================================================
# マルチプロセス用ワーカー
# ============================================================
def worker_task(args):
    (
        worker_id, seed, schedule, vis_map, teff_map, target_priority, target_category_code, target_n_slots,
        slot_night_idx, order_pairs_arr, co_adj_matrix, n_nights, fine_alt, fine_az,
        fine_rot, fine_teff, fine_night_minutes, night_slots_start, night_slots_end, co_indices_arr,
        T0, ALPHA, N_ITER, T_MIN
    ) = args

    print(f"  [Worker {worker_id}] Starting SA with seed {seed}...")
    best_sched, best_score = sa_optimize(
        schedule, vis_map, teff_map, target_priority, target_category_code, target_n_slots, slot_night_idx,
        order_pairs_arr, co_adj_matrix, n_nights, fine_alt, fine_az, fine_rot, fine_teff,
        fine_night_minutes, night_slots_start, night_slots_end, co_indices_arr,
        T0, ALPHA, N_ITER, T_MIN, seed, worker_id
    )
    print(f"  [Worker {worker_id}] Finished. Score = {best_score}")
    return best_sched, best_score, seed


# ============================================================
# メイン
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="PFS SA Scheduler (Numba JIT Multiprocessing Version)")
    parser.add_argument("-j", "--jobs", type=int, default=4, help="Number of parallel jobs to run (default: 4)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument("--iter", type=int, default=N_ITER, help=f"Number of SA iterations (default: {N_ITER})")
    parser.add_argument("-v", "--vis-map", type=str, default=str(VIS_MAP_FILE), help=f"Path to input vis_map.npz (default: {VIS_MAP_FILE.name})")
    parser.add_argument("-o", "--output-txt", type=str, default=str(OUTPUT_TXT), help=f"Path to output schedule text (default: {OUTPUT_TXT.name})")
    parser.add_argument("--output-json", type=str, default=str(OUTPUT_JSON), help=f"Path to output schedule JSON (default: {OUTPUT_JSON.name})")
    parser.add_argument("--output-plot", type=str, default=str(OUTPUT_PLOT), help=f"Path to output schedule plot image (default: {OUTPUT_PLOT.name})")
    args_cli = parser.parse_args()

    vis_map_file = Path(args_cli.vis_map)
    output_txt = Path(args_cli.output_txt)
    output_json = Path(args_cli.output_json)
    output_plot = Path(args_cli.output_plot)

    if not vis_map_file.exists():
        print(f"Error: input file not found at {vis_map_file}")
        sys.exit(1)

    print("=" * 60)
    print("PFS SA Scheduler (Numba JIT Multiprocessing Version)")
    print(f"  Jobs: {args_cli.jobs}, Base Seed: {args_cli.seed}, Iterations: {args_cli.iter}")
    print(f"  Input NPZ: {vis_map_file}")
    print(f"  Output TXT: {output_txt}")
    print(f"  Output JSON: {output_json}")
    print(f"  Output Plot: {output_plot}")
    print("=" * 60)

    # 1. データ読み込み
    print(f"\n[1] Loading {vis_map_file}...")
    data = load_vis_map(vis_map_file)
    vis_map = data["vis_map"]
    teff_map = data["target_teff"]
    target_priority = data["target_priority"].astype(int)
    slot_night_idx = data["slot_night_idx"].astype(int)
    n_targets, n_slots = vis_map.shape
    print(f"  Targets: {n_targets},  Slots: {n_slots}")

    # 順序制約
    ra = data["target_ra"]
    dec = data["target_dec"]
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
    order_pairs_arr = np.array(order_pairs, dtype=np.int32) if len(order_pairs) > 0 else np.zeros((0, 2), dtype=np.int32)
    print(f"  Detected {len(order_pairs)} target pairs requiring order preservation.")

    # CO隣接関係
    target_category = data["target_category"]
    n_nights = len(data["night_names"])
    co_indices = [i for i in range(n_targets) if str(target_category[i]) == "CO"]
    co_indices_arr = np.array(co_indices, dtype=np.int32)
    
    co_neighbors = {i: [] for i in co_indices}
    
    import math
    def get_separation(ra1, dec1, ra2, dec2):
        r1, d1 = math.radians(ra1), math.radians(dec1)
        r2, d2 = math.radians(ra2), math.radians(dec2)
        val = math.sin(d1)*math.sin(d2) + math.cos(d1)*math.cos(d2)*math.cos(r1 - r2)
        val = min(1.0, max(-1.0, val))
        return math.degrees(math.acos(val))

    for i in co_indices:
        for j in co_indices:
            if i != j:
                if get_separation(ra[i], dec[i], ra[j], dec[j]) <= 1.4:
                    co_neighbors[i].append(j)
                    
    co_adj_matrix = np.zeros((n_targets, n_targets), dtype=np.bool_)
    for i in co_indices:
        for j in co_neighbors[i]:
            co_adj_matrix[i, j] = True
    print(f"  Precomputed adjacency list for CO targets. {len(co_indices)} targets found.")

    # カテゴリの整数エンコーディング (0=CO, 1=GA, 2=GE)
    target_category_code = np.zeros(n_targets, dtype=np.int32)
    for i in range(n_targets):
        cat = str(target_category[i])
        if cat == "CO":
            target_category_code[i] = 0
        elif cat == "GA":
            target_category_code[i] = 1
        elif cat == "GE":
            target_category_code[i] = 2

    # 各夜の開始と終了スロットインデックス
    night_slots_start = np.zeros(n_nights, dtype=np.int32)
    night_slots_end = np.zeros(n_nights, dtype=np.int32)
    for d in range(n_nights):
        slots = np.where(slot_night_idx == d)[0]
        if len(slots) > 0:
            night_slots_start[d] = slots[0]
            night_slots_end[d] = slots[-1] + 1

    # 優先度順ソート (Greedy用)
    priority_order = np.argsort(target_priority, kind="stable")

    # 3D 浮動小数点配列の確保
    fine_alt = data["fine_alt"].astype(np.float64)
    fine_az = data["fine_az"].astype(np.float64)
    fine_rot = data["fine_rot"].astype(np.float64)
    fine_teff = data["fine_teff"].astype(np.float64)
    fine_night_minutes = data["fine_night_minutes"].astype(np.int32)

    # target_n_slots の動的計算
    slot_sec = config['scheduling']['slot_duration_minutes'] * 60
    target_exptime = data["target_exptime"].astype(np.int32)
    target_n_slots = np.ceil(target_exptime / slot_sec).astype(np.int32)
    target_n_slots = np.maximum(target_n_slots, 1)

    # 2. 初期解（貪欲法 JIT版）
    print("\n[2] Greedy warm-start (JIT)...")
    schedule = greedy_initial_schedule(
        target_priority, target_category_code, target_n_slots, slot_night_idx, order_pairs_arr,
        fine_alt, fine_az, fine_rot, fine_night_minutes, night_slots_start, night_slots_end, priority_order
    )

    # 3. SA JITコンパイルの事前実行（親プロセスでコンパイルを完了させて子プロセスに継承させる）
    print("\n[3] Pre-compiling JIT functions (Warmup)...")
    print("  (Compiling JIT functions first time... please wait a moment)")
    sa_optimize(
        schedule, vis_map, teff_map, target_priority, target_category_code, target_n_slots, slot_night_idx, order_pairs_arr, co_adj_matrix, n_nights,
        fine_alt, fine_az, fine_rot, fine_teff, fine_night_minutes, night_slots_start, night_slots_end, co_indices_arr,
        T0, ALPHA, 1, T_MIN, 42, -1
    )
    print("  Compilation finished.")

    # 4. マルチプロセスアニーリング実行
    print(f"\n[4] Simulated Annealing (JIT Multiprocessing with {args_cli.jobs} jobs)...")
    
    worker_args = []
    for w_id in range(args_cli.jobs):
        w_seed = args_cli.seed + w_id
        worker_args.append((
            w_id, w_seed, schedule, vis_map, teff_map, target_priority, target_category_code, target_n_slots,
            slot_night_idx, order_pairs_arr, co_adj_matrix, n_nights, fine_alt, fine_az,
            fine_rot, fine_teff, fine_night_minutes, night_slots_start, night_slots_end, co_indices_arr,
            T0, ALPHA, args_cli.iter, T_MIN
        ))
    
    with multiprocessing.Pool(processes=args_cli.jobs) as pool:
        results = pool.map(worker_task, worker_args)
    
    # 最良の結果を選択
    best_schedule = None
    best_score = -9e18
    best_seed = -1
    best_worker = -1
    
    print("\nAll workers finished. Results:")
    for res_sched, res_score, res_seed in results:
        w_id = res_seed - args_cli.seed
        print(f"  [Worker {w_id}] Seed = {res_seed:4d}  Score = {res_score:.4f}")
        if res_score > best_score:
            best_score = res_score
            best_schedule = res_sched.copy()
            best_seed = res_seed
            best_worker = w_id
            
    print(f"\nSelected best result from [Worker {best_worker}] (Seed = {best_seed}) with Score = {best_score}")
    
    # スコア履歴のダミー（Numbaで可変長リスト出力を避けるため、簡易履歴を準備）
    score_history = [(0, best_score), (args_cli.iter, best_score)]

    # 4. テキスト出力
    print("\n[4] Writing text schedule...")
    text = format_schedule_text(best_schedule, data)
    with open(output_txt, "w") as f:
        f.write(text)
    print(text)
    print(f"  Saved: {output_txt}")

    # 5. JSON出力
    # 実際の teff 合計を計算
    actual_teff_total = 0.0
    slot_start_temp = 0
    night_n_slots = data["night_n_slots"]

    for n_idx, n_slots_n in enumerate(night_n_slots):
        accum_delay = 0.0
        last_tgt = -1
        for s in range(n_slots_n):
            si = slot_start_temp + s
            ti = best_schedule[si]
            if ti >= 0:
                if last_tgt >= 0 and ti != last_tgt:
                    m1 = (s - 1) * 20
                    m2 = s * 20
                    t_s = calculate_slew_time(
                        fine_alt[last_tgt, n_idx, m1], fine_az[last_tgt, n_idx, m1], fine_rot[last_tgt, n_idx, m1],
                        fine_alt[ti, n_idx, m2], fine_az[ti, n_idx, m2], fine_rot[ti, n_idx, m2]
                    )
                    accum_delay += t_s
                st_sec = s * 1200.0 + accum_delay
                st_min = int(st_sec // 60)
                ed_min = int((st_sec + 900.0) // 60)
                n_len = fine_night_minutes[n_idx]
                if ed_min < n_len:
                    alts = fine_alt[ti, n_idx, st_min : ed_min + 1]
                    rots = fine_rot[ti, n_idx, st_min : ed_min + 1]
                    is_visible_alt = np.all(alts >= 32.5) and np.all(alts <= 75.0)
                    is_visible_rot = np.all(rots >= -174.0) and np.all(rots <= 174.0)
                    r_start = rots[0]
                    r_end = rots[-1]
                    cross_180 = (r_start * r_end < 0) and (abs(r_start) + abs(r_end) > 180.0)
                    if is_visible_alt and is_visible_rot and not cross_180:
                        actual_teff_total += np.mean(fine_teff[ti, n_idx, st_min : ed_min + 1])
                last_tgt = ti
        slot_start_temp += n_slots_n

    result_json = {
        "score": float(best_score),
        "n_targets": n_targets,
        "n_slots": n_slots,
        "total_teff": float(actual_teff_total),
        "schedule": [int(ti) for ti in best_schedule],
        "target_codes": [str(c) for c in data["target_codes"]],
        "target_category": [str(c) for c in data["target_category"]],
        "target_priority": [int(p) for p in data["target_priority"]],
        "slot_times_iso": [str(t) for t in data["slot_times_iso"]],
        "score_history": score_history,
    }
    with open(output_json, "w") as f:
        json.dump(result_json, f, indent=2)
    print(f"  JSON saved: {output_json}")

    # 6. ガントチャート
    print("\n[5] Plotting schedule...")
    plot_schedule(best_schedule, data, score_history, output_plot)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
