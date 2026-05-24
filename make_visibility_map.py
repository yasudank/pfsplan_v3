#!/usr/bin/env python3
"""
make_visibility_map.py
======================
Step 1: PFS観測スケジューリング 可視マップ生成スクリプト

Subaru望遠鏡の位置情報とAstropyを用いて、
各天体・各スロット（20分単位）の観測可否（高度 >= 30度）を計算し、
vis_map.npz として保存する。また可視マップのヒートマップを出力する。
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from pathlib import Path
from datetime import datetime, timedelta

from astropy.coordinates import EarthLocation, SkyCoord, AltAz, get_body
from astropy.time import Time
import astropy.units as u
from astropy.table import Table, vstack
from astroplan import Observer

# ============================================================
# 設定定数
# ============================================================

# Subaru望遠鏡の位置（Mauna Kea, Hawaii）
SUBARU = EarthLocation(
    lat=19.8258 * u.deg,
    lon=-155.4750 * u.deg,
    height=4139.0 * u.m,
)

HST_OFFSET_H = -10        # HST = UTC - 10h
MIN_ALT_DEG = 32.5        # 最低観測高度 [deg]
MAX_ALT_DEG = 75.0        # 最高観測高度 [deg]
MIN_ROT_DEG = -174.0      # 最小ローテーター回転角 [deg]
MAX_ROT_DEG = 174.0      # 最大ローテーター回転角 [deg]
SLOT_MINUTES = 20         # 1スロット = 20分
SUN_SET_ALT_DEG = -0.833  # 日没の定義（気差補正込み）

OBSDIR = Path(__file__).parent
TARGETS_DIR = OBSDIR / "targets"
OBSDATES_FILE = OBSDIR / "obsdates_2026May.txt"
OUTPUT_NPZ = OBSDIR / "vis_map.npz"
OUTPUT_PLOT = OBSDIR / "vis_map_plot.png"

# --- Copied from plan_observations_v2.py ---
class MoonBrightnessModel:
    def __init__(self):
        self.k = {
            "g": 0.15, "r": 0.10, "i": 0.09, "z": 0.08, "y": 0.1
        }
        self.Q = {
            "g": 0.129, "r": 0.152, "i": 0.114, "z": 0.048, "y": 0.038
        }
        self.mu_sky = {
            "g": 22.25, "r": 21.18, "i": 20.32, "z": 19.59, "y": 18.28
        }
        self.lam_eff = {
            "g": 478.0, "r": 617.0, "i": 766.0, "z": 888.0, "y": 974.0
        }
        self.Msun = {
            "g": -26.520, "r": -26.922, "i": -27.042, "z": -27.054, "y": -27.059, "V": -26.756
        }

    def X(self, z):
        z = np.radians(z)
        return 1.0 / np.sqrt(1.0 - 0.96 * np.sin(z)**2)

    def tR(self, band, X):
        p = 608.0  # Pressure at Mauna Kea (hPa)
        H = 4.2  # Height of Mauna Kea (km)
        lam = self.lam_eff[band] * 1.0E-03
        tauR = p / 1013.25 * (0.00864 + 6.5E-06 * H) * lam**(-(3.916 + 0.074 * lam + 0.050 / lam))
        return np.exp(-tauR * X)

    def tM(self, band, X):
        lam = self.lam_eff[band] * 1.0E-03
        alpha = -1.38
        kM = np.where(lam < 0.4, 0.050, 0.013 * lam**alpha)
        return 10.0**(-0.4 * kM * X)

    def Bmoon(self, band, alpha, z_moon, X_sky, rho):
        XV = self.Msun[band] - self.Msun["V"]
        phi = 180 - alpha
        Istar = 10.0**(-0.4 * (3.84 + 0.026 * np.abs(phi) + 4.0E-09 * phi**4)) * 10.0**(-0.4 * XV)
        X_moon = self.X(z_moon)
        rho = np.radians(rho)
        fR = 10.0**0.92 * (1.06 + np.cos(rho)**2)
        fM = 10.0**(2.44 - np.degrees(rho) / 40.0)
        BmoonR = fR * Istar * 10.0**(-0.4 * self.k[band] * X_moon) * (1.0 - self.tR(band, X_sky))
        BmoonM = fM * Istar * 10.0**(-0.4 * self.k[band] * X_moon) * (1.0 - self.tM(band, X_sky))
        return BmoonR + BmoonM

    def deltaMag(self, band, alpha, z_moon, z_sky, rho):
        if np.any(z_moon < 90.0):
            X_sky = self.X(z_sky)
            B0 = self.Q[band] * 5.48E+06 * 10.0**(-0.4 * self.mu_sky[band]) * X_sky
            Bm = self.Bmoon(band, alpha, z_moon, X_sky, rho)
            return -2.5 * np.log10((Bm + B0) / B0)
        else:
            return np.zeros_like(z_moon)


def calculate_teff(observer, target_coord, target_alt, target_airmass, moon_coord, moon_altaz, moon_phase_deg, mbm):
    observer_lat = observer.location.lat.deg
    target_dec = target_coord.dec.deg
    
    zmin = abs(target_dec - observer_lat)
    if zmin > 89.9: zmin = 89.9
            
    airmass0 = mbm.X(zmin)
    if airmass0 > 100: airmass0 = 100
    teff0 = 1.0 / (airmass0 * 10**(0.8*mbm.k['r']*(airmass0-1.0)))
    if teff0 == 0: return 0

    airmass = target_airmass    
    z_obs = 90. - target_alt
    z_moon = 90. - moon_altaz.alt.deg
    
    if z_moon >= 90.:
        dmu = 0.0
    else:
        moon_sep = moon_coord.separation(target_coord).deg
        dmu = mbm.deltaMag("r",
                           moon_phase_deg,
                           z_moon,
                           z_obs,
                           moon_sep)

    teff_abs = (1.0 / (10**(-0.4*dmu) * airmass * 10**(0.8*mbm.k['r']*(airmass-1.0))))
    
    return teff_abs / teff0

# カテゴリ別表示色
CAT_COLORS = {"CO": "#4fc3f7", "GA": "#81c784", "GE": "#ffb74d"}


# ============================================================
# ユーティリティ関数
# ============================================================

def parse_hst_time_str(date_str: str, time_str: str) -> datetime:
    """
    日付文字列(YYYY-MM-DD)と時刻文字列(HH:MM, 例: 28:10)を
    HST datetimeとして返す。28:10 は翌日 04:10 として処理。
    """
    base = datetime.strptime(date_str, "%Y-%m-%d")
    h, m = map(int, time_str.strip().split(":"))
    extra_days, h = divmod(h, 24)
    return base + timedelta(days=extra_days, hours=h, minutes=m)


def hst_to_utc(dt_hst: datetime) -> datetime:
    """HST → UTC変換"""
    return dt_hst + timedelta(hours=(-HST_OFFSET_H))


def utc_to_hst(dt_utc: datetime) -> datetime:
    """UTC → HST変換"""
    return dt_utc + timedelta(hours=HST_OFFSET_H)


def calc_sunset_hst(date_str: str) -> datetime:
    """
    指定日(HST)のSubaruにおける天文日没時刻(HST)を計算する。
    太陽高度が SUN_SET_ALT_DEG を下回る最初の時刻を返す。
    """
    base = datetime.strptime(date_str, "%Y-%m-%d")
    # 17:00〜21:00 HSTの範囲を1分刻みでサンプリング
    search_start_hst = base + timedelta(hours=17)
    times_hst = [search_start_hst + timedelta(minutes=m) for m in range(0, 5 * 60)]
    times_utc = [hst_to_utc(t) for t in times_hst]

    astropy_times = Time(
        [t.strftime("%Y-%m-%dT%H:%M:%S") for t in times_utc],
        format="isot",
        scale="utc",
    )
    altaz_frame = AltAz(obstime=astropy_times, location=SUBARU)
    sun_coord = get_body("sun", astropy_times, SUBARU)
    sun_alts = sun_coord.transform_to(altaz_frame).alt.deg

    for i in range(len(sun_alts) - 1):
        if sun_alts[i] >= SUN_SET_ALT_DEG > sun_alts[i + 1]:
            # 線形補間で精度を上げる
            frac = (sun_alts[i] - SUN_SET_ALT_DEG) / (sun_alts[i] - sun_alts[i + 1])
            sunset_hst = times_hst[i] + timedelta(minutes=frac)
            return sunset_hst

    # フォールバック（通常は起きない）
    print(f"  [WARNING] sunset not found for {date_str}, using 19:00 HST")
    return base + timedelta(hours=19)


# ============================================================
# 観測夜の読み込み
# ============================================================

def load_obsdates(filepath: Path) -> list[dict]:
    """
    観測夜ファイルを読み込み、各夜のスロット時刻リスト(UTC)を生成する。

    Returns:
        list of dict with keys:
            night       : 日付文字列 (YYYY-MM-DD, HST)
            start_hst   : 観測開始(HST datetime)
            end_hst     : 観測終了(HST datetime)
            slots_utc   : スロット開始時刻リスト(Astropy Time, UTC)
            n_slots     : スロット数
    """
    nights = []
    with open(filepath) as f:
        lines = [
            l.strip()
            for l in f
            if l.strip() and not l.strip().startswith("date")
        ]

    print(f"Loading observation dates from: {filepath}")
    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        date_str, start_str, end_str = parts[0], parts[1], parts[2]

        # 開始時刻
        if start_str.lower() == "sun_set":
            print(f"  Calculating sunset for {date_str}...", end=" ")
            start_hst = calc_sunset_hst(date_str)
            # 日没後5分のバッファを取る（薄明終了の目安として最低限）
            start_hst += timedelta(minutes=5)
            print(f"=> {start_hst.strftime('%H:%M')} HST")
        else:
            start_hst = parse_hst_time_str(date_str, start_str)

        # 終了時刻
        end_hst = parse_hst_time_str(date_str, end_str)

        # スロット生成（スロット開始時刻のみ記録）
        slots_utc = []
        t_hst = start_hst
        while t_hst + timedelta(minutes=SLOT_MINUTES) <= end_hst:
            slots_utc.append(
                Time(
                    hst_to_utc(t_hst).strftime("%Y-%m-%dT%H:%M:%S"),
                    format="isot",
                    scale="utc",
                )
            )
            t_hst += timedelta(minutes=SLOT_MINUTES)

        n_slots = len(slots_utc)
        print(
            f"  Night {date_str}: {n_slots} slots  "
            f"({start_hst.strftime('%H:%M')} – {end_hst.strftime('%H:%M')} HST)"
        )
        nights.append(
            {
                "night": date_str,
                "start_hst": start_hst,
                "end_hst": end_hst,
                "slots_utc": slots_utc,
                "n_slots": n_slots,
            }
        )

    total = sum(n["n_slots"] for n in nights)
    print(f"Total: {len(nights)} nights, {total} slots\n")
    return nights


# ============================================================
# 天体カタログの読み込み
# ============================================================

def load_targets(targets_dir: Path) -> Table:
    """
    targets/ 以下の全カテゴリ(CO/GA/GE)の ppcList.ecsv を読み込み、
    'category' カラムを追加して統合テーブルを返す。
    """
    tables = []
    for cat_dir in sorted(targets_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        ecsv_file = cat_dir / "ppcList.ecsv"
        if not ecsv_file.exists():
            continue
        t = Table.read(ecsv_file)
        t["category"] = cat_dir.name
        tables.append(t)
        print(f"  [{cat_dir.name}] {len(t)} targets")

    if not tables:
        raise FileNotFoundError(f"No ppcList.ecsv found in {targets_dir}")

    combined = vstack(tables)
    # インデックスを振り直す
    combined["target_idx"] = np.arange(len(combined), dtype=np.int32)
    print(f"  Total: {len(combined)} targets\n")
    return combined


# ============================================================
# 可視マップ計算
# ============================================================

def compute_visibility_map(
    targets: Table, nights: list[dict]
) -> tuple[np.ndarray, np.ndarray, list, list, list, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    vis_map[target_idx][slot_idx] = 1 (可視) or 0 (不可) を計算する。
    また teff_map[target_idx][slot_idx] = teff値 も同時に計算する。
    さらに、slew time 動的シフトシミュレーション評価用に、1分刻みの微細テーブルを同時に計算する。

    Returns:
        vis_map         : shape (n_targets, n_slots), dtype int8
        teff_map        : shape (n_targets, n_slots), dtype float
        all_slots_utc   : 全スロット時刻リスト (Astropy Time)
        slot_night_idx  : 各スロットが属する夜のインデックス
        slot_within_night : 各スロットが属する夜内のインデックス
        fine_alt        : shape (n_targets, n_nights, max_minutes), dtype float32
        fine_az         : shape (n_targets, n_nights, max_minutes), dtype float32
        fine_rot        : shape (n_targets, n_nights, max_minutes), dtype float32
        fine_teff       : shape (n_targets, n_nights, max_minutes), dtype float32
        fine_night_minutes : shape (n_nights,), dtype int32
    """
    # 全スロットをフラット化
    all_slots_utc = []
    slot_night_idx = []
    slot_within_night = []
    for ni, night in enumerate(nights):
        for si, slot in enumerate(night["slots_utc"]):
            all_slots_utc.append(slot)
            slot_night_idx.append(ni)
            slot_within_night.append(si)

    n_targets = len(targets)
    n_slots = len(all_slots_utc)
    print(f"Computing visibility map & teff: {n_targets} targets × {n_slots} slots")

    from astropy.time import TimeDelta

    # Astropy Time配列を一括作成
    all_times_start = Time(all_slots_utc)
    all_times_end = all_times_start + TimeDelta(SLOT_MINUTES * 60, format="sec")

    # AltAzフレームを全スロット分まとめて作成（パフォーマンス最適化）
    altaz_frame_start = AltAz(obstime=all_times_start, location=SUBARU)
    altaz_frame_end = AltAz(obstime=all_times_end, location=SUBARU)

    # Observerの初期化
    observer = Observer(location=SUBARU)
    mbm = MoonBrightnessModel()

    # 各スロットの中間時刻（10分後）
    all_times_mid = all_times_start + TimeDelta(600, format="sec")

    # 太陽と月の座標（全スロット分一括）
    print("  Calculating moon and sun coordinates...")
    sun_coords = get_body("sun", all_times_mid, location=SUBARU)
    moon_coords = get_body("moon", all_times_mid, location=SUBARU)

    # 月の地平座標（中間時刻）
    altaz_frame_mid = AltAz(obstime=all_times_mid, location=SUBARU)
    moon_altazs = moon_coords.transform_to(altaz_frame_mid)

    # 月の離角 (separation)
    moon_phases = moon_coords.separation(sun_coords)

    vis_map = np.zeros((n_targets, n_slots), dtype=np.int8)
    teff_map = np.zeros((n_targets, n_slots), dtype=float)
    target_alt = np.zeros((n_targets, n_slots), dtype=float)
    target_az = np.zeros((n_targets, n_slots), dtype=float)
    target_rot = np.zeros((n_targets, n_slots), dtype=float)

    for ti in range(n_targets):
        coord = SkyCoord(
            ra=targets["ppc_ra"][ti] * u.deg,
            dec=targets["ppc_dec"][ti] * u.deg,
        )
        # スロット開始時の高度
        altaz_start = coord.transform_to(altaz_frame_start)
        # スロット終了時の高度
        altaz_end = coord.transform_to(altaz_frame_end)
        
        # 開始・終了の両方で 32.5 <= alt <= 75 を満たすか判定
        is_visible_alt_start = (altaz_start.alt.deg >= MIN_ALT_DEG) & (altaz_start.alt.deg <= MAX_ALT_DEG)
        is_visible_alt_end = (altaz_end.alt.deg >= MIN_ALT_DEG) & (altaz_end.alt.deg <= MAX_ALT_DEG)
        is_visible_alt = is_visible_alt_start & is_visible_alt_end

        # Rotation Angle条件 (pa + ppc_pa)
        # 各スロットの開始・終了時の parallactic angle (PA)
        pa_start_ang = observer.parallactic_angle(all_times_start, coord)
        pa_end_ang = observer.parallactic_angle(all_times_end, coord)
        
        pa_start = pa_start_ang.deg
        pa_end = pa_end_ang.deg
        
        ppc_pa = targets["ppc_pa"][ti]
        
        rot_start = pa_start + ppc_pa
        rot_end = pa_end + ppc_pa
        
        # 角度を [-180, 180] に丸める
        rot_start = (rot_start + 180) % 360 - 180
        rot_end = (rot_end + 180) % 360 - 180
        
        # 条件1: -174 <= rot <= 174
        is_visible_rot_start = (rot_start >= MIN_ROT_DEG) & (rot_start <= MAX_ROT_DEG)
        is_visible_rot_end = (rot_end >= MIN_ROT_DEG) & (rot_end <= MAX_ROT_DEG)
        is_visible_rot = is_visible_rot_start & is_visible_rot_end
        
        # 条件2: 途中で +/- 180 度をまたがない
        # 符号が異なり、且つ絶対値の和が 180 度より大きい場合は「またぐ」
        cross_180 = (rot_start * rot_end < 0) & (np.abs(rot_start) + np.abs(rot_end) > 180.0)
        is_visible_no_cross = ~cross_180
        
        # 可視性マスク
        vis_mask = is_visible_alt & is_visible_rot & is_visible_no_cross
        vis_map[ti, :] = vis_mask.astype(np.int8)

        # teffの計算
        altaz_mid = coord.transform_to(altaz_frame_mid)
        alts_mid = altaz_mid.alt.deg
        airmasses_mid = altaz_mid.secz

        for si in range(n_slots):
            if vis_mask[si]:
                teff_val = calculate_teff(
                    observer, coord, alts_mid[si], airmasses_mid[si],
                    moon_coords[si], moon_altazs[si], moon_phases[si].deg, mbm
                )
                teff_map[ti, si] = teff_val
            else:
                teff_map[ti, si] = 0.0

        if (ti + 1) % 20 == 0 or (ti + 1) == n_targets:
            frac = vis_map[ti].mean() * 100
            print(
                f"  [{ti+1:3d}/{n_targets}] {targets['ppc_code'][ti][:30]:<30s} "
                f"visible: {frac:.0f}%"
            )

    total_obs = vis_map.mean() * 100
    print(f"\nDone. Mean observability: {total_obs:.1f}%")

    # ----------------------------------------------------
    # 分刻みの 3D テーブル計算 (slew time 動的シフト評価用)
    # ----------------------------------------------------
    print("\n  Computing fine-grained (1-minute) tables for dynamic slew simulation...")
    
    n_nights = len(nights)
    fine_night_minutes = np.zeros(n_nights, dtype=np.int32)
    times_utc_list = []
    night_times_indices = [] # list of (ni, m)
    
    for ni, night in enumerate(nights):
        minutes = int((night["end_hst"] - night["start_hst"]).total_seconds() // 60)
        fine_night_minutes[ni] = minutes
        start_hst = night["start_hst"]
        for m in range(minutes):
            t_hst = start_hst + timedelta(minutes=m)
            times_utc_list.append(hst_to_utc(t_hst))
            night_times_indices.append((ni, m))
            
    max_minutes = int(np.max(fine_night_minutes))
    
    # 3D 配列の初期化
    fine_alt = np.zeros((n_targets, n_nights, max_minutes), dtype=np.float32)
    fine_az = np.zeros((n_targets, n_nights, max_minutes), dtype=np.float32)
    fine_rot = np.zeros((n_targets, n_nights, max_minutes), dtype=np.float32)
    fine_teff = np.zeros((n_targets, n_nights, max_minutes), dtype=np.float32)
    
    # Astropy Time の一括作成
    print("    Calculating fine-grained moon/sun coordinates...")
    all_fine_times = Time(
        [t.strftime("%Y-%m-%dT%H:%M:%S") for t in times_utc_list],
        format="isot",
        scale="utc"
    )
    all_fine_altaz_frame = AltAz(obstime=all_fine_times, location=SUBARU)
    
    fine_sun_coords = get_body("sun", all_fine_times, location=SUBARU)
    fine_moon_coords = get_body("moon", all_fine_times, location=SUBARU)
    fine_moon_altazs = fine_moon_coords.transform_to(all_fine_altaz_frame)
    fine_moon_phases = fine_moon_coords.separation(fine_sun_coords)
    
    # 月の高度
    fine_moon_alts = fine_moon_altazs.alt.deg
    
    # 天体ごとのループ
    for ti in range(n_targets):
        coord = SkyCoord(
            ra=targets["ppc_ra"][ti] * u.deg,
            dec=targets["ppc_dec"][ti] * u.deg,
        )
        fine_altaz = coord.transform_to(all_fine_altaz_frame)
        fine_alts = fine_altaz.alt.deg
        fine_azs = fine_altaz.az.deg
        fine_airmasses = fine_altaz.secz
        
        fine_pas_ang = observer.parallactic_angle(all_fine_times, coord)
        fine_pas = fine_pas_ang.deg
        
        ppc_pa = targets["ppc_pa"][ti]
        fine_rots = fine_pas + ppc_pa
        fine_rots = (fine_rots + 180) % 360 - 180
        
        # teff 計算を一括で行うためのベクトル化 (速度向上)
        observer_lat = observer.location.lat.deg
        target_dec = coord.dec.deg
        zmin = abs(target_dec - observer_lat)
        if zmin > 89.9: zmin = 89.9
        airmass0 = mbm.X(zmin)
        if airmass0 > 100: airmass0 = 100
        teff0 = 1.0 / (airmass0 * 10**(0.8*mbm.k['r']*(airmass0-1.0)))
        
        if teff0 > 0:
            z_obs = 90.0 - fine_alts
            z_moon = 90.0 - fine_moon_alts
            moon_sep = fine_moon_coords.separation(coord).deg
            
            dmu = np.zeros_like(z_moon)
            valid_moon_idx = z_moon < 90.0
            if np.any(valid_moon_idx):
                dmu[valid_moon_idx] = mbm.deltaMag(
                    "r",
                    fine_moon_phases.deg[valid_moon_idx],
                    z_moon[valid_moon_idx],
                    z_obs[valid_moon_idx],
                    moon_sep[valid_moon_idx]
                )
            
            teff_abs = (1.0 / (10**(-0.4*dmu) * fine_airmasses * 10**(0.8*mbm.k['r']*(fine_airmasses-1.0))))
            fine_teffs = teff_abs / teff0
            fine_teffs[fine_alts <= 0] = 0.0
        else:
            fine_teffs = np.zeros_like(fine_alts)
            
        # 3D配列に格納
        for idx, (ni, m) in enumerate(night_times_indices):
            fine_alt[ti, ni, m] = fine_alts[idx]
            fine_az[ti, ni, m] = fine_azs[idx]
            fine_rot[ti, ni, m] = fine_rots[idx]
            fine_teff[ti, ni, m] = fine_teffs[idx]
            
        if (ti + 1) % 20 == 0 or (ti + 1) == n_targets:
            print(f"    [{ti+1:3d}/{n_targets}] Calculated fine-grained data")

    print("  Fine-grained tables computed successfully.")
    return (
        vis_map, teff_map, all_slots_utc, slot_night_idx, slot_within_night,
        fine_alt, fine_az, fine_rot, fine_teff, fine_night_minutes
    )


# ============================================================
# 保存
# ============================================================

def save_vis_map(
    vis_map: np.ndarray,
    teff_map: np.ndarray,
    targets: Table,
    nights: list[dict],
    all_slots_utc: list,
    slot_night_idx: list,
    slot_within_night: list,
    fine_alt: np.ndarray,
    fine_az: np.ndarray,
    fine_rot: np.ndarray,
    fine_teff: np.ndarray,
    fine_night_minutes: np.ndarray,
    output_path: Path,
) -> None:
    """可視マップをnpz形式で保存"""
    np.savez_compressed(
        output_path,
        # 可視マップ本体
        vis_map=vis_map,
        target_teff=teff_map,
        # 天体情報
        target_codes=np.array(targets["ppc_code"], dtype=str),
        target_ra=np.array(targets["ppc_ra"], dtype=float),
        target_dec=np.array(targets["ppc_dec"], dtype=float),
        target_priority=np.array(targets["ppc_priority"], dtype=int),
        target_category=np.array(targets["category"], dtype=str),
        target_exptime=np.array(targets["ppc_exptime"], dtype=int),
        target_nframes=np.array(targets["ppc_nframes"], dtype=int),
        target_ppc_pa=np.array(targets["ppc_pa"], dtype=float),
        # スロット情報
        slot_times_iso=np.array([t.isot for t in all_slots_utc], dtype=str),
        slot_night_idx=np.array(slot_night_idx, dtype=int),
        slot_within_night=np.array(slot_within_night, dtype=int),
        # 夜情報
        night_names=np.array([n["night"] for n in nights], dtype=str),
        night_n_slots=np.array([n["n_slots"] for n in nights], dtype=int),
        night_start_hst=np.array(
            [n["start_hst"].strftime("%Y-%m-%dT%H:%M") for n in nights], dtype=str
        ),
        night_end_hst=np.array(
            [n["end_hst"].strftime("%Y-%m-%dT%H:%M") for n in nights], dtype=str
        ),
        # 微細シミュレーション用テーブル
        fine_alt=fine_alt,
        fine_az=fine_az,
        fine_rot=fine_rot,
        fine_teff=fine_teff,
        fine_night_minutes=fine_night_minutes,
    )
    print(f"Saved: {output_path}")


# ============================================================
# 可視化
# ============================================================

def plot_visibility_map(
    vis_map: np.ndarray,
    targets: Table,
    nights: list[dict],
    slot_night_idx: list,
    output_path: Path,
) -> None:
    """
    可視マップをヒートマップとして可視化する。
    行=天体（カテゴリ別に色分け）、列=スロット、夜の境界を赤線で表示。
    """
    n_targets, n_slots = vis_map.shape
    categories = list(targets["category"])

    # カテゴリ別に行を色付けするためのカラーマップを作成
    cat_list = ["CO", "GA", "GE"]
    cat_idx_map = {c: i for i, c in enumerate(cat_list)}

    fig, ax = plt.subplots(figsize=(max(14, n_slots * 0.06), max(8, n_targets * 0.18)))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    # 可視マップをカスタムカラーで描画
    # 不可視=暗い灰色、可視=カテゴリ色のグラデーション
    rgba = np.zeros((n_targets, n_slots, 4))
    for ti, cat in enumerate(categories):
        color_hex = CAT_COLORS.get(cat, "#aaaaaa")
        r, g, b = mcolors.to_rgb(color_hex)
        for si in range(n_slots):
            if vis_map[ti, si] == 1:
                rgba[ti, si] = [r, g, b, 0.85]
            else:
                rgba[ti, si] = [0.1, 0.1, 0.15, 1.0]

    ax.imshow(rgba, aspect="auto", origin="upper",
              extent=[-0.5, n_slots - 0.5, n_targets - 0.5, -0.5])

    # 夜の境界線
    night_boundaries = []
    cumulative = 0
    for ni, night in enumerate(nights):
        if ni > 0:
            ax.axvline(x=cumulative - 0.5, color="#ff6b6b", linewidth=1.5, alpha=0.9, zorder=3)
            night_boundaries.append(cumulative)
        mid = cumulative + night["n_slots"] / 2
        ax.text(mid, -1.2, night["night"][5:], ha="center", va="bottom",
                fontsize=8, color="#e0e0e0", fontweight="bold")
        cumulative += night["n_slots"]

    # カテゴリ区切り線
    current_cat = categories[0]
    for ti, cat in enumerate(categories):
        if cat != current_cat:
            ax.axhline(y=ti - 0.5, color="#ffeb3b", linewidth=1.0, alpha=0.6, zorder=3)
            current_cat = cat

    # Y軸ラベル
    ytick_labels = [
        f"[{cat}] {code[:28]}"
        for cat, code in zip(categories, targets["ppc_code"])
    ]
    ax.set_yticks(range(n_targets))
    ax.set_yticklabels(ytick_labels, fontsize=6, color="#e0e0e0")

    ax.set_xlabel("Slot index (20 min/slot)", fontsize=10, color="#e0e0e0")
    ax.tick_params(axis="x", colors="#aaaaaa")
    ax.spines[:].set_color("#333366")

    # 凡例
    patches = [
        mpatches.Patch(color=c, label=f"{cat} ({sum(1 for x in categories if x==cat)} targets)")
        for cat, c in CAT_COLORS.items()
        if cat in categories
    ]
    patches.append(mpatches.Patch(color="#1a1a30", label="Not observable"))
    ax.legend(handles=patches, loc="upper right", fontsize=8,
              facecolor="#1a1a2e", edgecolor="#444466", labelcolor="#e0e0e0")

    ax.set_title(
        f"PFS Visibility Map — {MIN_ALT_DEG}° ≤ Alt ≤ {MAX_ALT_DEG}° (Subaru, Mauna Kea)\n"
        f"{n_targets} targets × {n_slots} slots over {len(nights)} nights",
        fontsize=12,
        fontweight="bold",
        color="#e0e0e0",
        pad=20,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Plot saved: {output_path}")
    plt.close()


# ============================================================
# メイン
# ============================================================

def main():
    print("=" * 60)
    print("PFS Visibility Map Generator")
    print("=" * 60)

    # 1. 観測夜の読み込み
    print("\n[1] Loading observation dates...")
    nights = load_obsdates(OBSDATES_FILE)

    # 2. 天体カタログの読み込み
    print("[2] Loading target catalogs...")
    targets = load_targets(TARGETS_DIR)

    # 3. 可視マップの計算
    print("[3] Computing visibility map...")
    (
        vis_map, teff_map, all_slots_utc, slot_night_idx, slot_within_night,
        fine_alt, fine_az, fine_rot, fine_teff, fine_night_minutes
    ) = compute_visibility_map(targets, nights)

    # 4. 保存
    print("\n[4] Saving vis_map.npz...")
    save_vis_map(
        vis_map, teff_map, targets, nights, all_slots_utc,
        slot_night_idx, slot_within_night,
        fine_alt, fine_az, fine_rot, fine_teff, fine_night_minutes,
        OUTPUT_NPZ
    )

    # 5. 可視化
    print("\n[5] Plotting visibility map...")
    plot_visibility_map(vis_map, targets, nights, slot_night_idx, OUTPUT_PLOT)

    # サマリー
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_slots = sum(n["n_slots"] for n in nights)
    print(f"  Nights   : {len(nights)}")
    print(f"  Slots    : {total_slots}")
    print(f"  Targets  : {len(targets)}")
    for cat in ["CO", "GA", "GE"]:
        n = sum(1 for c in targets["category"] if c == cat)
        if n:
            print(f"    {cat}: {n} targets")
    print(f"  Mean observability : {vis_map.mean()*100:.1f}%")
    print(f"\nOutputs:")
    print(f"  {OUTPUT_NPZ}")
    print(f"  {OUTPUT_PLOT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
