import csv
import numpy as np
from astropy.time import Time, TimeDelta
from astropy.coordinates import get_body, SkyCoord
import astropy.units as u
import datetime
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import warnings
import json

from obs_utils import setup_observer, read_targets, read_obsdates, read_priorities, read_targets_from_ppcList

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_schedule_from_json_and_npz(json_path, npz_path, observer):
    """
    Load the schedule from JSON and NPZ files and reconstruct the observation dictionaries.
    """
    with open(json_path, 'r') as f:
        res = json.load(f)
    
    data = np.load(npz_path, allow_pickle=True)
    
    target_codes = data["target_codes"]
    target_ra = data["target_ra"]
    target_dec = data["target_dec"]
    target_ppc_pa = data["target_ppc_pa"]
    target_exptime = data["target_exptime"]
    slot_times_iso = data["slot_times_iso"]
    slot_night_idx = data["slot_night_idx"]
    
    schedule = []
    
    print("Reconstructing schedule observations...")
    for si, ti in enumerate(res["schedule"]):
        if ti < 0:
            continue
        
        # Calculate time steps
        t_start = Time(slot_times_iso[si])
        t_mid = t_start + TimeDelta(600, format="sec")
        t_end = t_start + TimeDelta(1200, format="sec")
        
        # Get target coordinates and ppc_pa
        ra = target_ra[ti]
        dec = target_dec[ti]
        ppc_pa = target_ppc_pa[ti]
        
        coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
        
        # Calculate AltAz and airmass at midpoint
        altaz_mid = observer.altaz(t_mid, coord)
        altitude = altaz_mid.alt.deg
        airmass = altaz_mid.secz.value if hasattr(altaz_mid.secz, 'value') else altaz_mid.secz
        
        # Calculate rotator angle (pa + ppc_pa wrapped to [-180, 180])
        pa_start = observer.parallactic_angle(t_start, coord).deg
        pa_end = observer.parallactic_angle(t_end, coord).deg
        pa_mid = observer.parallactic_angle(t_mid, coord).deg
        
        rot_start = (pa_start + ppc_pa + 180) % 360 - 180
        rot_end = (pa_end + ppc_pa + 180) % 360 - 180
        rotator_angle = (pa_mid + ppc_pa + 180) % 360 - 180
        
        row = {
            'night': int(slot_night_idx[si]) + 1, # 1-based night
            'start_time': t_start.isot,
            'target': str(target_codes[ti]),
            'ra': float(ra),
            'dec': float(dec),
            'exptime': float(target_exptime[ti]),
            'altitude': float(altitude),
            'airmass': float(airmass),
            'rotator_angle': float(rotator_angle),
            'rot_start': float(rot_start),
            'rot_end': float(rot_end)
        }
        schedule.append(row)
        
    return schedule

def get_target_colors(all_targets):
    """
    Create a dictionary mapping target ID to color based on ppc_code group.
    """
    # Filter IDs
    co_ids_raw = list(set(t['id'] for t in all_targets if t['id'].startswith('SSP_CO')))
    ga_ids = sorted(list(set(t['id'] for t in all_targets if t['id'].startswith('SSP_GA'))))
    ge_ids = sorted(list(set(t['id'] for t in all_targets if t['id'].startswith('SSP_GE'))))

    # Custom sort for SSP_CO based on the number before 'h'
    def co_sort_key(tid):
        match = re.search(r'_(\d+)h_', tid)
        if match:
            return int(match.group(1))
        return 999999999 # Fallback for IDs not matching the pattern

    co_ids = sorted(co_ids_raw, key=co_sort_key)
    
    # Create colormaps
    cm_co = LinearSegmentedColormap.from_list("co", ["blue", "violet"])
    cm_ga = LinearSegmentedColormap.from_list("ga", ["yellow", "green"])
    cm_ge = LinearSegmentedColormap.from_list("ge", ["red", "orange"])
    
    target_colors = {}
    
    def assign_colors(ids, cmap):
        n = len(ids)
        for i, tid in enumerate(ids):
            norm = i / (n - 1) if n > 1 else 0.5
            target_colors[tid] = cmap(norm)
            
    assign_colors(co_ids, cm_co)
    assign_colors(ga_ids, cm_ga)
    assign_colors(ge_ids, cm_ge)
    
    return target_colors

def plot_altitude_time(schedule, nights, observer, target_colors=None):
    """
    Plot Altitude vs Time for the schedule.
    One panel per night.
    Common x-axis range (17:00 to 07:00 HST).
    Mark astronomical twilight start/end.
    Differentiate Manual vs Auto targets.
    """
    print("Generating Altitude vs Time plot...")
    
    n_nights = len(nights)
    
    fig, axes = plt.subplots(n_nights, 1, figsize=(10, 7), sharex=False, sharey=True)
    if n_nights == 1:
        axes = [axes]
    
    colors = plt.cm.jet(np.linspace(0, 1, n_nights))
    
    for i, (start_utc, end_utc) in enumerate(nights):
        night_idx = i + 1
        ax = axes[i]
        
        # Twilight times in HST
        start_hst = start_utc.datetime - datetime.timedelta(hours=10)
        end_hst = end_utc.datetime - datetime.timedelta(hours=10)

        # Set title with date
        date_str = start_hst.strftime('%Y-%m-%d')
        ax.set_title(f"{date_str} (HST)", loc='left', fontsize=10)
        
        # Set common x-axis limits: 17:00 previous day to 07:00 current day
        anchor_date = start_hst.date()
        if start_hst.hour < 15:
            anchor_date -= datetime.timedelta(days=1)
            
        xlim_start = datetime.datetime.combine(anchor_date, datetime.time(18, 0))
        xlim_end = xlim_start + datetime.timedelta(hours=12) # 06:00 next day
        
        # --- Draw Twilight Shade ---
        # Create a time grid for calculation (200 points)
        total_minutes = (xlim_end - xlim_start).total_seconds() / 60
        minutes_grid = np.linspace(0, total_minutes, 200)
        times_dt_naive_hst = [xlim_start + datetime.timedelta(minutes=m) for m in minutes_grid]
        times_dt_naive_utc = [t + datetime.timedelta(hours=10) for t in times_dt_naive_hst]
        times_astropy = Time(times_dt_naive_utc)
        
        # Calculate Sun Altitude
        sun_coo = get_body("sun", times_astropy, location=observer.location)
        sun_alt = observer.altaz(times_astropy, sun_coo).alt.deg

        # Calculate Moon Altitude
        moon_coo = get_body("moon", times_astropy, location=observer.location)
        moon_alt = observer.altaz(times_astropy, moon_coo).alt.deg
        
        # Create gradient image data (1 row, N cols, RGBA)
        img_data = np.zeros((1, len(minutes_grid), 4))
        img_data[:, :, 0] = 1.0
        img_data[:, :, 1] = 0.7
        img_data[:, :, 2] = 0.2
        
        # Calculate Alpha based on altitude
        max_alpha = 0.5
        alphas = np.zeros_like(sun_alt)
        
        mask_day = sun_alt >= 0
        mask_twilight = (sun_alt < 0) & (sun_alt > -18)
        
        alphas[mask_day] = max_alpha
        alphas[mask_twilight] = max_alpha * (sun_alt[mask_twilight] + 18) / 18.0
        
        img_data[:, :, 3] = alphas
        
        # Draw gradient using imshow
        ax.imshow(img_data, extent=[mdates.date2num(xlim_start), mdates.date2num(xlim_end), 0, 90], 
                  aspect='auto', origin='lower', zorder=0)
        
        # Plot Moon Altitude
        ax.plot(times_dt_naive_hst, moon_alt, color='black', linestyle='--', alpha=0.6, label='Moon', zorder=5)

        # Plot twilight lines
        ax.axvline(start_hst, color='red', linestyle='--', alpha=0.5, label='Twilight')
        ax.axvline(end_hst, color='red', linestyle='--', alpha=0.5)
        
        # Filter observations for this night
        night_obs = [s for s in schedule if s['night'] == night_idx]
        
        if night_obs:
            # Plot All Observations with target colors
            times = [Time(s['start_time']).datetime - datetime.timedelta(hours=10) for s in night_obs]
            alts = [s['altitude'] for s in night_obs]
            
            if target_colors:
                c = [target_colors.get(s['target'], 'grey') for s in night_obs]
                ax.scatter(times, alts, label="Observations", c=c, s=30, marker='o', zorder=10)
            else:
                ax.scatter(times, alts, label="Observations", color=colors[i], s=30, marker='o', zorder=10)
        
        ax.grid(True, alpha=0.3)
        ax.set_xlim(xlim_start, xlim_end)
        ax.set_yticks(range(0, 91, 30))
        ax.set_ylim(0, 90)

        # Warning zone for low altitude (Altitude < 30)
        ax.fill_between([xlim_start, xlim_end], 0, 30, color='red', alpha=0.15, zorder=0)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        if i < n_nights - 1:
            ax.tick_params(labelbottom=False)

    fig.supylabel('Altitude (deg)', fontsize=14)
    axes[-1].set_xlabel("Time (HST)")
    plt.tight_layout()
    
    plt.savefig("altitude_vs_time.png")
    plt.close(fig)
    print("Saved altitude_vs_time.png")

def plot_rotator_angle_time(schedule, nights, observer, target_colors=None):
    """
    Plot Rotator Angle vs Time for the schedule.
    """
    print("Generating Rotator Angle vs Time plot...")
    
    n_nights = len(nights)
    
    fig, axes = plt.subplots(n_nights, 1, figsize=(10, 7), sharex=False, sharey=True)
    if n_nights == 1:
        axes = [axes]
    
    colors = plt.cm.jet(np.linspace(0, 1, n_nights))
    
    for i, (start_utc, end_utc) in enumerate(nights):
        night_idx = i + 1
        ax = axes[i]
        
        # Twilight times in HST
        start_hst = start_utc.datetime - datetime.timedelta(hours=10)
        end_hst = end_utc.datetime - datetime.timedelta(hours=10)

        # Set title with date
        date_str = start_hst.strftime('%Y-%m-%d')
        ax.set_title(f"{date_str} (HST)", loc='left', fontsize=10)
        
        # Set common x-axis limits
        anchor_date = start_hst.date()
        if start_hst.hour < 15:
            anchor_date -= datetime.timedelta(days=1)
            
        xlim_start = datetime.datetime.combine(anchor_date, datetime.time(18, 0))
        xlim_end = xlim_start + datetime.timedelta(hours=12) 
        
        # --- Draw Twilight Shade ---
        total_minutes = (xlim_end - xlim_start).total_seconds() / 60
        minutes_grid = np.linspace(0, total_minutes, 200)
        times_dt_naive_hst = [xlim_start + datetime.timedelta(minutes=m) for m in minutes_grid]
        times_dt_naive_utc = [t + datetime.timedelta(hours=10) for t in times_dt_naive_hst]
        times_astropy = Time(times_dt_naive_utc)
        
        # Calculate Sun Altitude
        sun_coo = get_body("sun", times_astropy, location=observer.location)
        sun_alt = observer.altaz(times_astropy, sun_coo).alt.deg
        
        # Create gradient image data (1 row, N cols, RGBA)
        img_data = np.zeros((1, len(minutes_grid), 4))
        img_data[:, :, 0] = 1.0
        img_data[:, :, 1] = 0.7
        img_data[:, :, 2] = 0.2
        
        # Calculate Alpha based on altitude
        max_alpha = 0.5
        alphas = np.zeros_like(sun_alt)
        
        mask_day = sun_alt >= 0
        mask_twilight = (sun_alt < 0) & (sun_alt > -18)
        
        alphas[mask_day] = max_alpha
        alphas[mask_twilight] = max_alpha * (sun_alt[mask_twilight] + 18) / 18.0
        
        img_data[:, :, 3] = alphas
        
        # Draw gradient using imshow
        ax.imshow(img_data, extent=[mdates.date2num(xlim_start), mdates.date2num(xlim_end), -180, 180], 
                  aspect='auto', origin='lower', zorder=0)

        # Plot twilight lines
        ax.axvline(start_hst, color='red', linestyle='--', alpha=0.5, label='Twilight')
        ax.axvline(end_hst, color='red', linestyle='--', alpha=0.5)
        
        # Filter observations for this night
        night_obs = [s for s in schedule if s['night'] == night_idx]
        
        if night_obs:
            # Plot All Observations with target colors
            times = [Time(s['start_time']).datetime - datetime.timedelta(hours=10) for s in night_obs]
            rots = [s.get('rot_start', s.get('rotator_angle', 0)) for s in night_obs]
            
            if target_colors:
                c = [target_colors.get(s['target'], 'grey') for s in night_obs]
                ax.scatter(times, rots, label="Observations", c=c, s=20, marker='o', zorder=10)
            else:
                ax.scatter(times, rots, label="Observations", color=colors[i], s=20, marker='o', zorder=10)
        
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-180, 180)
        ax.set_yticks(range(-180, 181, 60))
        
        ax.set_xlim(xlim_start, xlim_end)
        
        # Warning zone for rotator limits (-174, 174)
        ax.fill_between([xlim_start, xlim_end], -180, -174, color='red', alpha=0.15, zorder=0)
        ax.fill_between([xlim_start, xlim_end], 174, 180, color='red', alpha=0.15, zorder=0)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        if i < n_nights - 1:
            ax.tick_params(labelbottom=False)
        
    fig.supylabel('Rotator Angle (deg)', fontsize=14)
    axes[-1].set_xlabel("Time (HST)")
    plt.tight_layout()
    
    plt.savefig("rotator_angle_vs_time.png")
    plt.close(fig)
    print("Saved rotator_angle_vs_time.png")

def plot_sky_coverage(schedule, all_targets, target_colors=None):
    """
    Plot Sky Coverage.
    """
    from matplotlib.lines import Line2D
    print("Generating Sky Coverage plot...")
    
    if target_colors is None:
        target_colors = get_target_colors(all_targets)
    
    try:
        scheduled_ids = set(s['target'] for s in schedule)
        
        # Determine RA range for SSP_CO
        co_ras = [t['target'].coord.ra.deg for t in all_targets if t['id'].startswith('SSP_CO')]
        if not co_ras:
            ra_min, ra_max = 0, 360
        else:
            if max(co_ras) - min(co_ras) > 180:
                co_ras_shifted = [ra - 360 if ra > 180 else ra for ra in co_ras]
                ra_min, ra_max = min(co_ras_shifted), max(co_ras_shifted)
                use_shifted = True
            else:
                ra_min, ra_max = min(co_ras), max(co_ras)
                use_shifted = False
        
        ra_min -= 5
        ra_max += 5
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        def get_color(tid):
            return target_colors.get(tid, 'grey')

        for t in all_targets:
            tid = t['id']
            ra = t['target'].coord.ra.deg
            dec = t['target'].coord.dec.deg
            
            if use_shifted and ra > 180:
                ra -= 360
                
            is_scheduled = tid in scheduled_ids
            color = get_color(tid)
            
            if is_scheduled:
                facecolor = color
                edgecolor = color
                alpha = 0.6
                fill = True
                zorder = 10
            else:
                facecolor = 'none'
                edgecolor = color
                alpha = 0.4
                fill = False
                zorder = 1
            
            if ra_min <= ra <= ra_max:
                circle = Circle((ra, dec), 0.7, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, fill=fill, zorder=zorder)
                ax.add_patch(circle)
            
        ax.set_xlim(ra_min, ra_max)
        ax.set_ylim(-10, 10) 
        ax.set_xlabel("RA (deg)")
        ax.set_ylabel("Dec (deg)")
        ax.set_title(f"Sky Coverage (RA {ra_min:.1f} to {ra_max:.1f})")
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.invert_xaxis()
        
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='SSP_CO', markerfacecolor='blueviolet', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='SSP_GA', markerfacecolor='yellowgreen', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='SSP_GE', markerfacecolor='orangered', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Scheduled (Filled)', markerfacecolor='grey', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Unobserved (Open)', markerfacecolor='none', markeredgecolor='grey', markersize=10),
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), 
                  fontsize='small', frameon=True, title="Legend")

        plt.tight_layout()
        plt.savefig("sky_coverage.png", bbox_inches='tight')
        plt.close(fig)
        print("Saved sky_coverage.png")
        
    except Exception as e:
        print(f"Error generating sky coverage plot: {e}")
        import traceback
        traceback.print_exc()
        if 'fig' in locals():
            plt.close(fig)

def plot_sky_coverage_mollweide(schedule, all_targets, target_colors=None):
    """
    Plot Sky Coverage for the entire sky using Mollweide projection.
    """
    from matplotlib.lines import Line2D
    print("Generating Sky Coverage (Mollweide) plot...")
    
    if target_colors is None:
        target_colors = get_target_colors(all_targets)
    
    try:
        if schedule:
            scheduled_ids = set(s['target'] for s in schedule)
        else:
            scheduled_ids = set()
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='mollweide')
        
        def get_category(tid):
            if tid.startswith('SSP_CO'): return 'SSP_CO'
            if tid.startswith('SSP_GA'): return 'SSP_GA'
            if tid.startswith('SSP_GE'): return 'SSP_GE'
            return 'Other'

        data = {}

        for t in all_targets:
            tid = t['id']
            ra_deg = t['target'].coord.ra.deg
            dec_deg = t['target'].coord.dec.deg
            
            ra_rad = np.radians(ra_deg)
            if ra_rad > np.pi:
                ra_rad -= 2 * np.pi
            ra_plot = -ra_rad
            dec_plot = np.radians(dec_deg)
            
            is_scheduled = tid in scheduled_ids
            cat = get_category(tid)
            color = target_colors.get(tid, 'grey')
            
            key = (cat, is_scheduled)
            if key not in data:
                data[key] = {'ra': [], 'dec': [], 'colors': []}
            
            data[key]['ra'].append(ra_plot)
            data[key]['dec'].append(dec_plot)
            data[key]['colors'].append(color)
            
        for (cat, is_scheduled), val in data.items():
            if not val['ra']:
                continue
                
            c_array = val['colors']
            if is_scheduled:
                ax.scatter(val['ra'], val['dec'], c=c_array, edgecolors=c_array, alpha=0.7, s=20, label=f"{cat} (Sched)")
            else:
                ax.scatter(val['ra'], val['dec'], facecolors='none', edgecolors=c_array, alpha=0.4, s=15, label=f"{cat} (Unobs)")

        ax.grid(True)
        ax.set_title("Sky Coverage (Mollweide, RA 0 at Center)")

        tick_locations = np.radians(np.arange(-150, 180, 30))
        ax.set_xticks(tick_locations)
        
        tick_labels = []
        for x in tick_locations:
            ra_val = np.degrees(-x)
            if ra_val < 0:
                ra_val += 360
            tick_labels.append(f"{int(round(ra_val))}°")
            
        ax.set_xticklabels(tick_labels)
        
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='SSP_CO', markerfacecolor='blue', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='SSP_GA', markerfacecolor='green', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='SSP_GE', markerfacecolor='red', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Scheduled (Filled)', markerfacecolor='grey', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Unobserved (Open)', markerfacecolor='none', markeredgecolor='grey', markersize=10),
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig("sky_coverage_mollweide.png")
        plt.close(fig)
        print("Saved sky_coverage_mollweide.png")
        
    except Exception as e:
        print(f"Error generating sky coverage mollweide plot: {e}")
        import traceback
        traceback.print_exc()
        if 'fig' in locals():
            plt.close(fig)

def animate_sky_coverage_progress(schedule, all_targets, target_colors=None):
    """
    Animate Sky Coverage progress (cumulative) over the nights.
    """
    from matplotlib.lines import Line2D
    print("Generating Sky Coverage Progress animation...")

    if target_colors is None:
        target_colors = get_target_colors(all_targets)
    
    if not schedule:
        print("No schedule to animate.")
        return

    nights = sorted(list(set(s['night'] for s in schedule)))
    if not nights:
        return
    max_night = max(nights)
    
    target_first_night = {}
    for s in schedule:
        tid = s['target']
        n = s['night']
        if tid not in target_first_night or n < target_first_night[tid]:
            target_first_night[tid] = n
            
    co_ras = [t['target'].coord.ra.deg for t in all_targets if t['id'].startswith('SSP_CO')]
    if not co_ras:
        ra_min, ra_max = 0, 360
    else:
        if max(co_ras) - min(co_ras) > 180:
            co_ras_shifted = [ra - 360 if ra > 180 else ra for ra in co_ras]
            ra_min, ra_max = min(co_ras_shifted), max(co_ras_shifted)
            use_shifted = True
        else:
            ra_min, ra_max = min(co_ras), max(co_ras)
            use_shifted = False
    
    ra_min -= 5
    ra_max += 5
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    def get_color(tid):
        return target_colors.get(tid, 'grey')
        
    patches_map = {}
    
    for t in all_targets:
        tid = t['id']
        ra = t['target'].coord.ra.deg
        dec = t['target'].coord.dec.deg
        color = get_color(tid)
        
        if use_shifted and ra > 180:
            ra -= 360
            
        patches_map[tid] = []
        
        if ra_min <= ra <= ra_max:
            c = Circle((ra, dec), 0.7, facecolor='none', edgecolor=color, alpha=0.4, fill=False, zorder=1)
            ax.add_patch(c)
            patches_map[tid].append(c)
            
    ax.set_xlim(ra_min, ra_max)
    ax.set_ylim(-10, 10) 
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.invert_xaxis()
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='SSP_CO', markerfacecolor='blueviolet', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='SSP_GA', markerfacecolor='yellowgreen', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='SSP_GE', markerfacecolor='orangered', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Scheduled (Filled)', markerfacecolor='grey', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Unobserved (Open)', markerfacecolor='none', markeredgecolor='grey', markersize=10),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), 
              fontsize='small', frameon=True, title="Legend")
    
    def update(frame):
        title_suffix = "Start" if frame == 0 else f"Night {frame}"
        ax.set_title(f"Sky Coverage Progress - {title_suffix}")
        
        for tid, patches in patches_map.items():
            color = get_color(tid)
            is_observed = (tid in target_first_night) and (target_first_night[tid] <= frame) and (frame > 0)
            
            for p in patches:
                if is_observed:
                    p.set_facecolor(color)
                    p.set_alpha(0.6)
                    p.set_fill(True)
                    p.set_zorder(10)
                else:
                    p.set_facecolor('none')
                    p.set_alpha(0.4)
                    p.set_fill(False)
                    p.set_zorder(1)
        return []

    frames = range(0, max_night + 1)
    anim = FuncAnimation(fig, update, frames=frames, interval=800)
    
    try:
        anim.save('sky_coverage_progress.gif', writer='pillow', fps=2)
        print("Saved sky_coverage_progress.gif")
    except Exception as e:
        print(f"Error saving animation: {e}")
        import traceback
        traceback.print_exc()

    plt.close(fig)

def plot_observation_counts(schedule, all_targets, target_colors=None):
    """
    Plot histogram of observation counts per target.
    """
    print("Generating Observation Counts plot...")
    from collections import Counter
    
    if target_colors is None:
        target_colors = get_target_colors(all_targets)

    counts = Counter(s['target'] for s in schedule)
    if not counts:
        print("No observations to plot.")
        return

    def get_group(tid):
        if tid.startswith('SSP_GA'):
            match = re.search(r'V\d{2}', tid)
            if match:
                return tid[:match.start()]
        elif tid.startswith('SSP_GE'):
            match = re.search(r'_obs\d{4}', tid)
            if match:
                return tid[:match.start()]
        return tid

    group_to_tids = {}
    for tid, count in counts.items():
        grp = get_group(tid)
        if grp not in group_to_tids:
            group_to_tids[grp] = {}
        group_to_tids[grp][tid] = count

    def group_sort_key(grp):
        if grp.startswith('SSP_CO'): cat = 1
        elif grp.startswith('SSP_GA'): cat = 2
        elif grp.startswith('SSP_GE'): cat = 3
        else: cat = 4
        
        match = re.search(r'_(\d+)h_', grp)
        num = int(match.group(1)) if match else 999999
        
        return (cat, num, grp)

    sorted_groups = sorted(group_to_tids.keys(), key=group_sort_key)

    final_groups = []
    prev_cat = None
    for grp in sorted_groups:
        curr_cat = 1 if grp.startswith('SSP_CO') else 2 if grp.startswith('SSP_GA') else 3 if grp.startswith('SSP_GE') else 4
        if prev_cat is not None and curr_cat != prev_cat:
            for _ in range(2):
                final_groups.append(None)
        final_groups.append(grp)
        prev_cat = curr_cat

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(final_groups))
    
    for i, grp in enumerate(final_groups):
        if grp is None:
            continue
        tids_in_grp = sorted(group_to_tids[grp].keys())
        current_bottom = 0
        for tid in tids_in_grp:
            count = group_to_tids[grp][tid]
            color = target_colors.get(tid, 'grey')
            ax.bar(i, count, bottom=current_bottom, color=color, edgecolor='none', width=0.8)
            current_bottom += count

    ax.set_ylabel("Observation Count")
    ax.set_title("Observation Counts per Target (Grouped GA/GE)")
    ax.margins(x=0.02)
    
    ax.set_xticks(x)
    x_labels = []
    co_counter = 0
    for grp in final_groups:
        if grp is None:
            x_labels.append("")
        elif grp.startswith('SSP_COs'):
            if co_counter % 2 == 0:
                x_labels.append(grp)
            else:
                x_labels.append("") 
            co_counter += 1
        else:
            label = grp.rstrip('_')
            x_labels.append(label)
            
    ax.set_xticklabels(x_labels, rotation=90, ha='center', fontsize=7)

    group_summary = {'CO': 0, 'GA': 0, 'GE': 0}
    for tid, count in counts.items():
        if tid.startswith('SSP_CO'): group_summary['CO'] += count
        elif tid.startswith('SSP_GA'): group_summary['GA'] += count
        elif tid.startswith('SSP_GE'): group_summary['GE'] += count
    
    ax_ins = fig.add_axes([0.15, 0.65, 0.15, 0.2]) 
    
    groups = ['CO', 'GA', 'GE']
    g_vals = [group_summary[g] for g in groups]
    g_colors = ['blueviolet', 'yellowgreen', 'orangered']
    
    ax_ins.bar(groups, g_vals, color=g_colors)
    ax_ins.set_title("Counts by Group", fontsize=9)
    ax_ins.tick_params(axis='both', which='major', labelsize=8)
    
    if g_vals:
        ax_ins.set_ylim(0, max(g_vals) * 1.3)
    
    total_obs = sum(g_vals)
    for i, v in enumerate(g_vals):
        pct = (v / total_obs * 100) if total_obs > 0 else 0
        ax_ins.text(i, v, f"{v}\n({pct:.1f}%)", ha='center', va='bottom', fontsize=7)
    
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_minor_locator(MaxNLocator(integer=True))
    ax_ins.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("observation_counts.png")
    plt.close(fig)
    print("Saved observation_counts.png")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot PFS Schedule and Coverage Plots")
    parser.add_argument("-o", "--obsdates", type=str, default="obsdates_2026May.txt", help="Path to obsdates text file (default: obsdates_2026May.txt)")
    parser.add_argument("-v", "--vis-map", type=str, default="vis_map.npz", help="Path to vis_map.npz (default: vis_map.npz)")
    parser.add_argument("-s", "--schedule", type=str, default="schedule_result.json", help="Path to schedule_result.json (default: schedule_result.json)")
    args = parser.parse_args()

    observer = setup_observer()
    
    # Read priorities first
    priorities = read_priorities('targets/CO/ppcList.ecsv')
    if not priorities:
        print("Warning: Could not read priorities. Proceeding with default priority for all targets.")
    
    all_targets = []
    target_files = ['targets/CO/ppcList.ecsv', 'targets/GA/ppcList.ecsv', 'targets/GE/ppcList.ecsv']
    
    for fname in target_files:
        try:
            targets = read_targets_from_ppcList(fname, priorities)
            all_targets.extend(targets)
            print(f"Loaded {len(targets)} targets from {fname}.")
        except FileNotFoundError:
            print(f"Warning: Target file '{fname}' not found.")
    
    print(f"Total targets loaded: {len(all_targets)}")

    try:
        schedule = load_schedule_from_json_and_npz(args.schedule, args.vis_map, observer)
    except Exception as e:
        print(f"Error loading schedule: {e}")
        return
        
    if not schedule:
        print("Schedule is empty. Generating empty sky coverage plot.")
        target_colors = get_target_colors(all_targets)
        plot_sky_coverage(schedule, all_targets, target_colors)
        plot_sky_coverage_mollweide(schedule, all_targets, target_colors)
        print("Altitude vs time plot not generated as there are no observations.")
        return

    print(f"Loaded {len(schedule)} observations from schedule.")

    try:
        nights = read_obsdates(args.obsdates, observer)
        print(f"Loaded {len(nights)} observation windows from {args.obsdates}.")
    except FileNotFoundError:
        print(f"Error: Observation dates file '{args.obsdates}' not found.")
        print("Cannot generate altitude vs time plot.")
        nights = []

    target_colors = get_target_colors(all_targets)
    if nights:
        plot_altitude_time(schedule, nights, observer, target_colors)
        plot_rotator_angle_time(schedule, nights, observer, target_colors)
    plot_sky_coverage(schedule, all_targets, target_colors)
    plot_sky_coverage_mollweide(schedule, all_targets, target_colors)
    animate_sky_coverage_progress(schedule, all_targets, target_colors)
    plot_observation_counts(schedule, all_targets, target_colors)

if __name__ == '__main__':
    main()
