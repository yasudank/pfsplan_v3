import csv
import datetime
import warnings
import os
import yaml

import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from astroplan import FixedTarget, Observer
from astropy.table import Table

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_config(filename='obs_config.yaml'):
    """
    Load observation parameters from a YAML file with sensible defaults.
    """
    defaults = {
        'location': {
            'name': 'Subaru',
            'timezone': 'US/Hawaii',
            'hst_offset_hours': -10
        },
        'twilight': {
            'horizon_deg': -18.0
        },
        'scheduling': {
            'split_margin_minutes': 5,
            'slot_duration_minutes': 20,
            'min_overhead_min': 5,
            'manual_readout_min': 15,
        },
        'constraints': {
            'max_airmass': 1.6,
            'max_airmass_relaxed': 1.85,
            'min_altitude': 32.5,
            'max_altitude': 75.0,
            'min_teff': 0.6,
            'rotator_min': -174.0,
            'rotator_max': 174.0,
            'min_moon_sep': 20.0,
            'max_moon_ill': 1.0,
            'max_moon_alt': 90.0,
            'max_relaxed_count': 8
        },
        'slew': {
            'speed_az': 0.5,
            'speed_el': 0.5,
            'speed_rot': 1.5
        },
        'scheduler': {
            'sa_t0': 500.0,
            'sa_alpha': 0.99999,
            'sa_iterations': 1000000,
            'sa_t_min': 0.5,
            'weight_hard': 1000000.0,
            'weight_split': 1000.0,
            'weight_priority_base': 100.0,
            'weight_teff': 100.0,
            'weight_conn': 0.0,
            'weight_empty': 5000.0,
            'weight_slew': 5.0,
            'max_priority': 4
        }
    }
    
    config = {}
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Failed to load config from {filename}: {e}. Using defaults.")
            
    # Merge defaults recursively
    for key, def_val in defaults.items():
        if key not in config:
            config[key] = def_val
        else:
            if isinstance(def_val, dict):
                for k, v in def_val.items():
                    if k not in config[key]:
                        config[key][k] = v
                        
    return config

# Load config globally for utility functions
GLOBAL_CONFIG = load_config()

def read_priorities(filename):
    """
    Read priorities from a ppcList.ecsv file.
    """
    priorities = {}
    try:
        table = Table.read(filename, format='ascii.ecsv')
        for row in table:
            # Check if 'ppc_priority' column exists and is not null
            if 'ppc_priority' in table.colnames and row['ppc_priority'] is not None:
                priorities[row['ppc_code']] = int(row['ppc_priority'])
    except FileNotFoundError:
        print(f"Warning: Priority file {filename} not found.")
    except Exception as e:
        print(f"Error reading priorities from {filename}: {e}")
    return priorities

def setup_observer():
    """
    Setup the Subaru Telescope observer.
    """
    # Subaru Telescope location
    # Longitude: 155.4761 W, Latitude: 19.8256 N, Elevation: 4139 m
    #location = EarthLocation(lon=-155.4761*u.deg, lat=19.8256*u.deg, height=4139*u.m)
    loc_name = GLOBAL_CONFIG['location']['name']
    tz = GLOBAL_CONFIG['location']['timezone']
    location = EarthLocation.of_site(loc_name)
    observer = Observer(location=location, name=loc_name, timezone=tz)
    return observer


def read_targets(filename, priorities):
    """
    Read targets from a CSV file.
    Returns a list of dictionaries containing target info and an astroplan FixedTarget.
    """
    targets = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            exptime = float(row['ppc_exptime'])
            if exptime < 900:
                continue
            ppc_code = row['ppc_code']
            priority = priorities.get(ppc_code, 99)
            coord = SkyCoord(ra=float(row['ppc_ra'])*u.deg, dec=float(row['ppc_dec'])*u.deg)
            target = FixedTarget(coord=coord, name=ppc_code)
            targets.append({
                'id': ppc_code,
                'target': target,
                'exptime': exptime,
                'observed': False,
                'priority': priority,
                'ppc_pa': float(row['ppc_pa']) if 'ppc_pa' in row else 0.0
            })
    return targets

def read_targets_from_ppcList(filename, priorities):
    """
    Read targets from a ECSV file.
    Returns a list of dictionaries containing target info and an astroplan FixedTarget.
    """
    targets = []
    table = Table.read(filename, format='ascii.ecsv')
    for row in table:
        exptime = float(row['ppc_exptime'])
        if exptime < 900:
            continue
        ppc_code = row['ppc_code']
        priority = priorities.get(ppc_code, 99)
        coord = SkyCoord(ra=float(row['ppc_ra'])*u.deg, dec=float(row['ppc_dec'])*u.deg)
        target = FixedTarget(coord=coord, name=ppc_code)
        targets.append({
            'id': ppc_code,
            'target': target,
            'exptime': exptime,
            'observed': False,
            'priority': priority,
            'ppc_pa': float(row['ppc_pa']) if 'ppc_pa' in table.colnames else 0.0
        })
    return targets

def parse_time(date_str, time_str, observer, is_end=False):
    """
    Parse time string from obsdates file.
    Handles 'sun_set', 'twilight_end', 'sun_rise', 'twilight_beg', and 'HH:MM'.
    Returns an astropy Time object in UTC.
    """
    # Base date in HST (assuming the file dates are local dates)
    # We need to be careful. Usually observing logs use the date of the start of the night.
    # e.g. 2025-11-12 means the night of Nov 12-13.
    
    # Create a base time at noon HST on that date to calculate sun_set/rise for THAT day
    offset_hours = GLOBAL_CONFIG['location']['hst_offset_hours']
    noon_hour = 12 - offset_hours
    noon_utc = Time(f"{date_str} {noon_hour:02d}:00:00")
    
    horizon = GLOBAL_CONFIG['twilight']['horizon_deg'] * u.deg
    
    if time_str in ['sun_set', 'twilight_end']:
        # Nearest sunset to noon of that day
        return observer.sun_set_time(noon_utc, which='nearest', horizon=horizon)
    elif time_str in ['sun_rise', 'twilight_beg']:
        # Nearest sunrise to noon of that day (which would be the next morning)
        # Actually, for "night of Nov 12", sunrise is Nov 13 morning.
        # observer.sun_rise_time(noon_utc, which='next') should work from noon.
        return observer.sun_rise_time(noon_utc, which='next', horizon=horizon)
    else:
        # Parse HH:MM
        # Handle 24+ format (e.g., 24:05 -> 00:05 next day)
        hours, minutes = map(int, time_str.split(':'))
        days_delta = 0
        if hours >= 24:
            days_delta = hours // 24
            hours = hours % 24
        
        # Construct datetime in HST
        dt = datetime.datetime(local_date.year, local_date.month, local_date.day, hours, minutes) if 'local_date' in locals() else datetime.datetime.strptime(date_str, "%Y-%m-%d") + datetime.timedelta(hours=hours, minutes=minutes)
        # Apply day delta if any
        if days_delta > 0:
            dt += datetime.timedelta(days=days_delta)
        
        # Convert HST to UTC
        dt_utc = dt - datetime.timedelta(hours=offset_hours)
        return Time(dt_utc)


def read_obsdates(filename, observer, skip_days=None, limit_days=None):
    """
    Read observation dates from file.
    Returns list of (start_time, end_time) tuples.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Skip header
    lines = lines[1:]
    
    nights = []
    for i, line in enumerate(lines):
        if skip_days and i < skip_days:
            continue
        if limit_days and i >= limit_days:
            break
        parts = line.split()
        if len(parts) < 3:
            continue
        
        date_str = parts[0]
        start_str = parts[1]
        end_str = parts[2]
        
        start_time = parse_time(date_str, start_str, observer)
        end_time = parse_time(date_str, end_str, observer, is_end=True)
        
        nights.append((start_time, end_time))
        
    return nights
