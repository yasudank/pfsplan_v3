import csv
import datetime
import warnings

import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from astroplan import FixedTarget, Observer
from astropy.table import Table

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

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
    location = EarthLocation.of_site('Subaru')
    observer = Observer(location=location, name="Subaru", timezone="US/Hawaii")
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
            ppc_code = row['ppc_code']
            priority = priorities.get(ppc_code, 99)
            coord = SkyCoord(ra=float(row['ppc_ra'])*u.deg, dec=float(row['ppc_dec'])*u.deg)
            target = FixedTarget(coord=coord, name=ppc_code)
            targets.append({
                'id': ppc_code,
                'target': target,
                'exptime': float(row['ppc_exptime']),
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
        ppc_code = row['ppc_code']
        priority = priorities.get(ppc_code, 99)
        coord = SkyCoord(ra=float(row['ppc_ra'])*u.deg, dec=float(row['ppc_dec'])*u.deg)
        target = FixedTarget(coord=coord, name=ppc_code)
        targets.append({
            'id': ppc_code,
            'target': target,
            'exptime': float(row['ppc_exptime']),
            'observed': False,
            'priority': priority,
            'ppc_pa': float(row['ppc_pa']) if 'ppc_pa' in table.colnames else 0.0
        })
    return targets

def parse_time(date_str, time_str, observer, is_end=False):
    """
    Parse time string from obsdates file.
    Handles 'sun_set', 'sun_rise', and 'HH:MM'.
    Returns an astropy Time object in UTC.
    """
    # Base date in HST (assuming the file dates are local dates)
    # We need to be careful. Usually observing logs use the date of the start of the night.
    # e.g. 2025-11-12 means the night of Nov 12-13.
    
    # Create a base time at noon HST on that date to calculate sun_set/rise for THAT day
    # Timezone 'US/Hawaii' is UTC-10
    
    # Naive parsing first
    local_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    
    # For sun_set/rise calculations, we need a reference time. 
    # Noon local time is a good reference for finding nearest sunset/sunrise.
    # 12:00 HST = 22:00 UTC
    noon_utc = Time(f"{date_str} 22:00:00")
    
    if time_str in ['sun_set', 'twilight_end']:
        # Nearest sunset to noon of that day
        return observer.sun_set_time(noon_utc, which='nearest', horizon=-18*u.deg)
    elif time_str in ['sun_rise', 'twilight_beg']:
        # Nearest sunrise to noon of that day (which would be the next morning)
        # Actually, for "night of Nov 12", sunrise is Nov 13 morning.
        # observer.sun_rise_time(noon_utc, which='next') should work from noon.
        return observer.sun_rise_time(noon_utc, which='next', horizon=-18*u.deg)
    else:
        # Parse HH:MM
        # Handle 24+ format (e.g., 24:05 -> 00:05 next day)
        hours, minutes = map(int, time_str.split(':'))
        days_delta = 0
        if hours >= 24:
            days_delta = hours // 24
            hours = hours % 24
        
        # Construct datetime in HST
        dt = datetime.datetime(local_date.year, local_date.month, local_date.day, hours, minutes)
        dt += datetime.timedelta(days=days_delta)
        
        # Convert HST to UTC (HST is UTC-10)
        dt_utc = dt + datetime.timedelta(hours=10)
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
