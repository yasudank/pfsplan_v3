#!/usr/bin/env module
#!/usr/bin/env python3
import argparse
import json
import os
import warnings
from datetime import datetime
from collections import defaultdict

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, get_body
from astropy.time import Time, TimeDelta
from astroplan import Observer, moon_illumination
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT

from obs_utils import setup_observer, load_config

# Suppress warnings
warnings.filterwarnings('ignore')

def load_schedule_data(json_path, npz_path, observer):
    """
    Load schedule result JSON and visibility map NPZ,
    and compute all required schedule details dynamically.
    Returns a list of dictionaries.
    """
    print(f"Loading schedule from {json_path}...")
    with open(json_path, 'r') as f:
        res = json.load(f)
    
    print(f"Loading visibility map from {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)
    
    target_codes = data["target_codes"]
    target_ra = data["target_ra"]
    target_dec = data["target_dec"]
    target_ppc_pa = data["target_ppc_pa"]
    slot_times_iso = data["slot_times_iso"]
    slot_night_idx = data["slot_night_idx"]
    
    # Optional / target_teff fallback
    target_teff = data["target_teff"] if "target_teff" in data else None
    
    schedule_data = []
    
    print("Reconstructing schedule observations and calculating parameters...")
    for si, ti in enumerate(res["schedule"]):
        if ti < 0:
            continue
            
        t_start = Time(slot_times_iso[si])
        t_mid = t_start + TimeDelta(600, format="sec")
        t_end = t_start + TimeDelta(1200, format="sec")
        
        ra = target_ra[ti]
        dec = target_dec[ti]
        ppc_pa = target_ppc_pa[ti]
        coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
        
        # Local Sidereal Time
        lst = observer.local_sidereal_time(t_start)
        lst_str = lst.to_string(sep=':', precision=0)
        
        # Airmass at midpoint
        altaz_mid = observer.altaz(t_mid, coord)
        altitude = altaz_mid.alt.deg
        airmass = altaz_mid.secz.value if hasattr(altaz_mid.secz, 'value') else altaz_mid.secz
        
        # Elevation start/end
        elevation_start = observer.altaz(t_start, coord).alt.deg
        elevation_end = observer.altaz(t_end, coord).alt.deg
        
        # Rotator start/end
        pa_start = observer.parallactic_angle(t_start, coord).deg
        pa_end = observer.parallactic_angle(t_end, coord).deg
        
        rot_start = (pa_start + ppc_pa + 180) % 360 - 180
        rot_end = (pa_end + ppc_pa + 180) % 360 - 180
        
        # Moon separation, illumination, and altitude at midpoint
        moon_coord = get_body("moon", t_mid, location=observer.location)
        moon_sep = coord.separation(moon_coord).deg
        moon_illum = moon_illumination(t_mid)
        moon_alt = observer.altaz(t_mid, moon_coord).alt.deg
        
        # Teff
        teff_val = target_teff[ti, si] if target_teff is not None else 1.0
        
        row = {
            'night': int(slot_night_idx[si]) + 1,  # 1-based night
            'start_time': t_start.isot,
            'end_time': t_end.isot,
            'target': str(target_codes[ti]),
            'lst': lst_str,
            'teff': float(teff_val),
            'airmass': float(airmass),
            'elevation_start': float(elevation_start),
            'elevation_end': float(elevation_end),
            'rot_start': float(rot_start),
            'rot_end': float(rot_end),
            'moon_sep': float(moon_sep),
            'moon_illum': float(moon_illum),
            'moon_alt': float(moon_alt)
        }
        schedule_data.append(row)
        
    return schedule_data

def create_pdf_report(schedule_json, vis_map_npz, output_pdf=None):
    # Set up observer
    print("Setting up observer...")
    try:
        observer = setup_observer()
    except Exception as e:
        print(f"Error setting up observer: {e}")
        return
        
    # Reconstruct schedule list
    schedule_list = load_schedule_data(schedule_json, vis_map_npz, observer)
    
    if not schedule_list:
        print("Warning: Loaded schedule is empty. No PDF will be generated.")
        return

    # Determine filename and title suffix based on content and execution date
    t_start = Time(schedule_list[0]['start_time'])
    t_hst = t_start - 22 * u.hour
    period_str = t_hst.datetime.strftime('%Y%b')

    exec_date_str = datetime.now().strftime('v%Y%m%d')
    
    if output_pdf is None:
        output_pdf = f"obsplan_{period_str}.{exec_date_str}.pdf"
    
    print(f"Output filename set to: {output_pdf}")

    # Load configuration constraints
    config = load_config()
    constraints = config.get('constraints', {})
    
    # Prepare ReportLab document
    doc = SimpleDocTemplate(
        output_pdf,
        pagesize=landscape(A4),
        rightMargin=1*cm, leftMargin=1*cm,
        topMargin=1*cm, bottomMargin=1*cm
    )
    
    elements = []
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        alignment=TA_CENTER,
        fontSize=16,
        spaceAfter=12
    )
    
    cell_style = ParagraphStyle(
        'CellStyle',
        parent=styles['Normal'],
        fontSize=8,
        leading=10,
        alignment=TA_LEFT
    )

    right_cell_style = ParagraphStyle(
        'RightCellStyle',
        parent=cell_style,
        alignment=TA_RIGHT
    )

    center_cell_style = ParagraphStyle(
        'CenterCellStyle',
        parent=cell_style,
        alignment=TA_CENTER
    )

    header_style = ParagraphStyle(
        'HeaderStyle',
        parent=styles['Normal'],
        fontSize=9,
        leading=11,
        textColor=colors.white,
        fontName='Helvetica-Bold',
        alignment=TA_CENTER
    )

    # --- Page 1: Cover / Visuals ---
    elements.append(Paragraph(f"Observation Plan Report - {period_str}", title_style))
    elements.append(Spacer(1, 0.5*cm))
    
    try:
        images_row = []
        if os.path.exists('altitude_vs_time.png'):
            img1 = Image('altitude_vs_time.png', width=13.5*cm, height=16*cm, kind='proportional')
            images_row.append(img1)
        else:
            images_row.append(Paragraph("Altitude Plot Missing", styles['Normal']))

        if os.path.exists('rotator_angle_vs_time.png'):
            img2 = Image('rotator_angle_vs_time.png', width=13.5*cm, height=16*cm, kind='proportional')
            images_row.append(img2)
        else:
            images_row.append(Paragraph("Rotator Plot Missing", styles['Normal']))
            
        if images_row:
            t_images = Table([images_row], colWidths=[13.8*cm, 13.8*cm])
            t_images.setStyle(TableStyle([
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('LEFTPADDING', (0,0), (-1,-1), 0),
                ('RIGHTPADDING', (0,0), (-1,-1), 0),
            ]))
            elements.append(t_images)
            
    except Exception as e:
        print(f"Warning: Could not add images to report: {e}")
        elements.append(Paragraph(f"Error loading images: {e}", styles['Normal']))

    elements.append(PageBreak())
    
    # --- Page 2: Additional Visuals ---
    elements.append(Paragraph("Sky Coverage & Statistics", title_style))
    elements.append(Spacer(1, 0.1*cm))
    
    try:
        if os.path.exists('sky_coverage.png'):
            img_sky = Image('sky_coverage.png', width=25*cm, height=8.5*cm, kind='proportional')
            elements.append(img_sky)
            elements.append(Spacer(1, 0.2*cm))
        else:
            elements.append(Paragraph("Sky Coverage Plot Missing", styles['Normal']))

        if os.path.exists('observation_counts.png'):
            img_counts = Image('observation_counts.png', width=27*cm, height=11.0*cm, kind='proportional')
            elements.append(img_counts)
        else:
            elements.append(Paragraph("Counts Plot Missing", styles['Normal']))
            
    except Exception as e:
        print(f"Warning: Could not add page 2 images to report: {e}")
        elements.append(Paragraph(f"Error loading images: {e}", styles['Normal']))

    elements.append(PageBreak())
    
    # --- Page 3+: Schedule ---
    # Group by night
    grouped = defaultdict(list)
    for row in schedule_list:
        grouped[row['night']].append(row)

    for night in sorted(grouped.keys()):
        group = grouped[night]
        first_time = Time(group[0]['start_time'])
        local_date_time = first_time - 22 * u.hour
        date_str = local_date_time.datetime.strftime('%Y-%m-%d')
        
        elements.append(Paragraph(f"Night {night} - {date_str} (HST)", title_style))
        elements.append(Spacer(1, 0.5*cm))
        
        headers = [
            Paragraph('Time (HST)', header_style),
            Paragraph('LST', header_style),
            Paragraph('Target', header_style),
            Paragraph('Air', header_style),
            Paragraph('Teff', header_style),
            Paragraph('El<br/>(Start)', header_style),
            Paragraph('El<br/>(End)', header_style),
            Paragraph('Rot<br/>(Start)', header_style),
            Paragraph('Rot<br/>(End)', header_style),
            Paragraph('Moon<br/>Sep', header_style),
            Paragraph('Moon<br/>Illum', header_style),
            Paragraph('Moon<br/>Alt', header_style)
        ]
        
        data = [headers]
        bg_commands = []
        warning_color = colors.Color(1, 0.85, 0.85)  # Light Red

        print(f"Processing Night {night}...")
        for row in group:
            t_start_utc = Time(row['start_time'])
            t_end_utc = Time(row['end_time'])
            
            t_start_hst = t_start_utc - 10 * u.hour
            t_end_hst = t_end_utc - 10 * u.hour
            
            t_fmt = f"{t_start_hst.datetime.strftime('%H:%M')} - {t_end_hst.datetime.strftime('%H:%M')}"
            
            lst_str = row['lst']
            teff_val = row['teff']
            teff_str = f"{teff_val:.2f}"
            
            airmass_val = row['airmass']
            airmass_str = f"{airmass_val:.2f}"
            
            elevation_start_val = row['elevation_start']
            elevation_start_str = f"{elevation_start_val:.1f}"
            
            elevation_end_val = row['elevation_end']
            elevation_end_str = f"{elevation_end_val:.1f}"
            
            rot_start_val = row['rot_start']
            rot_start_str = f"{rot_start_val:.1f}"
            
            rot_end_val = row['rot_end']
            rot_end_str = f"{rot_end_val:.1f}"
            
            moon_sep_val = row['moon_sep']
            moon_sep_str = f"{moon_sep_val:.1f}"
            
            moon_illum_val = row['moon_illum']
            moon_illum_str = f"{moon_illum_val:.2f}"
            
            moon_alt_val = row['moon_alt']
            moon_alt_str = f"{moon_alt_val:.1f}"
            
            row_num = len(data)

            # Check Constraints and Highlight
            if 'max_airmass' in constraints and airmass_val > constraints['max_airmass']:
                bg_commands.append(('BACKGROUND', (3, row_num), (3, row_num), warning_color))
            
            if 'min_teff' in constraints and teff_val < constraints['min_teff']:
                bg_commands.append(('BACKGROUND', (4, row_num), (4, row_num), warning_color))
            
            if 'min_altitude' in constraints and elevation_start_val < constraints['min_altitude']:
                bg_commands.append(('BACKGROUND', (5, row_num), (5, row_num), warning_color))
            elif 'max_altitude' in constraints and elevation_start_val > constraints['max_altitude']:
                bg_commands.append(('BACKGROUND', (5, row_num), (5, row_num), warning_color))
            
            if 'min_altitude' in constraints and elevation_end_val < constraints['min_altitude']:
                bg_commands.append(('BACKGROUND', (6, row_num), (6, row_num), warning_color))
            elif 'max_altitude' in constraints and elevation_end_val > constraints['max_altitude']:
                bg_commands.append(('BACKGROUND', (6, row_num), (6, row_num), warning_color))

            if 'rotator_min' in constraints and rot_start_val < constraints['rotator_min']:
                bg_commands.append(('BACKGROUND', (7, row_num), (7, row_num), warning_color))
            elif 'rotator_max' in constraints and rot_start_val > constraints['rotator_max']:
                bg_commands.append(('BACKGROUND', (7, row_num), (7, row_num), warning_color))

            if 'rotator_min' in constraints and rot_end_val < constraints['rotator_min']:
                bg_commands.append(('BACKGROUND', (8, row_num), (8, row_num), warning_color))
            elif 'rotator_max' in constraints and rot_end_val > constraints['rotator_max']:
                bg_commands.append(('BACKGROUND', (8, row_num), (8, row_num), warning_color))

            if 'min_moon_sep' in constraints and moon_sep_val < constraints['min_moon_sep']:
                bg_commands.append(('BACKGROUND', (9, row_num), (9, row_num), warning_color))

            if 'max_moon_ill' in constraints and moon_illum_val > constraints['max_moon_ill']:
                bg_commands.append(('BACKGROUND', (10, row_num), (10, row_num), warning_color))

            if 'max_moon_alt' in constraints and moon_alt_val > constraints['max_moon_alt']:
                bg_commands.append(('BACKGROUND', (11, row_num), (11, row_num), warning_color))

            target_p = Paragraph(str(row['target']), cell_style)
            
            data.append([
                Paragraph(t_fmt, center_cell_style),
                Paragraph(lst_str, right_cell_style),
                target_p,
                Paragraph(airmass_str, right_cell_style),
                Paragraph(teff_str, right_cell_style),
                Paragraph(elevation_start_str, right_cell_style),
                Paragraph(elevation_end_str, right_cell_style),
                Paragraph(rot_start_str, right_cell_style),
                Paragraph(rot_end_str, right_cell_style),
                Paragraph(moon_sep_str, right_cell_style),
                Paragraph(moon_illum_str, right_cell_style),
                Paragraph(moon_alt_str, right_cell_style)
            ])
            
        col_widths = [
            2.8*cm,  # Time
            1.7*cm,  # LST
            6.5*cm,  # Target
            1.2*cm,  # Air
            1.2*cm,  # Teff
            1.4*cm,  # El Start
            1.4*cm,  # El End
            1.8*cm,  # Rot Start
            1.8*cm,  # Rot End
            1.6*cm,  # Moon Sep
            1.6*cm,  # Moon Illum
            1.6*cm   # Moon Alt
        ]
        
        t = Table(data, colWidths=col_widths, repeatRows=1)
        
        base_style = [
            ('BACKGROUND', (0,0), (-1,0), colors.darkblue),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('ALIGN', (0,0), (-1,0), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (1, 1), (-1, -1), [colors.white, colors.whitesmoke]),
            ('LEFTPADDING', (0,0), (-1,-1), 3),
            ('RIGHTPADDING', (0,0), (-1,-1), 3),
            ('TOPPADDING', (0,0), (-1,-1), 2),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ]
        
        t.setStyle(TableStyle(base_style + bg_commands))
        
        elements.append(t)
        elements.append(PageBreak())
        
    print(f"Building PDF: {output_pdf}")
    try:
        doc.build(elements)
        print("Done.")
    except Exception as e:
        print(f"Error building PDF: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate PDF report from SA Scheduler results")
    parser.add_argument("-s", "--schedule", type=str, default="schedule_result.json", help="Path to schedule JSON result file (default: schedule_result.json)")
    parser.add_argument("-v", "--vis-map", type=str, default="vis_map.npz", help="Path to visibility map NPZ file (default: vis_map.npz)")
    parser.add_argument("-o", "--output-pdf", type=str, default=None, help="Path to output PDF file (default: auto-generated name)")
    args = parser.parse_args()
    
    if not os.path.exists(args.schedule):
        print(f"Error: Schedule result file '{args.schedule}' not found.")
        return
        
    if not os.path.exists(args.vis_map):
        print(f"Error: Visibility map file '{args.vis_map}' not found.")
        return

    create_pdf_report(args.schedule, args.vis_map, args.output_pdf)

if __name__ == '__main__':
    main()
