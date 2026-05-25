#!/usr/bin/env python3
import sys
import re
import datetime
import urllib.request
import urllib.parse
from html.parser import HTMLParser

# Month abbreviation mapping for NAOJ schedule CGI
MONTH_MAP = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}

class ScheduleHTMLParser(HTMLParser):
    """
    Robust parser for the Subaru telescope schedule CGI table.
    Handles implicit row/cell closures due to sloppy CGI output.
    """
    def __init__(self):
        super().__init__()
        self.in_table = False
        self.current_row = []
        self.current_cell = None
        self.rows = []
        
    def close_cell(self):
        if self.current_cell:
            self.current_row.append(self.current_cell)
            self.current_cell = None

    def close_row(self):
        self.close_cell()
        if self.current_row:
            self.rows.append(self.current_row)
            self.current_row = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == "table":
            if "border" in attrs_dict and "cellpadding" in attrs_dict:
                self.in_table = True
                
        if self.in_table:
            if tag == "tr":
                self.close_row()
            elif tag in ["th", "td"]:
                self.close_cell()
                self.current_cell = {
                    "tag": tag,
                    "attrs": attrs_dict,
                    "text": ""
                }
                
    def handle_endtag(self, tag):
        if self.in_table:
            if tag == "table":
                self.close_row()
                self.in_table = False
            elif tag == "tr":
                self.close_row()
            elif tag in ["th", "td"]:
                self.close_cell()
                    
    def handle_data(self, data):
        if self.in_table and self.current_cell:
            self.current_cell["text"] += data


class ObstimeHTMLParser(HTMLParser):
    """
    Parser for the Observing Time Table in def_obstime.html.
    """
    def __init__(self):
        super().__init__()
        self.in_table = False
        self.in_tbody = False
        self.current_row = []
        self.current_cell = None
        self.rows = []
        
    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.in_table = True
        elif tag == "tbody" and self.in_table:
            self.in_tbody = True
        elif tag == "tr" and self.in_tbody:
            self.current_row = []
        elif tag in ["td"] and self.in_tbody:
            self.current_cell = ""
            
    def handle_endtag(self, tag):
        if tag == "table":
            self.in_table = False
            self.in_tbody = False
        elif tag == "tbody":
            self.in_tbody = False
        elif tag == "tr" and self.in_tbody:
            if self.current_row:
                self.rows.append(self.current_row)
                self.current_row = []
        elif tag in ["td"] and self.in_tbody:
            if self.current_cell is not None:
                self.current_row.append(self.current_cell.strip())
                self.current_cell = None
            
    def handle_data(self, data):
        if self.in_tbody and self.current_cell is not None:
            self.current_cell += data


def normalize_month(month_str):
    month_str = month_str.strip()
    if month_str.isdigit():
        m_num = int(month_str)
        if 1 <= m_num <= 12:
            return MONTH_MAP[m_num]
    else:
        m_lower = month_str.lower()
        for k, v in MONTH_MAP.items():
            if v.lower() == m_lower[:3]:
                return v
    raise ValueError(f"Invalid month: {month_str}")


def get_month_num(month_name):
    for k, v in MONTH_MAP.items():
        if v.lower() == month_name.lower():
            return k
    raise ValueError(f"Invalid month name: {month_name}")


def parse_html_schedule(html_content):
    parser = ScheduleHTMLParser()
    parser.feed(html_content)
    
    rows = parser.rows
    weeks = []
    current_week = None
    month_pattern = re.compile(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d+)', re.IGNORECASE)
    
    for row in rows:
        if not row:
            continue
        
        # Check if this is a header row for a week
        is_header = False
        row_days = [None] * 7
        for i, cell in enumerate(row[:7]):
            text = cell["text"].strip()
            match = month_pattern.search(text)
            if match:
                is_header = True
                row_days[i] = int(match.group(2))
                
        if is_header:
            if current_week:
                weeks.append(current_week)
            current_week = {
                "days": row_days,
                "assignments": []
            }
        elif current_week:
            # Check if this is a weekday header row (Sun, Mon...)
            first_cell_text = row[0]["text"].strip()
            if first_cell_text in ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]:
                continue
            current_week["assignments"].append(row)
            
    if current_week:
        weeks.append(current_week)
        
    results = []
    for week in weeks:
        days = week["days"]
        assign_rows = week["assignments"]
        n_assign = len(assign_rows)
        if n_assign == 0:
            continue
            
        grid_rows = max(n_assign, 2)
        grid = [[None] * 7 for _ in range(grid_rows)]
        
        for r_idx, row in enumerate(assign_rows):
            c_idx = 0
            for cell in row:
                while c_idx < 7 and grid[r_idx][c_idx] is not None:
                    c_idx += 1
                if c_idx >= 7:
                    break
                    
                colspan = int(cell["attrs"].get("colspan", 1))
                rowspan = int(cell["attrs"].get("rowspan", 1))
                
                for dr in range(rowspan):
                    for dc in range(colspan):
                        if r_idx + dr < grid_rows and c_idx + dc < 7:
                            grid[r_idx + dr][c_idx + dc] = cell
                c_idx += colspan
                
        for col in range(7):
            day = days[col]
            if day is None:
                continue
                
            first_half_cell = grid[0][col]
            second_half_cell = grid[1][col] if grid_rows > 1 else None
            
            def is_ssp_pfs(cell):
                if not cell:
                    return False
                text = cell["text"].strip().upper()
                text_clean = "".join(text.split())
                return "SSPPFS" in text_clean
                
            first_ssp = is_ssp_pfs(first_half_cell)
            second_ssp = is_ssp_pfs(second_half_cell)
            
            is_whole_night = False
            if first_half_cell and second_half_cell and first_half_cell == second_half_cell:
                is_whole_night = True
            elif n_assign == 1:
                is_whole_night = True
                
            if is_whole_night:
                if first_ssp:
                    results.append((day, "whole"))
            else:
                if first_ssp:
                    results.append((day, "first"))
                if second_ssp:
                    results.append((day, "second"))
                    
    return results


def parse_obstime_table(html_content):
    parser = ObstimeHTMLParser()
    parser.feed(html_content)
    return parser.rows


def lookup_obstime(dt, obstime_rows):
    for row in obstime_rows:
        period = row[0]
        match = re.match(r'(\d+)/(\d+)-(\d+)/(\d+)', period)
        if match:
            start_m = int(match.group(1))
            start_d = int(match.group(2))
            end_m = int(match.group(3))
            end_d = int(match.group(4))
            
            start_date = datetime.date(dt.year, start_m, start_d)
            end_date = datetime.date(dt.year, end_m, end_d)
            
            if start_date <= dt <= end_date:
                return row
    return None


def add_minutes_to_time_str(time_str, minutes_to_add):
    h, m = map(int, time_str.split(":"))
    total_minutes = h * 60 + m + minutes_to_add
    if total_minutes < 0:
        total_minutes += 1440
    new_h = total_minutes // 60
    new_m = total_minutes % 60
    return f"{new_h:02d}:{new_m:02d}"


def format_time_24h_plus(time_str, ref_start_hour=12):
    h, m = map(int, time_str.split(":"))
    if h < ref_start_hour:
        h += 24
    return f"{h:02d}:{m:02d}"


def main():
    # 1. Parse command line arguments
    if len(sys.argv) >= 3:
        year_str = sys.argv[1]
        month_str = sys.argv[2]
    else:
        print("Usage: python3 get_obsdates.py <year> <month>")
        print("Example: python3 get_obsdates.py 2026 May")
        # Interactive fallback
        try:
            year_str = input("Enter Year (e.g. 2026): ").strip()
            month_str = input("Enter Month (e.g. May or 5): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAborted.")
            sys.exit(1)
            
    try:
        year = int(year_str)
        month = normalize_month(month_str)
        month_num = get_month_num(month)
    except Exception as e:
        print(f"Error parsing input parameters: {e}")
        sys.exit(1)
        
    print(f"Targeting: Year={year}, Month={month} ({month_num:02d})")
    
    # 2. Fetch Observing Time Table definitions from Subaru website
    print("Fetching observing time definitions from NAOJ website...")
    obstime_url = "https://www.naoj.org/Observing/def_obstime.html"
    try:
        with urllib.request.urlopen(obstime_url) as response:
            obstime_html = response.read().decode("utf-8")
        obstime_rows = parse_obstime_table(obstime_html)
        print(f"Successfully loaded {len(obstime_rows)} observing time ranges.")
    except Exception as e:
        print(f"Warning: Failed to fetch observing time table: {e}")
        print("Will fall back to standard May 2026 constants.")
        obstime_rows = []

    # 3. Fetch schedule from NAOJ CGI
    print("Fetching telescope schedule from NAOJ CGI...")
    schedule_url = "https://www.naoj.org/cgi-bin/opecenter/schedule.cgi"
    data = urllib.parse.urlencode({"year": str(year), "month": month}).encode("utf-8")
    req = urllib.request.Request(schedule_url, data=data, method="POST")
    try:
        with urllib.request.urlopen(req) as response:
            schedule_html = response.read().decode("iso-8859-1")
    except Exception as e:
        print(f"Error fetching schedule: {e}")
        sys.exit(1)
        
    # 4. Parse schedule
    print("Parsing schedule table...")
    allocations = parse_html_schedule(schedule_html)
    if not allocations:
        print(f"No SSP PFS allocations found for {year} {month}.")
        sys.exit(0)
        
    print(f"Found {len(allocations)} SSP PFS allocations:")
    for day, slot in allocations:
        print(f"  Day {day:02d}: {slot} night")
        
    # 5. Format and compute times for output
    output_lines = []
    output_lines.append(f"{'date':<11}{'start':<15}{'end'}")
    
    for day, slot in sorted(allocations):
        dt = datetime.date(year, month_num, day)
        date_str = dt.strftime("%Y-%m-%d")
        
        # Determine standard times from obstime rows or fallback
        obstime_row = lookup_obstime(dt, obstime_rows)
        if obstime_row:
            # e.g., ['5/13-5/25', '19:40', '5:00', '560', '9:20', '0:20']
            t_start = obstime_row[1]
            t_end = obstime_row[2]
            t_split = obstime_row[5]
        else:
            # Fallback to May 2026 values
            t_start = "19:40"
            t_end = "5:00"
            t_split = "0:20"
            
        # Apply offset rules:
        # - First half end = split_time - 5 minutes (formatted 24h+)
        # - Second half start = split_time + 5 minutes (formatted 24h+)
        if slot == "whole":
            start_val = "twilight_end"
            end_val = "twilight_beg"
        elif slot == "first":
            start_val = "twilight_end"
            end_val = format_time_24h_plus(add_minutes_to_time_str(t_split, -5))
        elif slot == "second":
            start_val = format_time_24h_plus(add_minutes_to_time_str(t_split, 5))
            end_val = "twilight_beg"
            
        output_lines.append(f"{date_str:<11}{start_val:<15}{end_val}")
        
    # 6. Save to output file
    output_filename = f"obsdates_{year}{month}.txt"
    try:
        with open(output_filename, "w") as f:
            f.write("\n".join(output_lines) + "\n")
        print(f"Output successfully written to: {output_filename}")
        
        # Print content for validation
        print("\n--- Output file content ---")
        print("\n".join(output_lines))
        print("---------------------------")
    except Exception as e:
        print(f"Error writing output file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
