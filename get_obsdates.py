#!/usr/bin/env python3
import sys
import re
import datetime
import urllib.request
import urllib.parse
from html.parser import HTMLParser
from obs_utils import load_config, setup_observer, parse_time

# Month abbreviation mapping for NAOJ schedule CGI
MONTH_MAP = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}

VOID_TAGS = {"br", "img", "hr", "meta", "link", "input", "p"}

class TagNode:
    """
    DOM Node representing an HTML element or text fragment.
    """
    def __init__(self, tag, attrs=None):
        self.tag = tag.lower()
        self.attrs = dict(attrs) if attrs else {}
        self.children = []

    def text(self):
        parts = []
        for child in self.children:
            if isinstance(child, str):
                parts.append(child)
            else:
                parts.append(child.text())
        return " ".join("".join(parts).split())

    def find_all(self, tag_name):
        results = []
        target = tag_name.lower()
        for child in self.children:
            if isinstance(child, TagNode):
                if child.tag == target:
                    results.append(child)
                results.extend(child.find_all(target))
        return results

    def find(self, tag_name):
        target = tag_name.lower()
        for child in self.children:
            if isinstance(child, TagNode):
                if child.tag == target:
                    return child
                found = child.find(target)
                if found is not None:
                    return found
        return None


class ScheduleDOMParser(HTMLParser):
    """
    Robust DOM tree parser for HTML containing tables, nested tables,
    and unclosed or implicit tags from CGI outputs.
    """
    def __init__(self):
        super().__init__()
        self.root = TagNode("root")
        self.stack = [self.root]

    def current_node(self):
        return self.stack[-1] if self.stack else self.root

    def close_tags_up_to(self, tag_names):
        while len(self.stack) > 1 and self.stack[-1].tag in tag_names:
            self.stack.pop()

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "tr":
            self.close_tags_up_to(["td", "th", "tr", "font", "b", "i", "a", "center", "span"])
        elif tag in ["td", "th"]:
            self.close_tags_up_to(["td", "th", "font", "b", "i", "a", "center", "span"])

        node = TagNode(tag, attrs)
        self.current_node().children.append(node)
        if tag not in VOID_TAGS:
            self.stack.append(node)

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in VOID_TAGS:
            return
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                self.stack = self.stack[:i]
                break

    def handle_data(self, data):
        if self.stack:
            self.stack[-1].children.append(data)


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


def extract_ssp_slot(sub_texts):
    """
    Given sub-allocations for a day (in chronological order from top to bottom),
    determine the SSP PFS slot (whole, first, second, first_<frac>, second_<frac>).
    """
    if not sub_texts:
        return None

    # Handle dash separated text if single string
    if len(sub_texts) == 1 and re.search(r'-{3,}', sub_texts[0]):
        sub_texts = [p.strip() for p in re.split(r'-{3,}', sub_texts[0]) if p.strip()]

    ssp_indices = []
    ssp_fracs = []
    
    for i, raw_text in enumerate(sub_texts):
        text = raw_text.upper()
        if "SSP" in text and "PFS" in text:
            m = re.search(r'SSP\s*\(\s*([\d\.]+)\s*\)', raw_text, re.IGNORECASE)
            frac = float(m.group(1)) if m else None
            ssp_indices.append(i)
            ssp_fracs.append(frac)

    if not ssp_indices:
        return None

    n = len(sub_texts)
    
    # If all slots are SSP PFS, or only 1 slot with no fraction or frac=1.0
    if len(ssp_indices) == n or (n == 1 and (ssp_fracs[0] is None or ssp_fracs[0] == 1.0)):
        return "whole"

    # If single SSP slot among multiple
    if len(ssp_indices) == 1:
        idx = ssp_indices[0]
        frac = ssp_fracs[0]
        if frac is None:
            frac = 0.5 if n == 2 else (1.0 / n)

        if frac == 1.0:
            return "whole"
        if idx == 0:
            return "first" if frac == 0.5 else f"first_{frac}"
        elif idx == n - 1:
            return "second" if frac == 0.5 else f"second_{frac}"
        else:
            return f"second_{frac}"

    total_frac = sum(f if f is not None else (1.0 / n) for f in ssp_fracs)
    if ssp_indices == [0]:
        return f"first_{total_frac}"
    else:
        return f"second_{total_frac}"


def parse_html_schedule(html_content):
    parser = ScheduleDOMParser()
    parser.feed(html_content)

    # Find the main schedule table containing weekday headers
    all_tables = parser.root.find_all("table")
    main_table = None
    for tbl in all_tables:
        header_ths = tbl.find_all("th")
        th_texts = [th.text() for th in header_ths]
        if any("Sun" in t for t in th_texts) and any("Mon" in t for t in th_texts):
            main_table = tbl
            break

    if not main_table:
        return []

    # Extract direct rows of main_table (including tbody wrapper if any)
    direct_trs = []
    for child in main_table.children:
        if isinstance(child, TagNode):
            if child.tag == "tr":
                direct_trs.append(child)
            elif child.tag == "tbody":
                for sub in child.children:
                    if isinstance(sub, TagNode) and sub.tag == "tr":
                        direct_trs.append(sub)

    month_pattern = re.compile(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d+)', re.IGNORECASE)
    
    weeks = []
    current_header = None

    for tr in direct_trs:
        cells = [c for c in tr.children if isinstance(c, TagNode) and c.tag in ["th", "td"]]
        if not cells:
            continue

        # Check if this row is the top weekday header (Sun, Mon, ...)
        cell_texts = [c.text().strip() for c in cells]
        if any(w in cell_texts for w in ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]):
            continue

        # Check if this row is a date header row (Sep 01, Sep 02...)
        has_dates = False
        row_days = [None] * 7
        col = 0
        for cell in cells:
            colspan = int(cell.attrs.get("colspan", 1))
            text = cell.text().strip()
            match = month_pattern.search(text)
            if match:
                has_dates = True
                day_num = int(match.group(2))
                if col < 7:
                    row_days[col] = day_num
            col += colspan

        if has_dates:
            current_header = row_days
        elif current_header is not None:
            # This is the assignment row following the date header row
            weeks.append({
                "days": current_header,
                "row_node": tr
            })
            current_header = None

    results = []
    for week in weeks:
        days = week["days"]
        tr_node = week["row_node"]
        cells = [c for c in tr_node.children if isinstance(c, TagNode) and c.tag in ["td", "th"]]
        
        col_idx = 0
        for cell in cells:
            colspan = int(cell.attrs.get("colspan", 1))
            nested_table = cell.find("table")
            if nested_table:
                sub_tds = nested_table.find_all("td")
                sub_texts = [std.text() for std in sub_tds]
            else:
                sub_texts = [cell.text()]

            ssp_slot = extract_ssp_slot(sub_texts)

            for offset in range(colspan):
                c = col_idx + offset
                if c < 7 and days[c] is not None and ssp_slot is not None:
                    results.append((days[c], ssp_slot))

            col_idx += colspan

    return sorted(results, key=lambda x: x[0])


def parse_obstime_table(html_content):
    parser = ObstimeHTMLParser()
    parser.feed(html_content)
    return parser.rows


def lookup_obstime(dt, obstime_rows):
    for row in obstime_rows:
        if not row:
            continue
        period = row[0]
        match = re.match(r'(\d+)/(\d+)-(\d+)/(\d+)', period)
        if match:
            start_m, start_d, end_m, end_d = map(int, match.groups())
            if start_m <= end_m:
                start_date = datetime.date(dt.year, start_m, start_d)
                end_date = datetime.date(dt.year, end_m, end_d)
                if start_date <= dt <= end_date:
                    return row
            else:
                # Crosses new year
                start_date_prev = datetime.date(dt.year - 1, start_m, start_d)
                end_date_curr = datetime.date(dt.year, end_m, end_d)
                start_date_curr = datetime.date(dt.year, start_m, start_d)
                end_date_next = datetime.date(dt.year + 1, end_m, end_d)
                if (start_date_prev <= dt <= end_date_curr) or (start_date_curr <= dt <= end_date_next):
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
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
    try:
        req_obstime = urllib.request.Request(obstime_url, headers=headers)
        with urllib.request.urlopen(req_obstime, timeout=15) as response:
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
    try:
        req = urllib.request.Request(schedule_url, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as response:
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
    config = load_config()
    split_margin = config['scheduling']['split_margin_minutes']
    print(f"Using split margin of {split_margin} minutes from configuration.")
    
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
        # - First half end = split_time - split_margin minutes (formatted 24h+)
        # - Second half start = split_time + split_margin minutes (formatted 24h+)
        if slot == "whole":
            start_val = "twilight_end"
            end_val = "twilight_beg"
        elif slot == "first":
            start_val = "twilight_end"
            end_val = format_time_24h_plus(add_minutes_to_time_str(t_split, -split_margin))
        elif slot == "second":
            start_val = format_time_24h_plus(add_minutes_to_time_str(t_split, split_margin))
            end_val = "twilight_beg"
        elif slot.startswith("first_") or slot.startswith("second_"):
            frac = float(slot.split("_")[1])
            observer = setup_observer()
            t_end_utc = parse_time(date_str, 'twilight_end', observer).datetime
            t_beg_utc = parse_time(date_str, 'twilight_beg', observer).datetime
            duration = t_beg_utc - t_end_utc
            
            if slot.startswith("first_"):
                split_time_utc = t_end_utc + duration * frac
                split_time_hst = split_time_utc + datetime.timedelta(hours=-10) - datetime.timedelta(minutes=split_margin)
                h = split_time_hst.hour
                if h < 12: h += 24
                start_val = "twilight_end"
                end_val = f"{h:02d}:{split_time_hst.minute:02d}"
            else:
                split_time_utc = t_end_utc + duration * (1.0 - frac)
                split_time_hst = split_time_utc + datetime.timedelta(hours=-10) + datetime.timedelta(minutes=split_margin)
                h = split_time_hst.hour
                if h < 12: h += 24
                start_val = f"{h:02d}:{split_time_hst.minute:02d}"
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
