"""B1 of config/extension_v16_protocol.json: machine-level production plan against the sessions (descriptive only).

Run with the system python (openpyxl):  python3 scripts/production_plan_check_v16.py
The plan lists excavator units by model group with a welding-assembly (W.A) date; it has no door-level or session-level
record, so it cannot confirm a single session. E25 units are rare (10 in two months), which makes their dates comparable
with the dates of the E25 LH FRT sessions.
"""
import collections
import csv
import datetime as dt
import glob
import hashlib
import json
from pathlib import Path
import site
import sys

sys.path.append(site.getusersitepackages())
import openpyxl

ROOT = Path(__file__).resolve().parents[1]; REPO = ROOT.parents[1]
PLAN = glob.glob(str(REPO / 'cad' / '*일별생산계획.xlsx'))[0]
TARGETS = ['20260903/E30_door_LH_RR/s_151736', '20260903/E30_door_LH_RR/s_183159', '20260905/E30_door_LH_RR/s_123451',
           '20260914/E30_door_LH_RR/s_170035', '20261029/E30_door_LH_FRT/s_075858', '20261022/E25_door_LH_FRT/s_084658',
           '20261022/E25_door_LH_FRT/s_123253']

if __name__ == '__main__':
    ws = openpyxl.load_workbook(PLAN, data_only=True, read_only=True)['상세데이터']
    units = [(str(r[9])[:10].replace('-', ''), r[3]) for r in list(ws.iter_rows(values_only=True))[1:] if r[3]]
    per_day = collections.Counter(units); groups = sorted({g for _, g in units})
    sess = list(csv.DictReader(open(ROOT / 'data/snapshots/v1/sessions.csv')))
    day = lambda s: dt.datetime.strptime(s, '%Y%m%d').date()
    e25_units = sorted(d for d, g in units if g == 'E25'); span = (min(d for d, _ in units), max(d for d, _ in units))
    e25_sessions = sorted(s['session_dir'] for s in sess if s['class_name'] == 'E25_door_LH_FRT')
    match = []
    for s in e25_sessions:
        d = s[:8]; diff = min(abs((day(d) - day(u)).days) for u in e25_units)
        match.append(dict(session=s, in_plan_span=span[0] <= d <= span[1], days_to_nearest_E25_unit=diff))
    inspan = [m for m in match if m['in_plan_span']]
    out = dict(created_at=dt.datetime.now().astimezone().isoformat(timespec='seconds'),
               protocol_sha256=hashlib.sha256((ROOT / 'config/extension_v16_protocol.json').read_bytes()).hexdigest(),
               plan=Path(PLAN).name, plan_sha256=hashlib.sha256(Path(PLAN).read_bytes()).hexdigest(), plan_span_wa_dates=span,
               units_by_group=dict(collections.Counter(g for _, g in units)), e25_unit_wa_dates=e25_units, e25_lh_frt_sessions=match,
               e25_lh_frt_summary=dict(sessions=len(match), in_plan_span=len(inspan), within_one_day=sum(m['days_to_nearest_E25_unit'] <= 1 for m in inspan)),
               target_dates={t: dict(in_plan_span=span[0] <= t[:8] <= span[1], units_on_date={g: per_day[(t[:8], g)] for g in groups}) for t in TARGETS},
               limits='Machine-level plan with planned dates; one unit needs one door of each position and E23 units use E25 rear and RH doors. A count on a date cannot tell which class a given session shows.')
    dst = ROOT / 'reports/extension_v16/production_plan_check.json'
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: out[k] for k in ('plan_span_wa_dates', 'units_by_group', 'e25_unit_wa_dates', 'e25_lh_frt_summary', 'target_dates')}, ensure_ascii=False, indent=1))
    print([(m['session'][:8], m['days_to_nearest_E25_unit']) for m in match])
