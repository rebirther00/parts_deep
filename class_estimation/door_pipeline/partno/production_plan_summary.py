"""생산계획(9-10월_통합모델_일별생산계획.xlsx) 월별 집계 — 기종 그룹 × 레이더 유무 → 도어 클래스별 기대 수량 (집계 대조 전용, 세션 1:1 매칭 아님).

  python partno/production_plan_summary.py                 # 표 출력 + partno/production_plan_summary.md 저장
  python partno/production_plan_summary.py --xlsx 경로 --db db/door_pipeline.db

읽는 시트: '상세데이터' (개체 단위: 라인·모델·그룹·생산번호·호기·국가·수량·L/ON 일자·W.A 제작일).
레이더 표기 해석(사용자 결정 2026-09-23, 레이더만 구분·AVM 무시):
  모델명에 '레이더 X' → 미장착(0), '확인하기' → 미정(None), 그 외(무표기) → 장착(1).
개체 1대 = 도어 3장: LH FRT(그룹별 클래스) + LH REAR(레이더 옵션) + RH(레이더 옵션). E23/E25 는 REAR·RH 공용, E30/E35/E38 은 RH 공용.
날짜는 고객사 L/ON 기준이며 도어 제작 시점과 1주 이상 차이·추석 등으로 어긋나므로 월 단위 비율만 쓴다.
openpyxl 없이 xlsx(zip+xml)를 직접 읽는다.
"""
import argparse, collections, datetime, html, os, re, sqlite3, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
DOOR = os.path.dirname(HERE)
DEFAULT_XLSX = os.path.normpath(os.path.join(DOOR, '..', '..', 'cad', '9-10월_통합모델_일별생산계획.xlsx'))
GROUP_CLASSES = {  # 그룹 → (FRT 클래스, LH_RR 클래스, RH 클래스)
    'E23': ('E23_door_LH_FRT', 'E25_door_LH_RR', 'E25_door_RH'),
    'E25': ('E25_door_LH_FRT', 'E25_door_LH_RR', 'E25_door_RH'),
    'E30': ('E30_door_LH_FRT', 'E30_door_LH_RR', 'E30_E38_door_RH'),
    'E35/38': ('E38_door_LH_FRT', 'E38_door_LH_RR', 'E30_E38_door_RH'),
}


def xl_date(serial):
    return datetime.date(1899, 12, 30) + datetime.timedelta(days=int(float(serial)))


def read_units(xlsx):
    z = zipfile.ZipFile(xlsx)
    ss = [html.unescape(''.join(re.findall(r'<t[^>]*>(.*?)</t>', si, re.S)))
          for si in re.findall(r'<si>(.*?)</si>', z.read('xl/sharedStrings.xml').decode('utf8'), re.S)]
    wb = z.read('xl/workbook.xml').decode('utf8')
    rels = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="([^"]+)"', z.read('xl/_rels/workbook.xml.rels').decode('utf8')))
    sheet = next(rels[rid] for name, rid in re.findall(r'<sheet name="([^"]+)"[^>]*r:id="(rId\d+)"', wb) if name == '상세데이터')
    xml = z.read('xl/' + sheet.lstrip('/').replace('xl/', '')).decode('utf8')
    hdr, units = None, []
    for rn, row in re.findall(r'<row [^>]*r="(\d+)"[^>]*>(.*?)</row>', xml, re.S):
        rec = {}
        for m in re.finditer(r'<c r="([A-Z]+)\d+"([^>]*?)(?:/>|>(.*?)</c>)', row, re.S):
            col, attrs, inner = m.groups()
            if inner is None: continue
            v = re.search(r'<v>(.*?)</v>', inner, re.S); t = re.search(r't="(\w+)"', attrs)
            if v: rec[col] = ss[int(v.group(1))] if (t and t.group(1) == 's') else v.group(1)
        if int(rn) == 1: hdr = rec; continue
        if 'C' in rec and 'D' in rec: units.append(rec)
    return hdr, units


def radar_flag(model):
    m = model.replace(' ', '')
    if '레이더X' in m: return 0
    if '확인' in m: return None
    return 1


def summarize(units):
    """월 × 그룹 → dict(units, radar={1,0,None}); 월 × 클래스 × radar → 도어 수."""
    by_grp = collections.defaultdict(lambda: dict(units=0, radar=collections.Counter()))
    by_cls = collections.Counter()
    for u in units:
        mon = xl_date(u['I']).strftime('%Y-%m'); g = u['D']; rf = radar_flag(u['C']); q = int(float(u.get('H', 1) or 1))
        by_grp[(mon, g)]['units'] += q; by_grp[(mon, g)]['radar'][rf] += q
        frt, rr, rh = GROUP_CLASSES[g]
        by_cls[(mon, frt, None)] += q; by_cls[(mon, rr, rf)] += q; by_cls[(mon, rh, rf)] += q
    return by_grp, by_cls


def field_counts(db):
    """DB 세션 수: 클래스 × option_radar(확정값, 없으면 NULL) — 월별(session_dir 날짜)."""
    if not db or not os.path.exists(db): return {}
    con = sqlite3.connect(db)
    try:
        rows = con.execute("""SELECT substr(s.session_dir,1,6) ym, c.name, s.option_radar, COUNT(DISTINCT s.id)
                              FROM capture_sessions s JOIN images i ON i.session_id=s.id JOIN classes c ON c.id=i.class_id
                              WHERE i.is_valid=1 AND c.name!='Unknown' GROUP BY ym, c.name, s.option_radar""").fetchall()
    except sqlite3.OperationalError:
        return {}
    return {(f"{ym[:4]}-{ym[4:]}", cls, rf): n for ym, cls, rf, n in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xlsx', default=DEFAULT_XLSX)
    ap.add_argument('--db', default=os.path.join(DOOR, 'db', 'door_pipeline.db'))
    ap.add_argument('--out', default=os.path.join(HERE, 'production_plan_summary.md'))
    a = ap.parse_args()
    hdr, units = read_units(a.xlsx)
    by_grp, by_cls = summarize(units)
    fc = field_counts(a.db)
    months = sorted({k[0] for k in by_grp})
    lines = [f"# 생산계획 월별 집계 (출처: {os.path.basename(a.xlsx)}, 개체 {len(units)}대, L/ON 일자 기준)", "",
             "레이더 표기 해석: '레이더 X'=미장착, '확인하기'=미정, 무표기=장착 (AVM 은 구분하지 않음). 개체 1대 = 도어 3장.", "",
             "## 기종 그룹 × 레이더 (개체 수)", "",
             "| 월 | 그룹 | 개체 | 레이더 장착 | 미장착 | 미정 | 장착 비율 |", "|---|---|---:|---:|---:|---:|---:|"]
    for mon in months:
        for g in GROUP_CLASSES:
            d = by_grp.get((mon, g))
            if not d: continue
            r1, r0, rn = d['radar'][1], d['radar'][0], d['radar'][None]
            lines.append(f"| {mon} | {g} | {d['units']} | {r1} | {r0} | {rn} | {100 * r1 / max(1, r1 + r0):.0f}% |")
    lines += ["", "## 도어 클래스 × 레이더 — 계획 도어 수 vs 현장 세션 수(DB 확정 옵션 기준)", "",
              "현장 세션은 카메라가 잡은 일부(주별 계획 도어의 40~60%)이고 날짜가 어긋나므로 비율만 비교한다.", "",
              "| 월 | 클래스 | 레이더 | 계획 도어 | 계획 비율 | 현장 세션 | 현장 비율 |", "|---|---|---|---:|---:|---:|---:|"]
    for mon in months:
        for cls in sorted({k[1] for k in by_cls if k[0] == mon}):
            tot = sum(v for k, v in by_cls.items() if k[0] == mon and k[1] == cls)
            ftot = sum(v for k, v in fc.items() if k[0] == mon and k[1] == cls)
            for rf in (None, 1, 0):
                n = by_cls.get((mon, cls, rf), 0)
                if not n and rf is not None: continue
                if rf is None and 'FRT' not in cls and not n: continue
                f = fc.get((mon, cls, rf), 0)
                lab = {None: '해당없음/미정', 1: '장착', 0: '미장착'}[rf]
                lines.append(f"| {mon} | {cls} | {lab} | {n} | {100 * n / max(1, tot):.0f}% | {f} | {100 * f / max(1, ftot):.0f}% |")
    open(a.out, 'w', encoding='utf8').write('\n'.join(lines) + '\n')
    print('\n'.join(lines)); print(f"\n→ {a.out}")


if __name__ == '__main__':
    main()
