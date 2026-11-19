"""합쳐진 추론 세션을 도어 단위로 분할 복원 (2026-11-19).

배경: 07_factory_infer 0.2.1 까지 빈 지그를 못 알아봐 도어 여러 대가 세션 하나에 합쳐진 경우가 있다(11/19 s_070136 = 8대,
DOC_factory_infer.md '비움 판정' 절). 세션 폴더에는 마지막으로 판정이 바뀐 순간의 프레임 3~5장만 남지만, 추론 PC 의
초 단위 지표 로그 metrics/metrics_YYYYMMDD.jsonl(1Hz: 지그 중앙 깊이 z_occ·홀 점 수 pts·프레임별 판정 pred·D, 14일 보존)에는
도어가 바뀐 순간이 남아 있다. 이 로그에 비움 규칙을 다시 돌려 도어 구간을 복원한다.

    python db/split_infer_sessions.py pull                 # NAS → datasets_factory_infer/{metrics/, events.jsonl} (rclone)
    python db/split_infer_sessions.py scan                 # 분할 후보 표 + db/infer_splits.json(계획) 작성. DB 는 읽기만
    python db/split_infer_sessions.py apply [--only DIR…]  # 계획을 DB 에 반영 (멱등). 먼저 DB 를 백업할 것
    python db/split_infer_sessions.py undo DIR             # 분할 취소: 복원 세션 삭제, 원 세션 제외 해제

분할 규칙
  - 비움(오프라인 복원용, 20초 지속): 홀 2점 미만이고 ①세로 바 벌린 설정 z_occ ≥ 1481  또는
    ②세로 바 모은 설정(E23 작업) z_occ 1464~1476 · ROI 중앙값 z ≥ 1466 · 정지.  ②는 운영 서비스(0.2.2)에는 아직 없는 규칙이다.
  - 구간 판정 = 그 구간에서 기하 게이트를 통과한(gate ok) 프레임의 pred 다수결. 통과 프레임 30장 미만·득표 80% 미만 구간은 '판정 없음'.
  - 판정 있는 구간이 2개 이상인 세션만 분할한다. 판정 없는 구간(도어 올리기 전·홀 가공 전 등)은 세션으로 만들지 않고 계획에만 남긴다.
DB 반영
  - 도어마다 새 세션 `<원 세션>_pN` (capture_sessions + inference_results, split_parent = 원 세션 id). NAS 에는 없는 DB 전용 레코드다
    (ingest_infer 는 NAS 에 있는 세션만 갱신하므로 건드리지 않는다). 계획 파일 db/infer_splits.json 이 재현 근거 — DB 를 NAS 에서 다시 만들면
    ingest_infer → 이 도구 apply 순으로 되살린다.
  - 원 세션은 지우지 않고 제외 표시(excluded_by='split'). 원 세션에 남아 있던 원본 프레임(images)은 촬영 시각이 속한 복원 세션으로 옮긴다.
  - 레이더 옵션은 초 단위 로그에 없어, 합쳐진 세션이 그 시각에 표시하던 값(result.json pred_history)을 참고값으로만 넣는다.
  - 같은 도어를 내렸다 다시 올린 경우도 2대로 나뉜다 — webapp 타임랩스로 확인해 아니면 한쪽을 제외한다.
"""
import argparse
import collections
import datetime as dt
import json
import shutil
import sqlite3
import statistics
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "db"))
from migrate_inference import migrate  # noqa: E402

DB = BASE_DIR / "db" / "door_pipeline.db"
MIRROR = BASE_DIR / "datasets_factory_infer"
METRICS = MIRROR / "metrics"
EVENTS = MIRROR / "events.jsonl"
PLAN = BASE_DIR / "db" / "infer_splits.json"
REMOTE = "nas:Guest/weld_infer"
DATASET_NAME = "door_factory_infer"
TOOL_VERSION = 1

Z_EMPTY_MIN = 1481            # 07_factory_infer Z_EMPTY_MIN_MM (세로 바 벌린 설정의 빈 지그)
NARROW_ZO = (1464, 1476)      # 세로 바 모은 설정(E23)의 빈 지그: OCC_ROI 중앙값이 세로 바 높이로 읽힘
NARROW_Z_MIN = 1466           # 그때 도어 ROI 전체 중앙값 (도어가 있으면 ≤1459)
LEAVE_S, ENTER_S = 20, 3      # 서비스와 같은 디바운스
GAP_S = 30                    # 로그가 이만큼 끊기면(서비스 재시작) 구간을 닫는다
MIN_OVERLAP_S = 60            # 세션과 이만큼 겹쳐야 그 세션의 구간으로 본다
MIN_OK, MIN_SHARE = 30, 0.8   # 구간 판정 최소 통과 프레임 수·득표율
GROUP = lambda c: "FRT" if "FRT" in c else "RH" if c.endswith("RH") else "RR"


def T(s):
    return dt.datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")


def S(t):
    return t.strftime("%Y-%m-%d %H:%M:%S")


def load_jsonl(p):
    rows = []
    with open(p, encoding="utf-8", errors="replace") as f:
        for ln in f:
            try:
                rows.append(json.loads(ln))
            except ValueError:
                pass
    return rows


def plate_seen(r):
    """그 초에 지그 중앙에 도어가 안 보이는가(움직임 무관) → 'wide' | 'narrow' | None."""
    if (r.get("pts") or 0) >= 2:
        return None
    zo, z = r.get("z_occ") or 0, r.get("z") or 0
    if zo >= Z_EMPTY_MIN:
        return "wide"
    if NARROW_ZO[0] <= zo <= NARROW_ZO[1] and z >= NARROW_Z_MIN:
        return "narrow"
    return None


def vacancy(r):
    """그 초를 '비움'으로 셀 것인가 — 모은 설정은 깊이 여유가 작아 정지 상태일 때만 센다."""
    k = plate_seen(r)
    return k if k == "wide" or (k == "narrow" and r.get("static")) else None


def segments(rows):
    """1Hz 로그 → 도어 구간 [{t0, t1, rows, end, gap_before_s, gap_kind}]. 진입 = 점유 3초, 이탈 = 비움 20초(이탈 시각 = 비움 시작)."""
    segs, cur, occ_since, emp_since, prev_t = [], None, None, None, None
    last_end, kinds = None, collections.Counter()
    for r in rows:
        try:
            t = T(r["ts"])
        except (KeyError, ValueError):
            continue
        if prev_t is not None and (t - prev_t).total_seconds() > GAP_S:
            if cur:
                segs.append(dict(t0=cur["t0"], t1=prev_t, rows=cur["rows"], end="gap", gap_before_s=cur["gap"], gap_kind=cur["kind"]))
                cur, last_end = None, None           # 로그 공백 너머로는 빈 지그 시간을 잴 수 없다
            occ_since = emp_since = None
            kinds.clear()
        vac = vacancy(r)
        d = r.get("diff")
        occ = plate_seen(r) is None and ((r.get("pts") or 0) >= 4 or (d is not None and d >= 0.15))
        occ_since = (occ_since or t) if occ else None
        emp_since = (emp_since or t) if vac else None
        if cur is None:
            if vac:
                kinds[vac] += 1
            if occ_since and (t - occ_since).total_seconds() >= ENTER_S:
                gap = (occ_since - last_end).total_seconds() if last_end else None
                cur = dict(t0=occ_since, rows=[], gap=gap, kind=(kinds.most_common(1)[0][0] if kinds else None))
                kinds.clear()
        else:
            cur["rows"].append(r)
            if emp_since and (t - emp_since).total_seconds() >= LEAVE_S:
                segs.append(dict(t0=cur["t0"], t1=emp_since, rows=cur["rows"], end="leave", gap_before_s=cur["gap"], gap_kind=cur["kind"]))
                cur, last_end = None, emp_since
                kinds.clear()
                kinds[vac] += 1
        prev_t = t
    if cur:
        segs.append(dict(t0=cur["t0"], t1=prev_t, rows=cur["rows"], end="end", gap_before_s=cur["gap"], gap_kind=cur["kind"]))
    return segs


def judge(rows, lo, hi):
    """구간 [lo, hi] 프레임의 판정 요약."""
    rs = [r for r in rows if lo <= T(r["ts"]) <= hi]
    ok = [r for r in rs if r.get("gate") == "ok" and r.get("pred")]
    votes = collections.Counter(r["pred"] for r in ok)
    top = votes.most_common(1)[0] if votes else (None, 0)
    ds = [r["D"] for r in ok if r["pred"] == top[0] and r.get("D")]
    out = dict(n_rows=len(rs), n_ok=len(ok), pred_class=None, n_vote=top[1], votes=dict(votes.most_common(3)),
               d_med=round(statistics.median(ds), 1) if ds else None,
               d_p10=round(sorted(ds)[len(ds) // 10], 1) if ds else None, d_p90=round(sorted(ds)[len(ds) * 9 // 10], 1) if ds else None,
               gate_counts=dict(collections.Counter(r.get("gate") or "none" for r in rs).most_common()),
               first_ok=ok[0]["ts"] if ok else None)
    if top[0] and top[1] >= MIN_OK and top[1] / len(ok) >= MIN_SHARE and top[0] != "unknown":
        out["pred_class"] = top[0]
    return out


def actual_sessions(con, events, day, day_end):
    """그날 실제 세션 [{id, dir, a0, a1, ...}] — 시작·종료는 events(enter/leave) 기준.
    DB 종료 시각은 result.json 마지막 갱신이라, 그날 마지막 도어처럼 전원이 꺼질 때까지 열려 있던 세션은 실제보다 이르다.
    같은 날 leave 가 없으면(다음 날 아침 고아 세션으로 마감) 다음 세션 시작 또는 그날 로그 끝(day_end)까지 열려 있던 것으로 본다."""
    pre = f"{day[:4]}-{day[4:6]}-{day[6:]}"
    ent, lev = {}, {}
    for e in events:
        if not e.get("ts", "").startswith(pre):
            continue
        if e.get("event") == "enter":
            ent[e["session_dir"]] = e["ts"]
        elif e.get("event") == "leave":
            lev[e["session_dir"]] = e["ts"]
    starts = sorted(T(v) for v in ent.values())
    out = []
    for r in con.execute(
            """SELECT s.id, s.session_dir, s.camera_type, s.dataset_id, r.status, r.pred_class, r.radar, r.started_at, r.ended_at,
                      r.service_version, r.confirmed_class, r.excluded, r.excluded_by, r.split_parent, r.result_json, r.thumb_path,
                      r.model_hole, r.model_hole_sha, r.model_cnn
               FROM capture_sessions s JOIN inference_results r ON r.session_id = s.id
               WHERE s.session_dir LIKE ? AND r.split_parent IS NULL ORDER BY r.started_at""", (day + "/%",)):
        m = dict(r)
        a0 = ent.get(m["session_dir"]) or m["started_at"]
        if not a0:
            continue
        a0 = T(a0)
        if m["session_dir"] in lev:
            a1 = T(lev[m["session_dir"]])
        else:
            nxt = [s for s in starts if s > a0]
            a1 = nxt[0] if nxt else day_end
            if m["ended_at"] and a1 is not None:
                a1 = max(a1, T(m["ended_at"]))
        if a1 is None or a1 <= a0:
            continue
        m.update(a0=a0, a1=a1, open_end=m["session_dir"] not in lev)
        out.append(m)
    return out


def running_radar(result_json, cls, t_end):
    """합쳐진 세션이 t_end 시각에 표시하던 레이더 값 — 그때 표시 클래스가 cls 일 때만. 없으면 None."""
    if "FRT" in cls:
        return None
    try:
        hist = json.loads(result_json or "{}").get("pred_history") or []
    except ValueError:
        return None
    cur = None
    for h in hist:
        try:
            if T(h["ts"]) <= t_end:
                cur = h
        except (KeyError, ValueError):
            pass
    if cur and cur.get("pred_class") == cls and cur.get("radar") in (0, 1):
        return cur["radar"]
    return None


def cmd_pull(args):
    METRICS.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rclone", "copy", f"{args.remote}/metrics", str(METRICS), "--include", "metrics_*.jsonl"], check=False)
    subprocess.run(["rclone", "copyto", f"{args.remote}/events.jsonl", str(EVENTS)], check=False)
    n = len(list(METRICS.glob("metrics_*.jsonl")))
    print(f"초 단위 로그 {n}일치 → {METRICS} · events → {EVENTS if EVENTS.exists() else '(없음)'}")


def build_plan(args):
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    have = {r[1] for r in con.execute("PRAGMA table_info(inference_results)")}
    if "split_parent" not in have:                   # 마이그레이션 전 DB 도 읽을 수 있게
        con.close()
        con = sqlite3.connect(":memory:")
        con.row_factory = sqlite3.Row
        src = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
        src.backup(con)
        src.close()
        migrate(con)
    events = load_jsonl(EVENTS) if EVENTS.exists() else []
    days = sorted(p.stem.split("_")[1] for p in METRICS.glob("metrics_*.jsonl"))
    if args.days:
        days = [d for d in days if d in args.days]
    parents, n_act, n_seg = [], 0, 0
    for day in days:
        rows = load_jsonl(METRICS / f"metrics_{day}.jsonl")
        segs = segments(rows)
        day_end = next((T(r["ts"]) for r in reversed(rows) if r.get("ts")), None)
        acts = actual_sessions(con, events, day, day_end)
        n_act += len(acts)
        n_seg += len(segs)
        for a in acts:
            parts = []
            for sg in segs:
                lo, hi = max(sg["t0"], a["a0"]), min(sg["t1"], a["a1"])
                if (hi - lo).total_seconds() < MIN_OVERLAP_S:
                    continue
                j = judge(sg["rows"], lo, hi)
                j.update(start=S(lo), end=S(hi), clipped=bool(sg["t0"] < a["a0"] or sg["t1"] > a["a1"]), seg_end=sg["end"],
                         gap_before_s=sg["gap_before_s"] if sg["t0"] >= a["a0"] else None, gap_kind=sg["gap_kind"])
                parts.append(j)
            judged = [p for p in parts if p["pred_class"]]
            if len(judged) < 2:
                continue
            for k, p in enumerate(judged, 1):
                p["k"] = k
                p["session_dir"] = f"{a['session_dir']}_p{k}"
                p["radar"] = running_radar(a["result_json"], p["pred_class"], T(p["end"]))
            for p, nxt in zip(judged, judged[1:]):
                p["gap_after_s"] = nxt["gap_before_s"]
            parents.append(dict(
                session_dir=a["session_dir"], session_id=a["id"], start=S(a["a0"]), end=S(a["a1"]), open_end=a["open_end"], status=a["status"],
                pred_class=a["pred_class"], confirmed_class=a["confirmed_class"], service_version=a["service_version"],
                parts=judged, dropped=[dict(start=p["start"], end=p["end"], n_rows=p["n_rows"], n_ok=p["n_ok"], votes=p["votes"])
                                       for p in parts if not p["pred_class"]]))
    return dict(tool_version=TOOL_VERSION, generated_at=S(dt.datetime.now()), days=days, n_sessions=n_act, n_segments=n_seg,
                rules=dict(z_empty_min=Z_EMPTY_MIN, narrow_zo=NARROW_ZO, narrow_z_min=NARROW_Z_MIN, leave_s=LEAVE_S, enter_s=ENTER_S,
                           min_ok=MIN_OK, min_share=MIN_SHARE),
                parents=parents)


def cmd_scan(args):
    if not list(METRICS.glob("metrics_*.jsonl")):
        sys.exit(f"초 단위 로그 없음: {METRICS} — 먼저 `pull` (또는 추론 PC 에서 복사)")
    plan = build_plan(args)
    old = json.loads(PLAN.read_text(encoding="utf-8")) if PLAN.exists() else {"parents": []}
    # 지난 계획 중 이번에 다시 계산하지 않은 날짜(로그가 이미 지워진 날)의 세션은 그대로 둔다
    keep = [p for p in old.get("parents", []) if p["session_dir"].split("/")[0] not in plan["days"]]
    plan["parents"] = sorted(keep + plan["parents"], key=lambda p: p["session_dir"])
    print(f"로그 {len(plan['days'])}일({plan['days'][0]}~{plan['days'][-1]}) · 세션 {plan['n_sessions']} · 복원 구간 {plan['n_segments']} "
          f"· 분할 대상 {len(plan['parents'])}세션 → {sum(len(p['parts']) for p in plan['parents'])}대")
    for p in plan["parents"]:
        print(f"\n{p['session_dir']}  {p['start'][11:16]}~{p['end'][11:16]}  [{p['status']} {p['pred_class'] or '—'}]"
              f"{'  ※ 이미 확정됨: ' + p['confirmed_class'] if p['confirmed_class'] else ''}")
        for q in p["parts"]:
            gap = "" if q.get("gap_before_s") is None else f" · 앞 빈 지그 {q['gap_before_s']:.0f}초({ {'wide': '벌린', 'narrow': '모은'}.get(q['gap_kind'], '?') } 설정)"
            rd = {1: " 레이더 O?", 0: " 레이더 X?"}.get(q["radar"], "")
            print(f"   p{q['k']} {q['start'][11:19]}~{q['end'][11:19]} {(T(q['end']) - T(q['start'])).total_seconds() / 60:4.0f}분 "
                  f"{q['pred_class']:18s} {q['n_vote']}/{q['n_ok']} D {q['d_med']}{rd}{gap}")
        for q in p["dropped"]:
            print(f"   —  {q['start'][11:19]}~{q['end'][11:19]} 판정 없음 (통과 프레임 {q['n_ok']}) → 세션으로 만들지 않음")
    if not args.dry_run:
        PLAN.write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n계획 저장: {PLAN} — 반영은 `apply`")


def _lookups(con, dataset_id):
    """기존 판정 행에서 (클래스, 레이더) → 품번, 클래스 → CAD D 를 모은다."""
    parts, cad = {}, {}
    for r in con.execute("SELECT pred_class, radar, part_no, part_name, candidates FROM inference_results WHERE pred_class IS NOT NULL"):
        if r["part_no"]:
            parts.setdefault((r["pred_class"], None if "FRT" in r["pred_class"] else r["radar"]), (r["part_no"], r["part_name"]))
        try:
            for c in json.loads(r["candidates"] or "[]"):
                cad.setdefault(c["class"], (c.get("group"), c["cad_D_mm"]))
        except (ValueError, KeyError, TypeError):
            pass
    return parts, cad


def _thumb(day, child_dir, start, end, parent_thumb, own_frames):
    """복원 세션 썸네일: 원본 프레임이 이 구간 것이면 원 세션 썸네일, 아니면 구간 한가운데 타임랩스(홀 6점 우선)."""
    dst = MIRROR / child_dir / "thumb.jpg"
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = None
    if own_frames and parent_thumb:
        p = Path(parent_thumb)
        p = p if p.is_absolute() else BASE_DIR / p
        src = p if p.exists() else None
    if src is None:
        idx = MIRROR / "timelapse" / day / "index.jsonl"
        if idx.exists():
            mid = T(start) + (T(end) - T(start)) / 2
            cands = []
            for f in load_jsonl(idx):
                try:
                    t = T(f["ts"])
                except (KeyError, ValueError):
                    continue
                if T(start) <= t <= T(end) and not f.get("boundary") and (idx.parent / f.get("file", "")).exists():
                    cands.append((-(min(f.get("pts") or 0, 5)), abs((t - mid).total_seconds()), idx.parent / f["file"]))
            if cands:
                src = sorted(cands)[0][2]
    if src is None:
        return None
    shutil.copyfile(src, dst)
    return str(dst.relative_to(BASE_DIR))


def cmd_apply(args):
    if not PLAN.exists():
        sys.exit(f"계획 없음: {PLAN} — 먼저 `scan`")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row
    migrate(con)
    now = S(dt.datetime.now())
    n_par = n_child = n_img = 0
    for p in plan["parents"]:
        if args.only and p["session_dir"] not in args.only:
            continue
        par = con.execute("SELECT s.*, r.confirmed_class, r.excluded, r.excluded_by, r.thumb_path, r.result_json, r.service_version, "
                          "r.model_hole, r.model_hole_sha, r.model_cnn FROM capture_sessions s JOIN inference_results r ON r.session_id = s.id "
                          "WHERE s.session_dir = ?", (p["session_dir"],)).fetchone()
        if par is None:
            print(f"건너뜀 {p['session_dir']}: DB 에 없음 (ingest_infer 먼저)")
            continue
        if par["confirmed_class"] and not args.force:
            print(f"건너뜀 {p['session_dir']}: 이미 확정된 세션({par['confirmed_class']}) — 확정을 취소하거나 --force")
            continue
        if par["excluded"] and par["excluded_by"] != "split" and not args.force:
            print(f"건너뜀 {p['session_dir']}: 사용자가 제외한 세션 — --force 로만 분할")
            continue
        day = p["session_dir"].split("/")[0]
        part_map, cad = _lookups(con, par["dataset_id"])
        try:
            frames = json.loads(par["result_json"] or "{}").get("frames") or []
        except ValueError:
            frames = []
        child_ids = []
        for q in p["parts"]:
            cls, radar = q["pred_class"], q.get("radar")
            own = [f for f in frames if f.get("ts") and q["start"] <= f["ts"] <= q["end"]]
            pno, pname = part_map.get((cls, None if "FRT" in cls else radar), (None, None))
            cands = sorted(({"class": c, "group": g, "cad_D_mm": d, "diff_mm": round(abs(q["d_med"] - d), 1)}
                            for c, (g, d) in cad.items()), key=lambda c: c["diff_mm"])[:4] if q.get("d_med") else []
            nearest = next((c["diff_mm"] for c in cands if c["class"] == cls), None)
            others = [c["diff_mm"] for c in cands if c["class"] != cls]
            margin = round(others[0] - nearest, 1) if others and nearest is not None else None
            radar_txt = "해당 없음" if "FRT" in cls else {1: "O (합쳐진 세션 표시값)", 0: "X (합쳐진 세션 표시값)"}.get(radar, "미정 (복원 세션)")
            dwell = (T(q["end"]) - T(q["start"])).total_seconds()
            info = dict(restored=True, tool_version=plan.get("tool_version"), parent=p["session_dir"], part=q["k"], parts=len(p["parts"]),
                        n_rows=q["n_rows"], n_ok=q["n_ok"], n_vote=q["n_vote"], votes=q["votes"], d_med=q["d_med"], d_p10=q.get("d_p10"),
                        d_p90=q.get("d_p90"), gap_before_s=q.get("gap_before_s"), gap_after_s=q.get("gap_after_s"), gap_kind=q.get("gap_kind"),
                        clipped=q.get("clipped"), seg_end=q.get("seg_end"), radar_src="pred_history" if radar in (0, 1) else None,
                        own_frames=[f.get("file") for f in own], rules=plan.get("rules"))
            con.execute(
                """INSERT INTO capture_sessions (dataset_id, session_dir, class_name, camera_type, capture_method, saved_pairs, valid_frames,
                                                 finished, stop_reason, started_at, ended_at, option_radar, option_source, option_at, notes)
                   VALUES (?, ?, ?, ?, 'snapshot', ?, ?, 1, 'judged', ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(session_dir) DO UPDATE SET started_at = excluded.started_at, ended_at = excluded.ended_at,
                     saved_pairs = excluded.saved_pairs, valid_frames = excluded.valid_frames,
                     class_name = CASE WHEN capture_sessions.option_source = 'user' THEN capture_sessions.class_name ELSE excluded.class_name END""",
                (par["dataset_id"], q["session_dir"], cls, par["camera_type"], len(own), len(own), q["start"], q["end"],
                 radar, "auto" if radar in (0, 1) else None, now if radar in (0, 1) else None,
                 f"[{now[:16]}] 합쳐진 세션 {p['session_dir']} 에서 초 단위 기록으로 복원 ({q['k']}/{len(p['parts'])}, split_infer_sessions.py)"))
            cid = con.execute("SELECT id FROM capture_sessions WHERE session_dir = ?", (q["session_dir"],)).fetchone()[0]
            thumb = _thumb(day, q["session_dir"], q["start"], q["end"], par["thumb_path"], own)
            con.execute(
                """INSERT INTO inference_results
                     (session_id, status, pred_class, pred_group, part_no, part_name, radar, radar_txt, source, conflict, confidence,
                      d_mm, margin_mm, nearest_mm, n_judged, n_window, n_processed, n_clean, gate_counts, candidates,
                      model_hole, model_hole_sha, model_cnn, service_version, dwell_s, judged_at, started_at, ended_at, thumb_path,
                      result_json, split_parent, split_part, split_info)
                   VALUES (?, 'judged', ?, ?, ?, ?, ?, ?, 'hole', 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(session_id) DO UPDATE SET
                     pred_class = excluded.pred_class, pred_group = excluded.pred_group, part_no = excluded.part_no, part_name = excluded.part_name,
                     radar = excluded.radar, radar_txt = excluded.radar_txt, confidence = excluded.confidence, d_mm = excluded.d_mm,
                     margin_mm = excluded.margin_mm, nearest_mm = excluded.nearest_mm, n_judged = excluded.n_judged, n_window = excluded.n_window,
                     n_processed = excluded.n_processed, n_clean = excluded.n_clean, gate_counts = excluded.gate_counts,
                     candidates = excluded.candidates, dwell_s = excluded.dwell_s, judged_at = excluded.judged_at,
                     started_at = excluded.started_at, ended_at = excluded.ended_at, thumb_path = excluded.thumb_path,
                     result_json = excluded.result_json, split_parent = excluded.split_parent, split_part = excluded.split_part,
                     split_info = excluded.split_info, updated_at = datetime('now','localtime')""",
                (cid, cls, GROUP(cls), pno, pname, radar, radar_txt, round(100.0 * q["n_vote"] / q["n_ok"], 1),
                 q["d_med"], margin, nearest, q["n_vote"], q["n_ok"], q["n_rows"], q["n_ok"],
                 json.dumps(q["gate_counts"], ensure_ascii=False), json.dumps(cands, ensure_ascii=False),
                 par["model_hole"], par["model_hole_sha"], par["model_cnn"], par["service_version"], dwell, q.get("first_ok"),
                 q["start"], q["end"], thumb, json.dumps(dict(info, session_dir=q["session_dir"], status="judged", pred_class=cls,
                                                               started_at=q["start"], ended_at=q["end"]), ensure_ascii=False),
                 par["id"], q["k"], json.dumps(info, ensure_ascii=False)))
            # 원 세션에 남은 원본 프레임 → 촬영 시각이 속한 복원 세션으로 (라벨 = 이 구간 판정 클래스)
            if own:
                model, part = (cls.split("_door_", 1)[0], "door_" + cls.split("_door_", 1)[1]) if "_door_" in cls else (None, None)
                con.execute("INSERT OR IGNORE INTO classes (dataset_id, name, display_name, model_name, part_type) VALUES (?, ?, ?, ?, ?)",
                            (par["dataset_id"], cls, cls.replace("_", " "), model, part))
                class_id = con.execute("SELECT id FROM classes WHERE dataset_id = ? AND name = ?", (par["dataset_id"], cls)).fetchone()[0]
                for f in own:
                    n_img += con.execute("UPDATE OR IGNORE images SET session_id = ?, class_id = ?, captured_at = ? "
                                         "WHERE session_id = ? AND rgb_filename = ?", (cid, class_id, f["ts"], par["id"], f["file"])).rowcount
            child_ids.append(cid)
            n_child += 1
        reason = f"도어 {len(p['parts'])}대로 분할 — 초 단위 기록으로 복원"
        con.execute("UPDATE inference_results SET excluded = 1, excluded_reason = ?, excluded_by = 'split', excluded_at = ?, split_info = ?, "
                    "updated_at = datetime('now','localtime') WHERE session_id = ?",
                    (reason, now, json.dumps(dict(children=[q["session_dir"] for q in p["parts"]], at=now, dropped=p.get("dropped")),
                                             ensure_ascii=False), par["id"]))
        con.execute("UPDATE images SET is_valid = 0 WHERE session_id = ?", (par["id"],))     # 어느 도어 것인지 못 가린 프레임만 남는다
        if not (par["excluded"] and par["excluded_by"] == "split"):
            con.execute("UPDATE capture_sessions SET notes = COALESCE(notes || '\n', '') || ? WHERE id = ?",
                        (f"[{now[:16]}] 현장 판정 분할: {reason} ({', '.join(q['session_dir'].split('/')[-1] for q in p['parts'])})", par["id"]))
        for cid_row in con.execute("SELECT id FROM classes WHERE dataset_id = ?", (par["dataset_id"],)).fetchall():
            con.execute("UPDATE classes SET image_count = (SELECT COUNT(*) FROM images WHERE class_id = ?) WHERE id = ?", (cid_row[0], cid_row[0]))
        n_par += 1
        print(f"분할 {p['session_dir']} → {len(child_ids)}대")
    con.commit()
    print(f"반영: 원 세션 {n_par}개 제외 표시 · 복원 세션 {n_child}개 · 원본 프레임 {n_img}장 이동")


def cmd_undo(args):
    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row
    migrate(con)
    for sd in args.dirs:
        par = con.execute("SELECT s.id, s.dataset_id, r.pred_class FROM capture_sessions s JOIN inference_results r ON r.session_id = s.id "
                          "WHERE s.session_dir = ? AND r.excluded_by = 'split'", (sd,)).fetchone()
        if par is None:
            print(f"건너뜀 {sd}: 분할된 세션이 아님")
            continue
        kids = con.execute("SELECT r.session_id, s.session_dir, r.confirmed_class FROM inference_results r JOIN capture_sessions s "
                           "ON s.id = r.session_id WHERE r.split_parent = ?", (par["id"],)).fetchall()
        if any(k["confirmed_class"] for k in kids) and not args.force:
            print(f"건너뜀 {sd}: 복원 세션 중 확정된 것이 있음 — --force")
            continue
        pcls = con.execute("SELECT id FROM classes WHERE dataset_id = ? AND name = ?",
                           (par["dataset_id"], par["pred_class"] or "Unknown")).fetchone()
        for k in kids:
            if pcls:
                con.execute("UPDATE OR IGNORE images SET session_id = ?, class_id = ? WHERE session_id = ?", (par["id"], pcls[0], k["session_id"]))
            con.execute("DELETE FROM images WHERE session_id = ?", (k["session_id"],))
            con.execute("DELETE FROM inference_results WHERE session_id = ?", (k["session_id"],))
            con.execute("DELETE FROM capture_sessions WHERE id = ?", (k["session_id"],))
            shutil.rmtree(MIRROR / k["session_dir"], ignore_errors=True)
        con.execute("UPDATE inference_results SET excluded = 0, excluded_reason = NULL, excluded_by = NULL, excluded_at = NULL, split_info = NULL, "
                    "updated_at = datetime('now','localtime') WHERE session_id = ?", (par["id"],))
        con.execute("UPDATE images SET is_valid = 1 WHERE session_id = ?", (par["id"],))
        con.execute("UPDATE capture_sessions SET notes = COALESCE(notes || '\n', '') || ? WHERE id = ?",
                    (f"[{S(dt.datetime.now())[:16]}] 현장 판정 분할 취소 (복원 세션 {len(kids)}개 삭제)", par["id"]))
        print(f"분할 취소 {sd}: 복원 세션 {len(kids)}개 삭제")
    con.commit()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", default=str(DB))
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("pull", help="NAS 에서 초 단위 로그·events 받기")
    s.add_argument("--remote", default=REMOTE)
    s.set_defaults(fn=cmd_pull)
    s = sub.add_parser("scan", help="분할 후보 계산 → 계획 파일")
    s.add_argument("--days", nargs="*", help="YYYYMMDD … (기본: 로그가 있는 모든 날)")
    s.add_argument("--dry-run", action="store_true", help="계획 파일을 쓰지 않음")
    s.set_defaults(fn=cmd_scan)
    s = sub.add_parser("apply", help="계획을 DB 에 반영")
    s.add_argument("--only", nargs="*", help="이 세션만")
    s.add_argument("--force", action="store_true", help="확정·사용자 제외 세션도 분할")
    s.set_defaults(fn=cmd_apply)
    s = sub.add_parser("undo", help="분할 취소")
    s.add_argument("dirs", nargs="+")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_undo)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
