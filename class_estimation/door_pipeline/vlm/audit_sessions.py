"""1·2번 — 현장 세션 VLM 독립 2차 감사 (Claude Opus 5.5 vision).

  --source collect : decisions.csv 의 유효·확정 세션(로컬 미러 datasets_factory_collect) — GT 와 일치율
  --source infer   : datasets_factory_infer/<날짜>/s_*/result.json (07_factory_infer 판정) — 판정 일치 + 미확정 세션 원인 설명(2번)
                     원본 프레임은 NAS weld_infer 에서 rgb_0000/0005 만 내려받아 artifacts/frames/ 에 둔다(thumb.jpg 는 점 오버레이가 있어 안 씀)

출력: artifacts/audit_<source>_<run>.jsonl (프레임 단위) · ../report/vlm_audit/audit_<source>_<run>.md (요약) · DB vlm_audits (--no-db 로 끔)
예:  python vlm/audit_sessions.py --source collect --dry-run
     python vlm/audit_sessions.py --source collect --limit 5
     python vlm/audit_sessions.py --source collect --frames 1
     python vlm/audit_sessions.py --source infer
     python vlm/audit_sessions.py --source collect --frames 3 --sessions artifacts/disagree.txt   # 불일치 세션만 3프레임 재감사
"""
import argparse
import collections
import csv
import glob
import json
import os
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlm_common as C

DB = os.path.join(C.DOOR, 'db', 'door_pipeline.db')
MIRROR = os.path.join(C.DOOR, 'datasets_factory_collect')
INFER = os.path.join(C.DOOR, 'datasets_factory_infer')
FRAMES = os.path.join(C.ART, 'frames')
REPORT = os.path.join(C.DOOR, 'report', 'vlm_audit')
RCLONE = os.path.expanduser('~/.local/bin/rclone')

DDL = """CREATE TABLE IF NOT EXISTS vlm_audits (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL, source TEXT NOT NULL, session_dir TEXT NOT NULL, frame TEXT NOT NULL, model TEXT NOT NULL,
  ref_class TEXT, ref_radar INTEGER, ref_status TEXT,
  vlm_door_present INTEGER, vlm_group TEXT, vlm_class TEXT, vlm_class_conf TEXT, vlm_radar TEXT,
  vlm_anomalies TEXT, vlm_description TEXT, vlm_reasoning TEXT,
  agree_group INTEGER, agree_class INTEGER, agree_radar INTEGER,
  input_tokens INTEGER, cache_read_tokens INTEGER, cache_write_tokens INTEGER, output_tokens INTEGER, cost_usd REAL,
  stop_reason TEXT, error TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);
CREATE INDEX IF NOT EXISTS idx_vlm_audits_session ON vlm_audits(session_dir);"""


# ---------- 대상 세션 ----------
def pick_frames(files, n):
    """정렬된 프레임 목록에서 n 장을 고르게(중앙 우선) 뽑는다."""
    if not files:
        return []
    if n == 1:
        return [files[len(files) // 2]]
    idx = sorted({round((i + 0.5) * len(files) / n - 0.5) for i in range(n)})
    return [files[min(max(i, 0), len(files) - 1)] for i in idx]


def targets_collect(frames_n, limit, only):
    rows = list(csv.DictReader(open(os.path.join(C.DOOR, 'db', 'decisions.csv'), encoding='utf-8')))
    out = []
    for r in rows:
        if r['valid'] != '1' or r['class_db'] not in C.CAD_D:
            continue
        d = os.path.join(MIRROR, r['session_dir'])
        if not os.path.isdir(d):
            continue
        if only and r['session_dir'] not in only:
            continue
        radar = int(r['option_radar']) if r['option_radar'] not in ('', None) else None
        if C.GROUP[r['class_db']] != 'FRT' and radar is None:
            continue                                   # 레이더 미확정 세션은 품번 GT 가 없음
        files = sorted(glob.glob(os.path.join(d, 'rgb_*.png')))
        for f in pick_frames(files, frames_n):
            out.append({'session_dir': r['session_dir'], 'frame': f, 'ref_class': r['class_db'], 'ref_radar': radar,
                        'ref_status': 'judged', 'split': r['split'], 'option_source': r['option_source']})
    if limit:
        seen, keep = set(), []
        for t in out:
            if len(seen) >= limit and t['session_dir'] not in seen:
                break
            seen.add(t['session_dir']); keep.append(t)
        out = keep
    return out


def ensure_infer_frames(date, s):
    """NAS weld_infer/<date>/<s>/rgb_0000.png, rgb_0005.png → artifacts/frames/<date>_<s>/"""
    dest = os.path.join(FRAMES, f'{date}_{s}')
    have = sorted(glob.glob(os.path.join(dest, 'rgb_*.png')))
    if have:
        return have
    os.makedirs(dest, exist_ok=True)
    cmd = [RCLONE, 'copy', f'nas:Guest/weld_infer/{date}/{s}/', dest, '--include', 'rgb_0000.png', '--include', 'rgb_0005.png']
    subprocess.run(cmd, check=False, timeout=300)
    return sorted(glob.glob(os.path.join(dest, 'rgb_*.png')))


def targets_infer(frames_n, limit, only):
    out = []
    for rj in sorted(glob.glob(os.path.join(INFER, '2026*', 's_*', 'result.json'))):
        sdir = os.path.relpath(os.path.dirname(rj), INFER)        # 20261103/s_143920
        if only and sdir not in only:
            continue
        date, s = sdir.split('/')
        r = json.load(open(rj, encoding='utf-8'))
        files = ensure_infer_frames(date, s)
        if not files:
            print(f'[warn] 프레임 없음: {sdir}', file=sys.stderr); continue
        radar = r.get('radar')
        for f in pick_frames(files, frames_n):
            out.append({'session_dir': sdir, 'frame': f, 'ref_class': r.get('pred_class'),
                        'ref_radar': (int(radar) if radar in (0, 1) else None), 'ref_status': r.get('status'),
                        'ref_part_no': r.get('part_no'), 'gate_counts': r.get('gate_counts')})
        if limit and len({t['session_dir'] for t in out}) >= limit:
            break
    return out


# ---------- 호출 ----------
def ask(client, prefix, t):
    content = prefix + [
        {'type': 'text', 'text': '검사 대상 이미지:'},
        C.image_block(t['frame']),
        {'type': 'text', 'text': '이 이미지를 판정하라. 참조 이미지와 폭·그릴·브래킷을 비교하고 JSON 으로만 답하라.'},
    ]
    resp = client.messages.create(
        model=C.MODEL, max_tokens=4000,
        output_config={'effort': 'medium', 'format': {'type': 'json_schema', 'schema': C.AUDIT_SCHEMA}},
        system=[{'type': 'text', 'text': C.DOMAIN_TEXT}],
        messages=[{'role': 'user', 'content': content}],
    )
    d, usd = C.usage_cost(resp.usage)
    rec = dict(t); rec.update({'model': resp.model, 'stop_reason': resp.stop_reason, 'usage': d, 'cost_usd': usd, 'error': None})
    if resp.stop_reason == 'refusal':
        rec['error'] = 'refusal:' + str(getattr(resp, 'stop_details', None))
        rec['vlm'] = None
        return rec
    try:
        rec['vlm'] = json.loads(C.text_of(resp))
    except Exception as e:                                   # max_tokens 등으로 JSON 이 깨진 경우
        rec['vlm'] = None; rec['error'] = f'json:{e}: {C.text_of(resp)[:200]}'
    return rec


def agree(rec):
    v = rec.get('vlm') or {}
    rc, rr = rec.get('ref_class'), rec.get('ref_radar')
    out = {'agree_group': None, 'agree_class': None, 'agree_radar': None}
    if not v or rc not in C.CAD_D:
        return out
    out['agree_group'] = int(v.get('group') == C.GROUP[rc])
    out['agree_class'] = int(v.get('class_name') == rc)
    if C.GROUP[rc] == 'FRT':
        out['agree_radar'] = int(v.get('radar') == 'N/A')
    elif rr is not None:
        out['agree_radar'] = int(v.get('radar') == ('O' if rr else 'X'))
    return out


def save_db(run, source, recs):
    con = sqlite3.connect(DB)
    con.executescript(DDL)
    for r in recs:
        v = r.get('vlm') or {}; a = agree(r); u = r.get('usage', {})
        con.execute("""INSERT INTO vlm_audits(run_id,source,session_dir,frame,model,ref_class,ref_radar,ref_status,
            vlm_door_present,vlm_group,vlm_class,vlm_class_conf,vlm_radar,vlm_anomalies,vlm_description,vlm_reasoning,
            agree_group,agree_class,agree_radar,input_tokens,cache_read_tokens,cache_write_tokens,output_tokens,cost_usd,stop_reason,error)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (run, source, r['session_dir'], os.path.relpath(r['frame'], C.DOOR), r.get('model', C.MODEL), r.get('ref_class'), r.get('ref_radar'), r.get('ref_status'),
             (None if not v else int(bool(v.get('door_present')))), v.get('group'), v.get('class_name'), v.get('class_confidence'), v.get('radar'),
             json.dumps(v.get('anomalies'), ensure_ascii=False) if v else None, v.get('description'), v.get('reasoning'),
             a['agree_group'], a['agree_class'], a['agree_radar'],
             u.get('input_tokens'), u.get('cache_read_tokens'), u.get('cache_write_tokens'), u.get('output_tokens'), r.get('cost_usd'),
             r.get('stop_reason'), r.get('error')))
    con.commit(); con.close()


# ---------- 요약 ----------
def vote(vals):
    vals = [v for v in vals if v not in (None, 'unsure', 'none')]
    if not vals:
        return 'unsure'
    c = collections.Counter(vals).most_common()
    return c[0][0] if len(c) == 1 or c[0][1] > c[1][1] else 'tie'


def summarize(run, source, recs, tally, elapsed):
    by = collections.defaultdict(list)
    for r in recs:
        by[r['session_dir']].append(r)
    L = [f'# VLM 독립 감사 — source={source} run={run}', '',
         f'- 모델 {C.MODEL} · 세션 {len(by)} · 프레임 {len(recs)} · 소요 {elapsed/60:.1f}분 · 비용 {tally}',
         '- 판정 단위: 프레임별 VLM 응답 → 세션 다수결(unsure 제외). 비교 기준: collect=사용자 확정 라벨(decisions.csv), infer=07_factory_infer result.json', '']
    n_err = sum(1 for r in recs if r.get('error'))
    if n_err:
        L.append(f'- 오류/거부 {n_err}건 (jsonl 의 error 참조)'); L.append('')
    sess = []
    for sd, rs in by.items():
        vs = [r.get('vlm') or {} for r in rs]
        ref_c, ref_r, st = rs[0].get('ref_class'), rs[0].get('ref_radar'), rs[0].get('ref_status')
        g, c, rd = vote([v.get('group') for v in vs]), vote([v.get('class_name') for v in vs]), vote([v.get('radar') for v in vs])
        an = sorted({a for v in vs for a in (v.get('anomalies') or [])})
        desc = ' | '.join(v.get('description', '') for v in vs)
        sess.append({'session_dir': sd, 'ref_class': ref_c, 'ref_radar': ref_r, 'ref_status': st, 'vlm_group': g, 'vlm_class': c,
                     'vlm_radar': rd, 'anomalies': an, 'description': desc})
    judged = [s for s in sess if s['ref_class'] in C.CAD_D]
    def rate(ok, n):
        return f'{ok}/{n} ({100*ok/n:.1f}%)' if n else '-'
    gk = sum(1 for s in judged if s['vlm_group'] == C.GROUP[s['ref_class']])
    ck = sum(1 for s in judged if s['vlm_class'] == s['ref_class'])
    rr = [s for s in judged if C.GROUP[s['ref_class']] != 'FRT' and s['ref_radar'] is not None]
    rk = sum(1 for s in rr if s['vlm_radar'] == ('O' if s['ref_radar'] else 'X'))
    fr = [s for s in judged if C.GROUP[s['ref_class']] == 'FRT']
    fk = sum(1 for s in fr if s['vlm_radar'] == 'N/A')
    L += ['## 세션 단위 일치율 (판정 세션)', '', '| 항목 | 일치 |', '|---|---|',
          f'| 형상군(FRT/LH_RR/RH) | {rate(gk, len(judged))} |', f'| 클래스 9종 | {rate(ck, len(judged))} |',
          f'| 레이더 O/X (RR·RH) | {rate(rk, len(rr))} |', f'| FRT 레이더 N/A | {rate(fk, len(fr))} |', '']
    # 클래스 혼동표
    cls = C.CLASSES + ['unsure', 'tie', 'none']
    conf = collections.Counter((s['ref_class'], s['vlm_class']) for s in judged)
    cols = [c for c in cls if any(conf[(r, c)] for r in C.CLASSES)]
    L += ['## 클래스 혼동표 (행=확정 라벨, 열=VLM)', '', '| 확정 \\ VLM | ' + ' | '.join(cols) + ' | 계 |', '|---|' + '---|' * (len(cols) + 1)]
    for r in C.CLASSES:
        n = sum(conf[(r, c)] for c in cls)
        if n:
            L.append(f'| {r} | ' + ' | '.join(str(conf[(r, c)] or '') for c in cols) + f' | {n} |')
    L.append('')
    # 형상군 혼동
    gconf = collections.Counter((C.GROUP[s['ref_class']], s['vlm_group']) for s in judged)
    gcols = ['FRT', 'LH_RR', 'RH', 'unsure', 'tie', 'none']
    L += ['## 형상군 혼동표', '', '| 확정 \\ VLM | ' + ' | '.join(gcols) + ' |', '|---|' + '---|' * len(gcols)]
    for g in ['FRT', 'LH_RR', 'RH']:
        if any(gconf[(g, c)] for c in gcols):
            L.append(f'| {g} | ' + ' | '.join(str(gconf[(g, c)] or '') for c in gcols) + ' |')
    L.append('')
    # 불일치 목록
    dis = [s for s in judged if s['vlm_group'] != C.GROUP[s['ref_class']] or s['vlm_class'] != s['ref_class']
           or (s in rr and s['vlm_radar'] != ('O' if s['ref_radar'] else 'X'))]
    L += [f'## 불일치 세션 ({len(dis)})', '', '| 세션 | 확정 | 레이더 | VLM 형상군 | VLM 클래스 | VLM 레이더 | 이상 | 설명 |', '|---|---|---|---|---|---|---|---|']
    for s in dis:
        L.append(f"| {s['session_dir']} | {s['ref_class']} | {s['ref_radar']} | {s['vlm_group']} | {s['vlm_class']} | {s['vlm_radar']} | {','.join(s['anomalies'])} | {s['description'][:120]} |")
    L.append('')
    # 이상 태그 집계 + 미확정 세션(2번)
    an = collections.Counter(a for s in sess for a in s['anomalies'])
    L += ['## 이상 태그 집계 (세션)', ''] + [f'- {k}: {v}' for k, v in an.most_common()] + ['']
    un = [s for s in sess if s['ref_class'] not in C.CAD_D]
    if un:
        L += [f'## 미확정·보류 세션 원인 설명 ({len(un)})', '', '| 세션 | 상태 | VLM 도어 | VLM 형상군/클래스/레이더 | 이상 | 설명 |', '|---|---|---|---|---|---|']
        for s in un:
            L.append(f"| {s['session_dir']} | {s['ref_status']} | {'-'} | {s['vlm_group']}/{s['vlm_class']}/{s['vlm_radar']} | {','.join(s['anomalies'])} | {s['description'][:160]} |")
        L.append('')
    os.makedirs(REPORT, exist_ok=True)
    p = os.path.join(REPORT, f'audit_{source}_{run}.md')
    open(p, 'w', encoding='utf-8').write('\n'.join(L))
    dp = os.path.join(C.ART, f'disagree_{source}_{run}.txt')
    open(dp, 'w').write('\n'.join(s['session_dir'] for s in dis))
    return p, sess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', choices=['collect', 'infer'], required=True)
    ap.add_argument('--frames', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0, help='세션 수 상한(파일럿)')
    ap.add_argument('--sessions', help='세션 dir 목록 파일(한 줄 하나) — 그 세션만')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--no-db', action='store_true')
    a = ap.parse_args()
    only = set(l.strip() for l in open(a.sessions) if l.strip()) if a.sessions else None
    ts = targets_collect(a.frames, a.limit, only) if a.source == 'collect' else targets_infer(a.frames, a.limit, only)
    ns = len({t['session_dir'] for t in ts})
    est = len(ts) * 0.035
    print(f'대상: 세션 {ns} · 프레임 {len(ts)} · 예상 비용 ≈ ${est:.2f} (프레임당 ~$0.035, 참조 15장 캐시 전제)')
    if a.dry_run or not ts:
        for t in ts[:5]:
            print('  예:', t['session_dir'], os.path.basename(t['frame']), t['ref_class'], t['ref_radar'], t['ref_status'])
        return
    client = C.get_client()
    run = C.run_id()
    prefix = C.reference_blocks()
    tally = C.Tally(); recs = []; t0 = time.time()
    outp = os.path.join(C.ART, f'audit_{a.source}_{run}.jsonl'); os.makedirs(C.ART, exist_ok=True)
    # 첫 호출은 단독으로(캐시 생성) → 이후 병렬(캐시 읽기)
    first = ask(client, prefix, ts[0]); recs.append(first); tally.add(first['usage'], first['cost_usd'])
    print(f"[1/{len(ts)}] {first['session_dir']} → {json.dumps(first.get('vlm'), ensure_ascii=False)[:160]} | {tally}")
    with open(outp, 'a', encoding='utf-8') as fo, ThreadPoolExecutor(max_workers=a.workers) as ex:
        fo.write(json.dumps(first, ensure_ascii=False) + '\n')
        futs = {ex.submit(ask, client, prefix, t): t for t in ts[1:]}
        for i, fu in enumerate(as_completed(futs), 2):
            t = futs[fu]
            try:
                r = fu.result()
            except Exception as e:
                r = dict(t); r.update({'vlm': None, 'error': f'exc:{type(e).__name__}:{e}', 'usage': {}, 'cost_usd': 0.0})
            recs.append(r); fo.write(json.dumps(r, ensure_ascii=False) + '\n'); fo.flush()
            if r.get('usage'):
                tally.add(r['usage'], r['cost_usd'])
            v = r.get('vlm') or {}
            flag = '' if not v or r.get('ref_class') not in C.CAD_D else ('OK' if v.get('class_name') == r['ref_class'] else '**DIFF**')
            print(f"[{i}/{len(ts)}] {r['session_dir']} ref={r.get('ref_class')}/{r.get('ref_radar')} vlm={v.get('group')}/{v.get('class_name')}/{v.get('radar')} {flag} {r.get('error') or ''} | {tally}")
    elapsed = time.time() - t0
    if not a.no_db:
        save_db(run, a.source, recs)
    p, _ = summarize(run, a.source, recs, tally, elapsed)
    print(f'\n완료: {len(recs)}프레임 {elapsed/60:.1f}분 {tally}\n요약: {p}\n원본: {outp}')


if __name__ == '__main__':
    main()
