"""cron: 현장 추론 세션의 VLM 독립 2차 감사 — ingest_infer 뒤에 돌려 새 세션만 감사한다 (2026-11-03).

학습 PC crontab 의 ingest_infer 줄 뒤에 붙는다(30분). 제드 박스는 건드리지 않는다.
  - 새 세션 = inference_results 에 있고 vlm_audits(source=infer) 에 없는 세션. NAS 에서 rgb 2장 pull → Claude → DB.
  - **무료 크레딧까지만**: UNTIL(기본 2026-11-17, 크레딧 만료일) 당일부터 API 를 부르지 않고 종료.
    지출 상한 CAP_USD(기본 100, 사용자 지정) 를 vlm_audits 누적 + 배치분(OTHER_SPEND) 이 넘어도 종료.
  - 잠금 파일로 중복 실행 방지. 로그 한 줄: 세션 수·불일치·비용·누적.

  python vlm/cron_audit_infer.py            # cron 용 (새 세션 없으면 조용히 종료)
  python vlm/cron_audit_infer.py --dry-run  # 대상만
  python vlm/cron_audit_infer.py --force 20261103/s_143920   # 이미 감사한 세션도 다시(시험용)
"""
import argparse
import datetime as dt
import fcntl
import json
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlm_common as C
import audit_sessions as A

UNTIL = os.environ.get('VLM_AUDIT_UNTIL', '2026-11-17')          # 이 날짜부터 중지(KST). 크레딧 만료일
CAP_USD = float(os.environ.get('VLM_AUDIT_CAP_USD', '100'))       # 누적 지출 상한 (사용자 조정 11/3: 180→100)
OTHER_SPEND = 6.0                                                  # DB 에 없는 지출: 인스트럭션 배치 $5.86 + 파일럿 $0.11 (11/3)
FRAMES, WORKERS = 2, 2
LOCK = os.path.join(C.ART, '.cron_audit.lock')
STATE = os.path.join(C.ART, 'cron_state.json')


def log(msg):
    print(f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}', flush=True)


def spent(con):
    try:
        return (con.execute('SELECT COALESCE(SUM(cost_usd),0) FROM vlm_audits').fetchone()[0] or 0.0) + OTHER_SPEND
    except sqlite3.OperationalError:
        return OTHER_SPEND


def new_sessions(con, force):
    if force:
        return set(force)
    try:
        done = {r[0] for r in con.execute("SELECT DISTINCT session_dir FROM vlm_audits WHERE source='infer'")}
    except sqlite3.OperationalError:
        done = set()
    rows = con.execute("""SELECT s.session_dir FROM inference_results r JOIN capture_sessions s ON s.id = r.session_id
                          ORDER BY r.started_at""").fetchall()
    return {r[0] for r in rows if r[0] not in done}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--force', nargs='*', help='세션 dir(이미 감사한 것도 다시)')
    a = ap.parse_args()
    today = dt.date.today().isoformat()
    if today >= UNTIL:
        log(f'중지: 크레딧 만료일({UNTIL}) 도달 — API 호출 안 함 (연장하려면 VLM_AUDIT_UNTIL)'); return
    os.makedirs(C.ART, exist_ok=True)
    lf = open(LOCK, 'w')
    try:
        fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log('건너뜀: 이전 실행이 아직 진행 중'); return
    con = sqlite3.connect(A.DB)
    used = spent(con)
    if used >= CAP_USD:
        log(f'중지: 누적 지출 ${used:.2f} ≥ 상한 ${CAP_USD:.0f}'); return
    only = new_sessions(con, a.force)
    con.close()
    if not only:
        log('새 세션 없음'); return
    ts = A.targets_infer(FRAMES, 0, only)
    log(f'대상 세션 {len(only)} · 프레임 {len(ts)} · 누적 지출 ${used:.2f}/{CAP_USD:.0f}')
    if a.dry_run or not ts:
        for t in ts:
            print('  ', t['session_dir'], os.path.basename(t['frame']), t['ref_status'], t['ref_class'])
        return
    client = C.get_client()
    run = C.run_id(); prefix = C.reference_blocks(); tally = C.Tally(); recs = []; t0 = time.time()
    outp = os.path.join(C.ART, f'audit_infer_cron_{run}.jsonl')
    from concurrent.futures import ThreadPoolExecutor, as_completed
    # 첫 호출은 단독(참조 15장 캐시 생성) → 나머지 병렬(캐시 읽기, 쓰기 1.25x 중복 방지)
    first = A.ask(client, prefix, ts[0]); recs.append(first); tally.add(first['usage'], first['cost_usd'])
    with open(outp, 'a', encoding='utf-8') as fo, ThreadPoolExecutor(max_workers=WORKERS) as ex:
        fo.write(json.dumps(first, ensure_ascii=False) + '\n')
        futs = {ex.submit(A.ask, client, prefix, t): t for t in ts[1:]}
        for fu in as_completed(futs):
            t = futs[fu]
            try:
                r = fu.result()
            except Exception as e:
                r = dict(t); r.update({'vlm': None, 'error': f'exc:{type(e).__name__}:{e}', 'usage': {}, 'cost_usd': 0.0})
            recs.append(r); fo.write(json.dumps(r, ensure_ascii=False) + '\n')
            if r.get('usage'):
                tally.add(r['usage'], r['cost_usd'])
    A.save_db(run, 'infer', recs)
    # 세션 단위 요약 한 줄씩
    by = {}
    for r in recs:
        by.setdefault(r['session_dir'], []).append(r)
    n_diff = 0
    for sd, rs in sorted(by.items()):
        vs = [r.get('vlm') or {} for r in rs]
        g, c, rd = A.vote([v.get('group') for v in vs]), A.vote([v.get('class_name') for v in vs]), A.vote([v.get('radar') for v in vs])
        ref_c, st = rs[0].get('ref_class'), rs[0].get('ref_status')
        an = sorted({x for v in vs for x in (v.get('anomalies') or [])})
        diff = ref_c in C.CAD_D and (g != C.GROUP[ref_c] or c != ref_c)
        n_diff += diff
        log(f"  {sd} 07={st}/{ref_c} VLM={g}/{c}/{rd}{' **불일치**' if diff else ''}{(' 이상=' + ','.join(an)) if an else ''} | {(vs[0].get('description') or '')[:90]}")
    total = used + tally.usd
    log(f'완료: 세션 {len(by)} 프레임 {len(recs)} 불일치 {n_diff} {tally} 소요 {time.time()-t0:.0f}s 누적 ${total:.2f}/{CAP_USD:.0f} → {outp}')
    json.dump({'last_run': run, 'sessions': len(by), 'cost_usd': tally.usd, 'cumulative_usd': total, 'until': UNTIL, 'cap_usd': CAP_USD},
              open(STATE, 'w'), ensure_ascii=False)


if __name__ == '__main__':
    main()
