"""3번 — 인스트럭션 셋 데이터 생성 (AI 팩토리 체크리스트 1-7 옵션 항목).

세션별 사실표(확정 클래스·품번·레이더·CAD 폭·홀 거리 D·마진·자세 z/tilt·레이더 검사 점수) + 대표 프레임 1장을
Claude Opus 5.5 에 주고, **주어진 사실만으로** 한국어 질문/응답 쌍 4개(identify/reason/radar/inspect)를 쓰게 한다.
Batch API(50% 할인) 로 제출하고, 결과를 LLaVA 식 conversations JSONL 로 저장한다.

  python vlm/build_instruction_set.py --pilot 3            # 동기 호출 3세션 — 품질 확인
  python vlm/build_instruction_set.py --submit              # 전체 배치 제출 → artifacts/instruction_batch.json
  python vlm/build_instruction_set.py --collect             # 배치 완료 대기·수집 → artifacts/instruction_set_v0.jsonl + DOC_instruction_set.md
"""
import argparse
import collections
import csv
import glob
import json
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlm_common as C

DB = os.path.join(C.DOOR, 'db', 'door_pipeline.db')
MIRROR = os.path.join(C.DOOR, 'datasets_factory_collect')
BATCH_FILE = os.path.join(C.ART, 'instruction_batch.json')
OUT_JSONL = os.path.join(C.ART, 'instruction_set_v0.jsonl')
DOC = os.path.join(C.DOOR, 'DOC_instruction_set.md')

SYSTEM = """너는 굴착기 사이드 도어 용접 라인의 검사 AI 를 학습시키기 위한 **인스트럭션 데이터셋 작성자**다.
입력: 도어 이미지 1장(지그 위에서 내려다본 RGB) + 그 세션의 확정 사실표(JSON).
출력: 한국어 질문/응답 쌍 4개. 각 쌍의 type 은 아래 넷을 하나씩 쓴다.
- identify : 이 도어가 무엇인지(형상군·클래스·품번·품명) 묻고 답한다.
- reason   : 왜 그 클래스인지 — 코너 홀 거리 D(측정)와 CAD 기준 D·마진·후보 간 차이를 근거로 설명한다.
- radar    : 레이더 옵션 유무와 그 근거(브래킷 판·검사 점수), 품번이 어떻게 갈리는지. FRT 면 "옵션 없음"임을 설명.
- inspect  : 이미지에서 보이는 상태(클램프·도어 방향·그릴·브래킷·가림·조명 등)를 묻고 사실대로 묘사한다.
규칙: 숫자·품번·명칭은 사실표에 있는 값만 쓴다(지어내지 말 것). 사실표에 없는 것은 "확인 필요"라고 답한다.
질문은 현장 작업자·품질 담당자가 실제로 물을 법한 자연스러운 문장으로, 응답은 2~5문장으로 정확하고 간결하게.
용어: 형상군(FRT/LH_RR/RH), 클래스(9종 이름), 품번(110982-xxxxx), 코너 홀 거리 D(mm), CAD D, 마진(mm), 레이더 브래킷.
JSON 으로만 답하라."""

SCHEMA = {
    'type': 'object',
    'properties': {'pairs': {'type': 'array', 'items': {
        'type': 'object',
        'properties': {'type': {'type': 'string', 'enum': ['identify', 'reason', 'radar', 'inspect']},
                       'instruction': {'type': 'string'}, 'response': {'type': 'string'}},
        'required': ['type', 'instruction', 'response'], 'additionalProperties': False}}},
    'required': ['pairs'], 'additionalProperties': False,
}


def facts():
    """decisions.csv(확정) + DB(세션 지표·레이더 검사) + part_numbers.json → 세션별 사실표"""
    pn = C.load_part_numbers()
    con = sqlite3.connect(DB); con.row_factory = sqlite3.Row
    hm = {r['session_dir']: dict(r) for r in con.execute(
        """SELECT s.session_dir, s.started_at, s.saved_pairs, m.n_frames, m.n_judged, m.d_pix_med, m.cad_d, m.dev_pix_mm, m.margin_pix_min,
                  m.z_med, m.tilt_med, m.pred_major, m.n_pred_major
           FROM capture_sessions s LEFT JOIN session_hole_metrics m ON m.session_id = s.id
           WHERE m.id = (SELECT MAX(id) FROM session_hole_metrics WHERE session_id = s.id) OR m.id IS NULL""")}
    rc = {r['session_dir']: dict(r) for r in con.execute(
        """SELECT s.session_dir, c.n_judged AS r_n, c.n_radar, c.n_none, c.score_med, c.auto_flag
           FROM capture_sessions s JOIN session_radar_check c ON c.session_id = s.id
           WHERE c.id = (SELECT MAX(id) FROM session_radar_check WHERE session_id = s.id)""")}
    con.close()
    out = []
    for r in csv.DictReader(open(os.path.join(C.DOOR, 'db', 'decisions.csv'), encoding='utf-8')):
        cls = r['class_db']
        if r['valid'] != '1' or cls not in C.CAD_D or not os.path.isdir(os.path.join(MIRROR, r['session_dir'])):
            continue
        radar = int(r['option_radar']) if r['option_radar'] not in ('', None) else None
        if C.GROUP[cls] != 'FRT' and radar is None:
            continue
        p = C.part_for(cls, radar, pn) or {}
        m = hm.get(r['session_dir'], {}); k = rc.get(r['session_dir'], {})
        others = sorted(((abs(C.CAD_D[c] - (m.get('d_pix_med') or C.CAD_D[cls])), c) for c in C.CAD_D if c != cls))[:2]
        f = {
            'session_dir': r['session_dir'], 'captured_at': m.get('started_at'),
            'group': C.GROUP[cls], 'class_name': cls, 'part_no': (p.get('part_no') + '-' + p.get('rev')) if p else r['part_no'],
            'part_name': p.get('name'), 'models': p.get('models'), 'door_width_mm_spec': p.get('width_mm'),
            'radar': {'FRT': '옵션 없음'}.get(C.GROUP[cls], ('장착(O)' if radar else '미장착(X)')),
            'part_no_by_radar': (None if C.GROUP[cls] == 'FRT' else {
                '레이더 장착(O)': (lambda q: (q['part_no'] + '-' + q['rev'] + ' ' + q['name']) if q else None)(pn.get((cls, 1))),
                '레이더 미장착(X)': (lambda q: (q['part_no'] + '-' + q['rev'] + ' ' + q['name']) if q else None)(pn.get((cls, 0)))}),
            'cad_D_mm': C.CAD_D[cls],
            'measured_D_mm': (round(m['d_pix_med'], 1) if m.get('d_pix_med') else None),
            'D_deviation_mm': (round(m['dev_pix_mm'], 1) if m.get('dev_pix_mm') is not None else None),
            'margin_to_next_class_mm': (round(m['margin_pix_min'], 1) if m.get('margin_pix_min') is not None else None),
            'nearest_other_classes': [{'class_name': c, 'cad_D_mm': C.CAD_D[c], 'diff_mm': round(d, 1)} for d, c in others],
            'frames_judged': f"{m.get('n_judged')}/{m.get('n_frames')}" if m.get('n_frames') else None,
            'pose_z_mm': (round(m['z_med'], 1) if m.get('z_med') else None), 'pose_tilt_deg': (round(m['tilt_med'], 1) if m.get('tilt_med') is not None else None),
            'radar_check': ({'frames_radar': k.get('n_radar'), 'frames_none': k.get('n_none'), 'score_med': round(k['score_med'], 3), 'auto_flag': k.get('auto_flag'),
                             'user_confirmed': r['option_source'] == 'user'} if k and C.GROUP[cls] != 'FRT' else None),
            'label_source': '작업자 버튼 라벨 + 홀 판별기 교차검증 + 사용자 확정(레이더 옵션)',
            'measurement_method': '홀 랜드마크 검출기(4채널) 6점 → 코너 홀 2점 픽셀 거리 × 고정 스케일(S_PIXEL=1477.8mm) → CAD D 최근접',
        }
        files = sorted(glob.glob(os.path.join(MIRROR, r['session_dir'], 'rgb_*.png')))
        f['image'] = os.path.relpath(files[len(files) // 2], C.DOOR) if files else None
        if f['image']:
            out.append(f)
    return out


def request_body(f):
    facts_txt = json.dumps({k: v for k, v in f.items() if k not in ('image',)}, ensure_ascii=False, indent=1)
    return dict(
        model=C.MODEL, max_tokens=6000,
        output_config={'effort': 'medium', 'format': {'type': 'json_schema', 'schema': SCHEMA}},
        system=[{'type': 'text', 'text': SYSTEM, 'cache_control': {'type': 'ephemeral'}}],
        messages=[{'role': 'user', 'content': [
            {'type': 'text', 'text': '도어 이미지:'},
            C.image_block(os.path.join(C.DOOR, f['image'])),
            {'type': 'text', 'text': '세션 사실표(JSON):\n' + facts_txt + '\n\n위 사실표와 이미지만 근거로 질문/응답 쌍 4개를 작성하라.'},
        ]}])


def to_record(f, pairs, model, src):
    conv = []
    for p in pairs:
        conv.append({'from': 'human', 'value': p['instruction'], 'type': p['type']})
        conv.append({'from': 'gpt', 'value': p['response']})
    return {'id': f['session_dir'].replace('/', '_'), 'image': f['image'], 'session_dir': f['session_dir'],
            'facts': {k: v for k, v in f.items() if k != 'image'}, 'conversations': conv,
            'generator': model, 'source': src, 'created_at': time.strftime('%Y-%m-%d %H:%M:%S')}


def write_doc(recs, n_sessions, cost_note):
    by_cls = collections.Counter(r['facts']['class_name'] for r in recs)
    by_type = collections.Counter(c.get('type') for r in recs for c in r['conversations'] if c['from'] == 'human')
    L = ['# 도어 검사 인스트럭션 셋 v0 (DOC_instruction_set.md)', '',
         f'작성: {time.strftime("%Y-%m-%d")}. AI 팩토리 데이터 체크리스트 1-7(옵션) "인스트럭션 셋 데이터" 산출물. 생성기 {C.MODEL}, 생성 스크립트 `vlm/build_instruction_set.py`.', '',
         '## 무엇인가', '',
         '- 현장 세션(사용자 확정 라벨·품번) 1건당 대표 프레임 1장 + 확정 사실표(클래스·품번·레이더·CAD D·측정 D·마진·자세·레이더 검사 점수)를 주고, **사실표 안의 값만으로** 한국어 질문/응답 4쌍(identify/reason/radar/inspect)을 쓰게 했다.',
         '- 형식: JSONL, 1행 = 1세션. `image`(door_pipeline 기준 상대경로) · `facts`(근거 사실표) · `conversations`(LLaVA 식 human/gpt 교대, human 에 `type`) · `generator` · `created_at`.',
         '- 용도: 시각-언어 검사 보조 모델의 지도 미세조정(SFT)·평가, 작업자 질의응답 데모. 라벨 정답은 사실표가 정본이며 응답 문장은 생성물이다.', '',
         '## 규모', '', f'- 세션 {len(recs)} / 대상 {n_sessions} · 질문/응답 쌍 {sum(by_type.values())}', f'- 유형별: ' + ', '.join(f'{k} {v}' for k, v in sorted(by_type.items())),
         '- 클래스별 세션: ' + ', '.join(f'{k} {v}' for k, v in sorted(by_cls.items())), f'- 비용: {cost_note}', '',
         '## 검수 방법(권장)', '', '- 응답의 숫자·품번이 `facts` 와 다르면 폐기(스크립트 `--verify` 로 숫자 포함 검사 가능 — 향후).',
         '- inspect 유형은 이미지 묘사라 사실표로 검증 불가 → 표본 육안 검수.', '',
         '## 파일', '', f'- `vlm/artifacts/instruction_set_v0.jsonl` (git 미추적, NAS 백업 권장)', '- 배치 ID·상태: `vlm/artifacts/instruction_batch.json`', '']
    open(DOC, 'w', encoding='utf-8').write('\n'.join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pilot', type=int, default=0, help='동기 호출로 N 세션만')
    ap.add_argument('--submit', action='store_true')
    ap.add_argument('--collect', action='store_true')
    ap.add_argument('--wait', type=int, default=0, help='--collect 시 완료까지 최대 대기(분)')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    fs = facts()
    print(f'대상 세션 {len(fs)} (확정 품번·로컬 미러 있음). 배치 예상 비용 ≈ ${len(fs)*0.03:.2f} (세션당 ~$0.06 의 50%)')
    if a.dry_run:
        print(json.dumps(fs[0], ensure_ascii=False, indent=1)); return
    os.makedirs(C.ART, exist_ok=True)
    if a.pilot:
        client = C.get_client(); tally = C.Tally(); recs = []
        for f in fs[:a.pilot]:
            resp = client.messages.create(**request_body(f))
            d, usd = C.usage_cost(resp.usage); tally.add(d, usd)
            if resp.stop_reason == 'refusal':
                print('refusal', f['session_dir']); continue
            pairs = json.loads(C.text_of(resp))['pairs']
            recs.append(to_record(f, pairs, resp.model, 'pilot'))
            print(f"\n=== {f['session_dir']} ({f['class_name']} / {f['part_no']} / radar={f['radar']}) ===")
            for p in pairs:
                print(f"[{p['type']}] Q: {p['instruction']}\n   A: {p['response']}")
        p = os.path.join(C.ART, 'instruction_pilot.jsonl')
        open(p, 'w', encoding='utf-8').write('\n'.join(json.dumps(r, ensure_ascii=False) for r in recs) + '\n')
        print(f'\n파일럿 {len(recs)}세션 {tally} → {p}')
        return
    if a.submit:
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request
        client = C.get_client()
        reqs = [Request(custom_id=f['session_dir'].replace('/', '__'), params=MessageCreateParamsNonStreaming(**request_body(f))) for f in fs]
        b = client.messages.batches.create(requests=reqs)
        json.dump({'batch_id': b.id, 'status': b.processing_status, 'n': len(reqs), 'submitted_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                   'facts_by_id': {f['session_dir'].replace('/', '__'): f for f in fs}}, open(BATCH_FILE, 'w', encoding='utf-8'), ensure_ascii=False)
        print(f'배치 제출: {b.id} status={b.processing_status} n={len(reqs)} → {BATCH_FILE}')
        return
    if a.collect:
        client = C.get_client()
        meta = json.load(open(BATCH_FILE, encoding='utf-8')); bid = meta['batch_id']
        t0 = time.time()
        while True:
            b = client.messages.batches.retrieve(bid)
            rc_ = b.request_counts
            print(f'{time.strftime("%H:%M:%S")} {b.processing_status} processing={rc_.processing} succeeded={rc_.succeeded} errored={rc_.errored}')
            if b.processing_status == 'ended' or not a.wait or (time.time() - t0) > a.wait * 60:
                break
            time.sleep(60)
        if b.processing_status != 'ended':
            print('아직 처리 중 — 나중에 --collect 다시 실행'); return
        recs, tally, errs = [], C.Tally(), collections.Counter()
        for res in client.messages.batches.results(bid):
            f = meta['facts_by_id'].get(res.custom_id)
            if res.result.type != 'succeeded':
                errs[res.result.type] += 1; continue
            msg = res.result.message
            d, usd = C.usage_cost(msg.usage); tally.add(d, usd * 0.5)
            if msg.stop_reason == 'refusal':
                errs['refusal'] += 1; continue
            try:
                pairs = json.loads(C.text_of(msg))['pairs']
            except Exception:
                errs['json'] += 1; continue
            recs.append(to_record(f, pairs, msg.model, f'batch:{bid}'))
        recs.sort(key=lambda r: r['id'])
        open(OUT_JSONL, 'w', encoding='utf-8').write('\n'.join(json.dumps(r, ensure_ascii=False) for r in recs) + '\n')
        meta.update({'status': 'ended', 'collected_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'n_ok': len(recs), 'errors': dict(errs), 'cost': str(tally)})
        json.dump(meta, open(BATCH_FILE, 'w', encoding='utf-8'), ensure_ascii=False)
        write_doc(recs, meta['n'], f'{tally} (배치 50% 적용)')
        print(f'수집: {len(recs)}세션 OK, 오류 {dict(errs)}, 비용 {tally} → {OUT_JSONL}\n문서: {DOC}')


if __name__ == '__main__':
    main()
