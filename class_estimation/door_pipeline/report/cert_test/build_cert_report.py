"""공인 시험·자체 평가 입회용 보고 페이지 생성 — report/cert_test/report.html (정적, 오프라인 열람 가능).

  python report/cert_test/build_cert_report.py --split split_20261106.json --run rehearsal_20261106 [--witness witness_20261110]

입력: split_<seed>.json(tools/cert_split.py), runs/<tag>/results.json·cm_*.png(tools/cert_eval.py), position_evidence.json(수기 관리),
      decisions.csv(사람 판단 기록, sha256 계산). --witness 를 주면 입회 당일 재실행 결과를 사전 리허설 옆에 나란히 표시.
출력: report.html + web/<tag>_cm_hole.png·cm_hybrid.png. 아티팩트로 올릴 때는 web/ 파일을 함께 올린다.
"""
import argparse, csv, hashlib, html, json, os, shutil, time

HERE = os.path.dirname(os.path.abspath(__file__))
DOOR = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.normpath(os.path.join(DOOR, '..', '..'))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def esc(x):
    return html.escape('' if x is None else str(x))


def fmt(n):
    return f"{n:,}" if isinstance(n, int) else esc(n)


def pill(kind, text):
    return f'<span class="pill p-{kind}">{esc(text)}</span>'


def table(headers, rows, cls='', caption=None, numeric=()):
    th = ''.join(f'<th class="{"n" if i in numeric else ""}">{esc(h)}</th>' for i, h in enumerate(headers))
    body = ''
    for r in rows:
        body += '<tr>' + ''.join(f'<td class="{"n" if i in numeric else ""}">{c if isinstance(c, Raw) else esc(c)}</td>' for i, c in enumerate(r)) + '</tr>'
    cap = f'<caption>{esc(caption)}</caption>' if caption else ''
    return f'<div class="tw"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody>{cap}</table></div>'


class Raw(str):
    pass


def load_run(tag):
    if not tag:
        return None
    d = os.path.join(HERE, 'runs', tag)
    p = os.path.join(d, 'results.json')
    if not os.path.exists(p):
        return None
    r = json.load(open(p, encoding='utf-8')); r['_dir'] = d; r['_tag'] = tag
    r['_sha'] = sha256(p)
    wrong = []
    with open(os.path.join(d, 'frames.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['correct_hybrid'] != '1':
                wrong.append(row)
    r['_wrong'] = wrong
    return r


def copy_png(run, web):
    out = {}
    for k in ('cm_hole', 'cm_hybrid'):
        src = os.path.join(run['_dir'], k + '.png')
        if os.path.exists(src):
            dst = os.path.join(web, f"{run['_tag']}_{k}.png"); shutil.copy(src, dst); out[k] = f"web/{run['_tag']}_{k}.png"
    return out


def run_block(run, title, imgs):
    if run is None:
        return f'<div class="box"><strong>{esc(title)}</strong><p class="note">아직 실행 전입니다. 입회 당일 아래 명령으로 실행하면 이 자리에 결과가 들어갑니다.</p></div>'
    h, y, c = run['hole'], run['hybrid'], run['cnn_alone']
    n = run['n_frames']
    rows = [
        ['홀 판별기 + CNN 폴백 (공식)', f"{y['correct']:,} / {n:,}", f"{y['accuracy']:.2f}%", f"홀 판정 {y['from_hole']:,}장 · CNN 폴백 {y['from_cnn']:,}장(정답 {y['cnn_correct_when_used']})"],
        ['홀 판별기 단독 (보류 = 오답)', f"{h['correct']:,} / {n:,}", f"{h['accuracy_all']:.2f}%", f"판정 {h['judged']:,}장 = 판정률 {h['judged_rate']:.2f}%, 판정 정확도 {h['accuracy_judged']:.2f}%, unknown {h['unknown']}"],
        ['CNN seed916 단독 (참고)', f"{c['correct']:,} / {n:,}", f"{c['accuracy']:.2f}%", '외관 폴백 모델만 썼을 때'],
        ['세션 다수결 (참고)', f"{run['sessions']['correct']} / {run['sessions']['n']}", f"{100.0 * run['sessions']['correct'] / max(1, run['sessions']['n']):.1f}%", '세션당 표본 ≤20장 다수결'],
    ]
    pc = [[k.replace('_door_', ' '), v['n'], v['hybrid_correct'], f"{100.0 * v['hybrid_correct'] / max(1, v['n']):.1f}%", v['hole_correct'], f"{100.0 * v['hole_correct'] / max(1, v['n']):.1f}%"]
          for k, v in run['per_class'].items()]
    gates = ', '.join(f"{k} {v}" for k, v in sorted(run['hole']['gates'].items(), key=lambda t: -t[1]))
    wrong = run['_wrong']
    wrong_tbl = table(['세션', '파일', '정답', '홀 판정', '게이트', 'CNN 판정(확률)', '하이브리드'],
                      [[w['session_dir'], w['file'].split('/')[-1], w['gt'].replace('_door_', ' '), (w['hole_pred'] or '보류').replace('_door_', ' '), w['gate'],
                        f"{w['cnn_pred'].replace('_door_', ' ')} ({w['cnn_prob']})", w['hybrid_pred'].replace('_door_', ' ')] for w in wrong[:60]],
                      caption=f"하이브리드 오답 {len(wrong)}장" + (' (앞 60장)' if len(wrong) > 60 else '')) if wrong else '<p class="note">하이브리드 오답 0장.</p>'
    env = run['env']
    return f'''
<div class="box">
  <strong>{esc(title)}</strong> <span class="note">실행 {esc(run['started_at'])} → {esc(run['finished_at'])} ({run['elapsed_s']}s), 테스트 {run['n_sessions']}세션 {n:,}장, 분할 시드 {run['split_seed']}</span>
  {table(['판정기', '정답 / 전체', '인식율', '비고'], rows, numeric=(1, 2))}
  <div class="charts2">
    <figure><div class="shot"><img src="{imgs.get('cm_hybrid', '')}" alt="하이브리드 혼동행렬" loading="lazy"></div><figcaption><b>혼동행렬 — 홀 + CNN 폴백(공식).</b> 행 = DB 확정 라벨, 열 = 예측.</figcaption></figure>
    <figure><div class="shot"><img src="{imgs.get('cm_hole', '')}" alt="홀 판별기 단독 혼동행렬(보류 열 포함)" loading="lazy"></div><figcaption><b>혼동행렬 — 홀 판별기 단독.</b> 마지막 열 '보류'는 게이트 미통과(분모에 포함). 게이트 분포: {esc(gates)}.</figcaption></figure>
  </div>
  {table(['클래스', '장수', '하이브리드 정답', '인식율', '홀 단독 정답', '홀 단독 인식율'], pc, numeric=(1, 2, 3, 4, 5))}
  {wrong_tbl}
  <p class="note">환경: {esc(env.get('host'))} · {esc(env.get('gpu'))} · PyTorch {esc(env.get('torch'))} / CUDA {esc(env.get('cuda'))} · Python {esc(env.get('python'))} · git {esc(env.get('git'))} · 스케일 모드 {esc(env.get('scale_mode'))}<br>
  모델 해시: 홀 판별기 <code>{esc(env.get('hole_model_sha256', '')[:16])}…</code> ({esc(env.get('hole_model'))}) · CNN <code>{esc(env.get('cnn_model_sha256', '')[:16])}…</code> ({esc(env.get('cnn_model'))})<br>
  결과 파일 results.json sha256 <code>{esc(run['_sha'][:16])}…</code>, 분할 파일 sha256 <code>{esc(run['split_sha256'][:16])}…</code></p>
</div>'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True, help='split_<seed>.json 파일명(report/cert_test 기준)')
    ap.add_argument('--run', required=True, help='사전 리허설 runs/<tag>')
    ap.add_argument('--witness', default=None, help='입회 당일 재실행 runs/<tag>')
    ap.add_argument('--out', default=os.path.join(HERE, 'report.html'))
    a = ap.parse_args()
    split = json.load(open(os.path.join(HERE, a.split), encoding='utf-8'))
    split_sha = sha256(os.path.join(HERE, a.split))
    pos = json.load(open(os.path.join(HERE, 'position_evidence.json'), encoding='utf-8'))
    run = load_run(a.run); wit = load_run(a.witness)
    web = os.path.join(HERE, 'web'); os.makedirs(web, exist_ok=True)
    imgs_r = copy_png(run, web) if run else {}
    imgs_w = copy_png(wit, web) if wit else {}
    dec = os.path.join(DOOR, 'decisions.csv')
    dec_sha = sha256(dec) if os.path.exists(dec) else None
    today = time.strftime('%Y-%m-%d')
    T = split['totals']; PC = split['per_class']

    # ── 인식율 탭 ──
    overview = table(['항목', '협약계획서(p.11·16·19)', '우리 구성'], [
        ['지표', '인식율 = 올바르게 예측된 샘플 ÷ 전체 테스트 샘플(혼동행렬 대각합 / 전체)', '같은 정의. 보류(게이트 미통과)도 분모에 포함해 오답으로 처리'],
        ['목표', '1단계-2차년도 80% 이상 (최종 99% 이상, 비중 10%)', Raw(f"사전 리허설 {run['hybrid']['accuracy']:.2f}%(하이브리드)" if run else '—')],
        ['평가자', '1단계-2차년도: 공인 시험기관 평가', '공인 시험기관 입회 하 재실행 + 사전 봉인 자료 열람'],
        ['데이터셋', '굴착기 부품 획득 이미지 기반, 학습:테스트 70:30 분할, 테스트 셋 추론', '명성공업 도어 용접 라인 무인 수집 현장 데이터(동결 372세션 36,770쌍, 세션당 표본 ≤20장). 세션 단위 70:30, 테스트 풀은 두 모델 모두 미학습 세션'],
        ['대상 부품', '굴착기 부품(주관기관 협의로 붐·센터프레임 → 도어로 변경)', '굴착기 도어 형상군 8종(E25·E30·E38 앞문·뒷문·우측; E23 은 수집 데이터셋에 없음), 품번 14종은 별도 표'],
        ['환경', 'GPU 워크스테이션, Windows/Linux, PyTorch, VS Code', Raw(f"세종 실험실 학습 PC {esc(run['env'].get('gpu')) if run else 'RTX 5090'}, Ubuntu, PyTorch {esc(run['env'].get('torch')) if run else ''}")],
    ])
    seal_rows = [
        ['동결 데이터셋 (DB door_factory_collect)', f"{T['sessions']}세션, 표본 {T['test_frames'] + T['train_val_frames'] + sum(s['n_frames'] for s in split['pool_unused']):,}장 (세션당 ≤20)", 'NAS 정본 + 학습 PC 미러, 야간 백업(로컬 14개·NAS 30일)'],
        ['정답(GT) 출처', '작업자 버튼 라벨 → 홀 판별기 다수결 대조 → 사람 육안 확인 정정', f"정정 누적 11세션(전부 판별기 적발), 테스트 세션 중 정정 이력 {sum(1 for s in split['test'] if s['label_corrected'])}건"],
        ['사람 판단 기록 decisions.csv', Raw(f"sha256 <code>{esc((dec_sha or '')[:16])}…</code>"), '라벨 정정·무효·분할·레이더 확정·품번, git 추적'],
        ['홀 랜드마크 판별기(4채널)', Raw(f"sha256 <code>{esc(run['env']['hole_model_sha256'][:16]) if run else ''}…</code>"), '실험실 134장 사람 라벨 + 현장 train 세션 의사 라벨(val 검증). test·미분할 세션 미사용'],
        ['CNN 폴백 seed916', Raw(f"sha256 <code>{esc(run['env']['cnn_model_sha256'][:16]) if run else ''}…</code>"), f"현장 train {PC and sum(v['train_val'] for v in PC.values())}세션(train 105·val 22) 학습, sqrt 가중치 레시피"],
        ['분할 파일', Raw(f"{esc(a.split)} · sha256 <code>{esc(split_sha[:16])}…</code>"), f"시드 {split['seed']}, 생성 {split['created_at']}"],
        ['코드', Raw(f"git <code>{esc(run['env'].get('git')) if run else ''}</code>"), 'tools/cert_split.py · tools/cert_eval.py · 이 페이지 build_cert_report.py'],
    ]
    pc_rows = [[k.replace('_door_', ' '), v['total'], v['train_val'], v['pool'], v['test'], v['test_frames']] for k, v in PC.items()]
    pc_rows.append(['합계', T['sessions'], T['train_val_sessions'], T['pool_sessions'], T['test_sessions'], T['test_frames']])
    test_list = ''.join(f"<li><code>{esc(s['session_dir'])}</code> {esc(s['class_name'].replace('_door_', ' '))} · {s['n_frames']}장" + (' · 레이더 O' if s['option_radar'] == 1 else ' · 레이더 X' if s['option_radar'] == 0 else '') + (' · 라벨 정정' if s['label_corrected'] else '') + '</li>' for s in split['test'])
    cmd = f"cd ~/parts_deep/class_estimation/door_pipeline\n$PY tools/cert_split.py --seed <입회자 시드>\n$PY tools/cert_eval.py --split report/cert_test/split_<시드>.json --tag witness_{today.replace('-', '')}\n$PY report/cert_test/build_cert_report.py --split split_<시드>.json --run {esc(a.run)} --witness witness_{today.replace('-', '')}"

    rec = f'''
<section id="r0"><h2><span class="sn">0</span>시험 개요 {pill('ok', '사전')}</h2>
<p>계획서 지표 6 '부품 인식율'의 정의·목표·평가 환경과 우리 구성을 나란히 둡니다. 시험 대상은 2026년 3월 주관기관·명성공업 협의로 굴착기 도어로 정했습니다.</p>{overview}</section>
<section id="r1"><h2><span class="sn">1</span>데이터셋·모델 봉인 {pill('ok', '사전')}</h2>
<p>테스트에 쓰는 데이터와 모델이 시험 전에 고정돼 있었음을 보이는 자료입니다. 파일 해시는 입회자가 현장에서 다시 계산해 대조합니다.</p>{table(['항목', '식별', '설명'], seal_rows)}</section>
<section id="r2"><h2><span class="sn">2</span>분할: 학습 70 · 테스트 30 {pill('ok', '사전 · 입회 당일 재추출 가능')}</h2>
<p>세션 단위로 나눕니다. 같은 세션의 프레임이 양쪽에 들어가지 않고, 테스트 풀에는 두 모델 모두 학습에 쓰지 않은 세션만 있습니다. 입회자가 시드를 정하면 같은 규칙으로 다시 뽑습니다.</p>
<p class="note">{esc(split['rule'])}</p>
{table(['클래스', '전체 세션', '학습측(train/val)', '테스트 풀', '테스트 추출', '테스트 장수'], pc_rows, numeric=(1, 2, 3, 4, 5), caption=f"시드 {split['seed']} · 테스트 {T['test_sessions']}세션 {T['test_frames']:,}장 (전체 세션의 {100.0 * T['test_sessions'] / T['sessions']:.0f}%)")}
<details><summary>테스트 세션 {T['test_sessions']}개 목록</summary><ul class="mono-list">{test_list}</ul></details></section>
<section id="r3"><h2><span class="sn">3</span>추론 실행 {pill('mid', '입회 당일 재실행')}</h2>
<p>테스트 세션 전 프레임을 한 번에 추론합니다. 홀 판별기가 먼저 판정하고, 게이트를 통과하지 못한 프레임만 CNN 폴백이 답합니다. 현장 서비스와 같은 우선순위입니다. 아래 명령 세 줄이 전부이며 몇 분 안에 끝납니다.</p>
<pre class="cmd">{esc(cmd)}</pre>
<p class="note">입회자가 확인할 것: 시드 입력 → 분할 파일 해시 → 추론 로그의 시작·종료 시각 → 결과 파일 해시. 사전 리허설과 당일 결과가 같은 양식으로 아래에 나란히 생성됩니다.</p></section>
<section id="r4"><h2><span class="sn">4</span>결과 {pill('ok', '사전 리허설')} {pill('mid', '입회 당일') if wit is None else pill('ok', '입회 당일 완료')}</h2>
{run_block(run, f"사전 리허설 — {a.run}", imgs_r)}
{run_block(wit, f"입회 당일 재실행 — {a.witness}", imgs_w) if a.witness else '<div class="box"><strong>입회 당일 재실행</strong><p class="note">당일 실행 뒤 <code>--witness</code> 태그로 이 페이지를 다시 생성하면 여기에 결과가 들어갑니다.</p></div>'}
<p class="take"><strong>읽는 법.</strong> 공식 수치는 홀 + CNN 폴백 하이브리드의 인식율(전체 분모)입니다. 홀 판별기 단독 행은 보류를 오답으로 센 보수적 수치이고, 판정 정확도는 답을 낸 프레임만의 정답률입니다. 올해 목표 80%, 최종 목표 99%.</p></section>
<section id="r5"><h2><span class="sn">5</span>보조 증빙 {pill('ok', '사전')}</h2>
{table(['항목', '수치', '출처'], [
  ['블라인드 누적(학습 제외, 8/27~10/29)', '245세션 5,026장 판정률 99.8% · 오판 0', 'report/hole_analysis/blind_eval_*, door_partno/report/blind_*'],
  ['고정 test(세션 분할, 9/23)', '23세션 596장 홀 100% · CNN seed916 99.8% · 품번 100%', 'DB evaluation_results'],
  ['VLM 독립 2차 감사(11/3, Claude Opus 5.5, 이미지만 제공)', '339세션 형상군 339/339 · 레이더 267/267 · 클래스 95.3%', 'report/vlm_audit/README.md'],
  ['현장 무인 추론 서비스(11/2~)', '그림자 모드 5일 37세션(판정 34), Jetson 홀 117ms·CNN 46ms', 'DOC_factory_infer.md, webapp /inference'],
  ['트래커 실측', '실물 도어 = CAD 6점 RMS 0.7~1.6mm, 픽셀 D 실물 대비 +0.6~1.2mm', 'pos_pipeline/GT_SAMPLE_20261002·20261008.md'],
])}
<p class="note">입회 당일 보조 시연: 실험실 실물 도어 앞 실시간 판정(18 서버) 1~2장, 경주 현장 서비스 대시보드·타임랩스 원격 열람.</p></section>
<section id="r6"><h2><span class="sn">6</span>입회 체크리스트</h2>
<ul class="check">{''.join(f'<li><label><input type="checkbox" id="rc{i}"> {esc(t)}</label></li>' for i, t in enumerate([
  '평가 절차서(지표 정의·데이터셋 요건·절차·이해상충 방지) 서명본 확인',
  '동결 데이터셋 세션 목록·GT 출처·정정 이력 열람',
  'decisions.csv·분할 파일·모델 2종 sha256 현장 재계산 대조',
  '입회자 시드로 분할 재추출, 테스트 세션 목록 확인(학습 세션 포함 0)',
  '추론 1회 실행 입회(시작·종료 시각, 로그)',
  '결과: 인식율(하이브리드·홀 단독), 혼동행렬, 오답 목록 확인',
  '결과 파일 해시 기록·서명, 사전 리허설 결과와 대조',
  '보조 시연(실물 도어 실시간 판정·현장 서비스 화면) 열람',
]))}</ul></section>'''

    # ── 위치 오차 탭 ──
    d = pos['definition']
    eq = table(['항목', '사양', '상태', '근거'], [[e['item'], e['spec'], Raw(pill('ok' if e['status'].startswith('완료') else 'mid', e['status'])), e['evidence']] for e in pos['equipment']])
    gt = table(['도어', '측정/분석', '홀', '반복 산포 중앙값(mm)', '실물 vs CAD 6점 RMS(mm)', '코너 D 실물−CAD', '픽셀 D 실물 대비', '깊이 3D 편향', '브래킷 간격 / 판 높이', '비고'],
               [[g['door'], g['date'], g['holes'], g['repeat_med_mm'], g['fit6_rms_mm'], g['corner_d_minus_cad_mm'] if g['corner_d_minus_cad_mm'] is not None else '—', g['pixel_d_vs_real_mm'], g['depth3d_bias_mm'],
                 f"{g['bracket_pitch_mm'] or '—'} / {g['plate_height_mm'] or '—'}", g['note']] for g in pos['gt_measurements']], cls='wide', numeric=(3,))
    cur = table(['지표', '값', '목표', '상태', '출처'], [[c['metric'], c['value'], c['target'], Raw(pill('ok' if c['status'] in ('충족',) else 'no' if '미달' in c['status'] else 'mid', c['status'])), c['source']] for c in pos['current_self_eval']])
    camp = pos['planned_campaign']
    posn = f'''
<section id="p0"><h2><span class="sn">0</span>지표 정의와 목표 {pill('ok', '사전')}</h2>
<p>{esc(d['text'])}</p>
<div class="formula">위치오차<sub>i</sub> = √[(x<sub>pred,i</sub> − x<sub>actual,i</sub>)² + (y<sub>pred,i</sub> − y<sub>actual,i</sub>)² + (z<sub>pred,i</sub> − z<sub>actual,i</sub>)²] ,&nbsp; 평균 위치 오차 = (1/n) Σ 위치오차<sub>i</sub> &nbsp;<span class="note">(정합 후, n = 기준점 수)</span></div>
{table(['연차', '목표 · 평가자'], [[k, v] for k, v in d['targets'].items()], caption=f"{d['plan_ref']} · 비중 {d['weight']} · 특징점: {d['feature_points']}")}</section>
<section id="p1"><h2><span class="sn">1</span>장비·교정·절차 {pill('ok', '대부분 완료')}</h2>{eq}</section>
<section id="p2"><h2><span class="sn">2</span>GT 측정 기록(레이저 트래커 SX-10) {pill('ok', '3장 완료')}</h2>
<p>샘플 도어 3장의 홀 좌표를 3회씩 쟀습니다. 실물이 CAD 와 1mm 안팎에서 일치하므로 현장 데이터의 CAD 기준 잔차를 자체 평가 수치로 쓸 수 있고, 영상 속 홀 간격으로 잰 거리 D 의 스케일이 전 기종에서 맞습니다.</p>{gt}</section>
<section id="p3"><h2><span class="sn">3</span>현재 자체 평가 수치 {pill('ok', '2차년도 목표 충족')} {pill('no', '최종 3mm 는 깊이 축 미달')}</h2>{cur}
<p class="take"><strong>해석.</strong> 2차년도 목표 10mm 는 현재 수치(정합 잔차 중앙값 4.6mm)로 충족합니다. 다만 이 값은 CAD 기준이고, 계획서 정의는 실측 좌표 기준이므로 아래 동시 측정 캠페인으로 같은 정의의 수치를 만들어 둡니다. 깊이 축 편향이 남아 최종 3mm 는 2단계 과제입니다.</p></section>
<section id="p4"><h2><span class="sn">4</span>{esc(camp['title'])} {pill('mid', camp['status'])}</h2>
<p><b>장소</b> {esc(camp['where'])}</p>
<ol>{''.join(f'<li>{esc(x)}</li>' for x in camp['design'])}</ol>
<div class="box"><strong>입회 당일 라이브 1회</strong><p>{esc(camp['live_demo'])}</p><strong>입회자</strong><p>{esc(camp['witness'])}</p><strong>위험</strong><p>{esc(camp['risk'])}</p></div>
<div class="box"><strong>결과 자리</strong><p class="note">캠페인 뒤 도어별·자세별 홀 오차 표와 전체 평균 위치 오차가 여기에 들어갑니다(05_evaluate_tracker_gt.py 출력).</p></div></section>
<section id="p5"><h2><span class="sn">5</span>입회 체크리스트</h2>
<ul class="check">{''.join(f'<li><label><input type="checkbox" id="pc{i}"> {esc(t)}</label></li>' for i, t in enumerate(pos['checklist']))}</ul></section>'''

    seal = table(['파일', 'sha256', '비고'], [
        [a.split, split_sha, '세션 분할(시드 %s)' % split['seed']],
        ['decisions.csv', dec_sha or '—', '사람 판단 기록'],
        [run['env']['hole_model'] if run else '홀 판별기', run['env']['hole_model_sha256'] if run else '—', '4채널 홀 랜드마크 검출기'],
        [run['env']['cnn_model'] if run else 'CNN', run['env']['cnn_model_sha256'] if run else '—', 'CNN seed916 폴백'],
        [f"runs/{a.run}/results.json", run['_sha'] if run else '—', '사전 리허설 결과'],
    ] + ([[f"runs/{a.witness}/results.json", wit['_sha'], '입회 당일 결과']] if wit else []), cls='wide')

    page = f'''<title>도어 판별 성능지표 입회 시험</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* 레이아웃: 탭 2개(인식율·위치 오차) + 봉인 탭, 각 탭은 계획서 평가 순서대로 절. 색 = 완료(초록)·당일(노랑)·미달(빨강) */
:root{{--bg:#f3f6f5;--surface:#fff;--sunk:#eaf0ee;--ink:#15211e;--ink2:#43524d;--muted:#67756f;--line:#d6dfdb;--accent:#0f6b5f;--accent-soft:#dff0ec;
--good:#1d7a4c;--good-bg:#e1f3e8;--warn:#9a6200;--warn-bg:#fbf0dc;--bad:#b4372b;--bad-bg:#f9e6e3;--s2:#e2672b;
--f-body:"IBM Plex Sans KR","Apple SD Gothic Neo","Malgun Gothic","Noto Sans KR",system-ui,sans-serif;--f-mono:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#101614;--surface:#171e1c;--sunk:#1e2724;--ink:#e4ebe8;--ink2:#b7c3be;--muted:#8b9a94;--line:#2b3632;--accent:#6fcdbd;--accent-soft:#17302b;--good:#5cc995;--good-bg:#16302a;--warn:#e3a849;--warn-bg:#33281a;--bad:#f07a6b;--bad-bg:#3a1f1c;--s2:#e8703a;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#101614;--surface:#171e1c;--sunk:#1e2724;--ink:#e4ebe8;--ink2:#b7c3be;--muted:#8b9a94;--line:#2b3632;--accent:#6fcdbd;--accent-soft:#17302b;--good:#5cc995;--good-bg:#16302a;--warn:#e3a849;--warn-bg:#33281a;--bad:#f07a6b;--bad-bg:#3a1f1c;--s2:#e8703a;color-scheme:dark}}
*{{box-sizing:border-box}}
body{{background:var(--bg);color:var(--ink);font-family:var(--f-body);font-size:16px;line-height:1.7;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:0 64px}}
main,header.top,footer{{max-width:980px;margin:0 auto}}
header.top{{padding-block:36px 18px;border-bottom:1px solid var(--line)}}
.eyebrow{{font-family:var(--f-mono);font-size:12.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}}
h1{{font-size:30px;line-height:1.25;margin:6px 0 10px;font-weight:700;text-wrap:balance}}
h2{{font-size:21px;margin:40px 0 8px;font-weight:700;text-wrap:balance;scroll-margin-top:16px}}
h2 .sn{{font-family:var(--f-mono);font-weight:500;color:var(--accent);margin-right:10px;font-size:18px}}
p{{margin:8px 0 12px;max-width:72ch}} .lede{{font-size:17px;color:var(--ink2);max-width:70ch}}
.meta{{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13.5px;color:var(--muted);margin-top:10px}} .meta b{{color:var(--ink2);font-weight:500}}
a{{color:var(--accent)}} code{{font-family:var(--f-mono);font-size:.88em}} .note{{font-size:13.5px;color:var(--muted)}}
.tabs{{display:flex;gap:6px;margin:18px 0 0;flex-wrap:wrap}}
.tabs a{{text-decoration:none;color:var(--ink2);padding:8px 14px;border:1px solid var(--line);border-bottom:0;border-radius:8px 8px 0 0;background:var(--sunk);font-weight:600;font-size:15px}}
.tabs a.on{{background:var(--surface);color:var(--accent);border-color:var(--accent)}}
.pane{{display:none}} .pane.on{{display:block}}
.tw{{overflow-x:auto;margin:12px 0 16px;border:1px solid var(--line);border-radius:8px;background:var(--surface)}}
table{{border-collapse:collapse;width:100%;font-size:14px;line-height:1.5}} th,td{{padding:8px 11px;text-align:left;vertical-align:top;border-bottom:1px solid var(--line)}}
thead th{{background:var(--sunk);font-weight:600;color:var(--ink2);font-size:13px;white-space:nowrap}} tbody tr:last-child td{{border-bottom:0}}
td.n,th.n{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}} caption{{caption-side:bottom;text-align:left;padding:8px 11px;font-size:13px;color:var(--muted)}} .wide{{min-width:760px}}
.pill{{display:inline-block;font-size:12.5px;font-weight:600;padding:1px 8px;border-radius:999px;white-space:nowrap;line-height:1.6;vertical-align:middle}}
.p-ok{{background:var(--good-bg);color:var(--good)}} .p-mid{{background:var(--warn-bg);color:var(--warn)}} .p-no{{background:var(--bad-bg);color:var(--bad)}}
.p-ok::before{{content:"● "}} .p-mid::before{{content:"◐ "}} .p-no::before{{content:"○ "}}
.box{{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:14px 16px;margin:14px 0}}
.take{{border-left:4px solid var(--s2);padding:2px 0 2px 14px;margin:14px 0;color:var(--ink2)}} .take strong{{color:var(--ink)}}
pre.cmd{{background:var(--sunk);border:1px solid var(--line);border-radius:8px;padding:12px 14px;font-family:var(--f-mono);font-size:13px;overflow-x:auto;line-height:1.6}}
.charts2{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}} .charts2 figure{{margin:8px 0;min-width:0}}
.shot{{background:#fff;border:1px solid var(--line);border-radius:8px;padding:6px}} .shot img{{display:block;width:100%;height:auto}}
figcaption{{font-size:13px;color:var(--muted);margin-top:6px}} figcaption b{{color:var(--ink2)}}
details{{margin:8px 0}} summary{{cursor:pointer;color:var(--accent);font-weight:600}}
ul.mono-list{{font-family:var(--f-mono);font-size:12.5px;columns:2;column-gap:24px;padding-left:18px;max-width:none}} ul.mono-list li{{margin:1px 0;break-inside:avoid}}
ul.check{{list-style:none;padding:0;max-width:none}} ul.check li{{padding:6px 0;border-bottom:1px dashed var(--line)}} ul.check input{{margin-right:10px;transform:scale(1.15)}}
.formula{{font-family:var(--f-mono);font-size:14px;background:var(--sunk);border-radius:8px;padding:10px 14px;margin:10px 0;overflow-x:auto}}
footer{{margin-top:48px;border-top:1px solid var(--line);padding-top:14px;font-size:13px;color:var(--muted)}}
:focus-visible{{outline:2px solid var(--accent);outline-offset:2px}}
@media (max-width:640px){{body{{font-size:15px}} h1{{font-size:24px}} .charts2{{grid-template-columns:minmax(0,1fr)}} ul.mono-list{{columns:1}}}}
</style>
<header class="top">
  <div class="eyebrow">굴착기 혼류 생산 과제(25453759) · 공동연구개발기관 4 한국건설기계연구원 · 1단계-2차년도 · {today} 생성</div>
  <h1>도어 판별 성능지표 입회 시험</h1>
  <p class="lede">계획서의 두 성능지표(6. 부품 인식율, 7. 부품 위치 추정 오차)를 입회자 앞에서 순서대로 보고하는 페이지입니다. 사전에 봉인한 자료와 리허설 결과를 먼저 보이고, 인식율은 같은 명령으로 그 자리에서 다시 실행해 결과를 나란히 붙입니다. 위치 오차는 트래커 측정 기록을 검토하고 라이브 1회로 확인합니다.</p>
  <div class="meta"><span><b>장소</b> 세종 실험실 학습 PC</span><span><b>대상</b> 굴착기 도어(주관기관 협의로 변경)</span><span><b>분할 시드</b> {split['seed']}</span><span><b>리허설</b> {esc(a.run)}</span></div>
  <nav class="tabs" aria-label="탭"><a href="#recognition" data-tab="recognition">6. 부품 인식율</a><a href="#position" data-tab="position">7. 부품 위치 추정 오차</a><a href="#seal" data-tab="seal">봉인 파일 해시</a></nav>
</header>
<main>
<div class="pane" id="pane-recognition">{rec}</div>
<div class="pane" id="pane-position">{posn}</div>
<div class="pane" id="pane-seal"><h2><span class="sn">#</span>봉인 파일 해시</h2><p>입회자가 현장에서 <code>sha256sum</code> 으로 다시 계산해 대조합니다.</p>{seal}</div>
</main>
<footer>근거: 협약계획서 p.11~19(docs/부품분류,위치추정_성능지표.pdf) · tools/cert_split.py · tools/cert_eval.py · report/cert_test/position_evidence.json · pos_pipeline/GT_SAMPLE_20261002·20261008.md · GT_TRACKER_PROTOCOL.md. 페이지 생성 build_cert_report.py ({today}).</footer>
<script>
(function(){{
  var tabs=document.querySelectorAll('.tabs a'),panes=document.querySelectorAll('.pane');
  function show(){{var h=(location.hash||'#recognition').replace('#','');if(!document.getElementById('pane-'+h))h='recognition';
    tabs.forEach(function(t){{t.classList.toggle('on',t.dataset.tab===h)}});panes.forEach(function(p){{p.classList.toggle('on',p.id==='pane-'+h)}});}}
  window.addEventListener('hashchange',show);show();
  document.querySelectorAll('ul.check input').forEach(function(c){{try{{c.checked=localStorage.getItem('cert:'+c.id)==='1'}}catch(e){{}}
    c.addEventListener('change',function(){{try{{localStorage.setItem('cert:'+c.id,c.checked?'1':'0')}}catch(e){{}}}});}});
}})();
</script>'''
    open(a.out, 'w', encoding='utf-8').write(page)
    print('저장', a.out, f"({os.path.getsize(a.out) // 1024}KB)", '이미지', list(imgs_r.values()) + list(imgs_w.values()))


if __name__ == '__main__':
    main()
