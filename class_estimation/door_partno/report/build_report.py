"""연구 결과 요약 — report/ 의 검출기·CNN·CV 결과 json 을 한 표로 모아 report/README.md 를 만든다.
  python report/build_report.py
"""
import glob, json, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
L = [f"# 학습형 품번·레이더 판정 연구 결과 ({time.strftime('%Y-%m-%d %H:%M')})", "",
     "기준선 = 규칙 검사(door_pipeline/partno/radar_check.py, 6점 투영 + 밝기·판 높이). GT = 사용자 확정 레이더 × DB 클래스. 고정 test = 세션 단위 분할 23세션 596장.", ""]
h = os.path.join(HERE, 'hole4_test.json')
if os.path.exists(h):
    r = json.load(open(h)); f, hd, s = r['frame'], r['hard'], r['session']
    L += ["## 4채널 홀 판별기(브래킷 채널) vs 규칙 — 고정 test", "",
          "| 구간 | n | 레이더 GT n | 검출기 판정률 | 검출기 정확도 | 규칙 판정률 | 규칙 정확도 | 품번 판정률 | 품번 정확도 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for q in (f, hd):
        L.append(f"| {q['label']} | {q['n']} | {q['radar_n']} | {100 * q['det_judged'] / max(1, q['radar_n']):.1f}% | {q['det_acc']:.1f}% | {100 * q['rule_judged'] / max(1, q['radar_n']):.1f}% | {q['rule_acc']:.1f}% | {100 * q['part_judged'] / max(1, q['part_n']):.1f}% | {q['part_acc']:.1f}% |")
    L += ["", f"세션 단위(GT {s['n']}): 검출기 {s['det_judged']}판정·{s['det_acc']:.1f}% / 규칙 {s['rule_judged']}판정·{s['rule_acc']:.1f}%. 형상군 판정 {f['cls_judged']}/{f['n']}·{f['cls_acc']:.1f}%. {r['ms_per_frame']}ms/장", ""]
ev = os.path.join(os.path.dirname(HERE), 'attribute_models', 'hole_landmarks_bracket', 'eval.json')
if os.path.exists(ev):
    e = json.load(open(ev)); ho = e['holdout']
    L += [f"검출기 6점 정밀도(사람 홀드아웃 {ho['n']}장): ≤8px {100 * (ho['six_le8'] or 0):.1f}%, 중앙값 {ho['six_med'] or 0:.1f}px · 브래킷 점 오차 중앙값(val) {e['val']['bracket_med'] if e['val']['bracket_med'] is not None else float('nan'):.1f}px", ""]
cn = sorted(glob.glob(os.path.join(HERE, 'cnn_partno_*_test.json')))
if cn:
    L += ["## CNN — 고정 test (프레임 품번 정확도 / 세션 다수결)", "", "| run | 프레임 품번 | n | 클래스 | 레이더 | 세션 품번 | ms/장 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for p in cn:
        r = json.load(open(p))
        if '_fold' in r['run']: continue
        L.append(f"| {r['run']} | {r['part_acc']:.1f}% | {r['part_n']} | {r['cls_acc'] if r['cls_acc'] is None else round(r['cls_acc'], 1)} | {r['radar_acc'] if r['radar_acc'] is None else round(r['radar_acc'], 1)} | {r['session_acc']:.1f}% ({r['session_n']}) | {r['ms_per_frame']} |")
    L.append("")
cv = sorted(glob.glob(os.path.join(HERE, 'cv_*.json')))
if cv:
    L += ["## CNN 세션 5-fold CV", ""]
    for p in cv:
        r = json.load(open(p)); tot = sum(v['part_n'] for v in r.values()); ok = sum(v['part_ok'] for v in r.values())
        L.append(f"- {os.path.basename(p)}: 품번 합산 {ok}/{tot} = **{100 * ok / max(1, tot):.1f}%**, fold별 {[round(v['part_acc'], 1) for v in r.values()]}")
    L.append("")
L += ["## 해석 (2026-09-23)", "",
      "- CNN 의 블라인드 오류는 전부 형상군(E30↔E38 LH FRT, 폭 차 47mm)이고 레이더 헤드 오류는 0. 학습률 1e-4 재실험: 2에폭 체크포인트(seed916 에 가까움) 99.8%, 오래 학습한 run 은 헤드 유무와 무관하게 91%대 → 학습 세션 외관에 과적합하는 CNN 의 구조적 한계.",
      "- 홀 거리 규칙과 4채널 검출기는 같은 세션에서 100% 유지 → 형상군은 기하, 레이더는 브래킷 채널(+규칙 폴백)로 운영.", ""]
bl = sorted(glob.glob(os.path.join(HERE, 'blind_*.json')))
if bl:
    L += ["## 블라인드 평가 (학습에 쓰지 않은 세션)", ""]
    for p in bl:
        r = json.load(open(p))
        L += [f"### {r['tag']} — 세션 {len(r['sessions'])}, 검출기 {'블라인드' if r['detector_blind'] else '비블라인드'}", "",
              "| 모델 | 프레임 | 판정률 | 품번 정확도 | 세션 | 세션 정확도 |", "|---|---:|---:|---:|---:|---:|"]
        for m, v in r['results'].items():
            L.append(f"| {m} | {v['frames']} | {v['judged_rate']:.1f}% | {v['acc']:.1f}% | {v['sessions']} | {v['s_acc']:.1f}% |")
        L.append("")
open(os.path.join(HERE, 'README.md'), 'w', encoding='utf8').write('\n'.join(L) + '\n'); print('\n'.join(L))
