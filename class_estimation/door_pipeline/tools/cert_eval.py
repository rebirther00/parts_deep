"""공인 시험용 인식율 평가 — 분할 JSON 의 테스트 세션 전 프레임을 홀 판별기(1순위)와 CNN 폴백으로 추론해 계획서 정의대로 채점.

  python tools/cert_eval.py --split report/cert_test/split_20261106.json --tag rehearsal_20261106
  python tools/cert_eval.py --split ... --tag witness_YYYYMMDD           # 입회 당일 재실행(같은 명령)

산출(report/cert_test/runs/<tag>/):
  frames.csv        프레임별 GT·홀 판정(게이트·D·마진)·CNN 판정(확률)·하이브리드 판정·정오
  sessions.csv      세션별 장수·정답 수·세션 다수결
  results.json      전체 수치(인식율 = 정답 ÷ 전체 테스트 샘플, 보류 포함), 혼동행렬(JSON), 환경·모델 해시·분할 해시·시각
  cm_hole.png / cm_hybrid.png   혼동행렬 그림(보류 열 포함 / 하이브리드)

지표 정의(협약계획서 p.19): 인식율 = 올바르게 예측된 샘플 총합 / 전체 테스트 샘플 수 (혼동행렬 대각합 / 전체).
  - 홀 판별기 단독: 게이트 미통과(보류)는 분모에 남고 정답이 아님. unknown 판정도 오답.
  - 하이브리드(공식): 홀 판별기가 판정하면 그 값, 보류면 CNN seed916 판정 — 현장 서비스(07/18)와 같은 우선순위.
"""
import argparse, csv, hashlib, json, os, platform, subprocess, sys, time, collections

HERE = os.path.dirname(os.path.abspath(__file__))
DOOR = os.path.dirname(HERE)
sys.path.insert(0, DOOR); sys.path.insert(0, os.path.join(DOOR, 'db'))
import numpy as np, cv2, torch   # noqa: E402
import hole_classifier as hc     # noqa: E402
import cnn_classifier            # noqa: E402
import build_dataset as bd       # noqa: E402
from camera_utils import intrinsics_for_image   # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def git_rev():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=DOOR, text=True).strip()
    except Exception:
        return None


def confusion(rows, key, classes, extra=None):
    cols = list(classes) + (list(extra) if extra else [])
    M = np.zeros((len(classes), len(cols)), int)
    for r in rows:
        p = r[key] if r[key] in cols else (extra[-1] if extra else None)
        if p is None:
            continue
        M[classes.index(r['gt']), cols.index(p)] += 1
    return cols, M


def save_cm(classes, cols, M, path, title):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['axes.unicode_minus'] = False          # 그림 글자는 영문(한글 폰트 의존 제거), 설명은 HTML 캡션에
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * len(cols), 0.9 + 0.75 * len(classes)))
    rs = M.sum(1, keepdims=True).clip(min=1)
    ax.imshow(M / rs, cmap='Blues', vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(cols)):
            if M[i, j]:
                ax.text(j, i, str(M[i, j]), ha='center', va='center', fontsize=9, color='white' if M[i, j] / rs[i, 0] > 0.6 else 'black')
    lab = lambda c: {'보류': 'abstain'}.get(c, c).replace('_door_', '\n')
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([lab(c) for c in cols], fontsize=8, rotation=45, ha='right')
    ax.set_yticks(range(len(classes))); ax.set_yticklabels([c.replace('_door_', ' ') for c in classes], fontsize=8)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Ground truth (DB confirmed label)'); ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_cms(res, out_dir):
    n = res['n_frames']; h = res['hole']; y = res['hybrid']; cm = res['confusion']
    save_cm(cm['hole']['rows'], cm['hole']['cols'], np.array(cm['hole']['M']), os.path.join(out_dir, 'cm_hole.png'),
            f"Hole landmark classifier only - recognition rate {h['accuracy_all']:.2f}% (judged {h['judged']}/{n}, accuracy of judged {h['accuracy_judged']:.2f}%)")
    save_cm(cm['hybrid']['rows'], cm['hybrid']['cols'], np.array(cm['hybrid']['M']), os.path.join(out_dir, 'cm_hybrid.png'),
            f"Hole classifier + CNN fallback - recognition rate {y['accuracy']:.2f}% ({y['correct']}/{n})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=False)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out', default=os.path.join(DOOR, 'report', 'cert_test', 'runs'))
    ap.add_argument('--limit-sessions', type=int, default=0, help='디버그용')
    ap.add_argument('--replot-only', action='store_true', help='추론 없이 results.json 으로 혼동행렬 png 만 다시 그림')
    a = ap.parse_args()
    if a.replot_only:
        d = os.path.join(a.out, a.tag); plot_cms(json.load(open(os.path.join(d, 'results.json'), encoding='utf-8')), d); print('replot', d); return
    assert a.split, '--split 필요'
    split = json.load(open(a.split, encoding='utf-8'))
    out_dir = os.path.join(a.out, a.tag); os.makedirs(out_dir, exist_ok=True)
    t0 = time.time(); started = time.strftime('%Y-%m-%d %H:%M:%S')
    net, dev = hc.load_model()
    cnn = cnn_classifier.CNNClassifier(cnn_classifier.MODEL_PATH, device=str(dev))
    classes = sorted({s['class_name'] for s in split['test']})
    rows = []
    sessions = split['test'][:a.limit_sessions] if a.limit_sessions else split['test']
    for si, s in enumerate(sessions):
        sdir = bd.MIRROR / s['session_dir']
        rgbs = sorted(p for p in sdir.rglob('rgb_*.png')) if sdir.exists() else []
        if not rgbs:
            print(f"  [경고] 로컬 표본 없음: {s['session_dir']}"); continue
        for rp in rgbs:
            rgb = cv2.imread(str(rp))
            if rgb is None:
                continue
            dp = rp.with_name(rp.name.replace('rgb_', 'depth_'))
            depth = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED) if dp.exists() else None
            intr = intrinsics_for_image(str(rp), rgb.shape)
            hr = hc.classify(net, dev, rgb, depth, intrinsics=intr)
            hole_pred = hr['pred'] if hr['gate'] == 'ok' else None
            c_cls, c_prob, _ = cnn.predict(rgb)
            hyb = hole_pred if hole_pred is not None else c_cls
            rows.append(dict(session_dir=s['session_dir'], file=os.path.relpath(str(rp), str(bd.MIRROR)), gt=s['class_name'],
                             hole_pred=hole_pred, gate=hr['gate'], D_mm=None if hr['D_mm'] is None else round(float(hr['D_mm']), 1),
                             margin_mm=None if hr['margin_mm'] is None else round(float(hr['margin_mm']), 1),
                             cnn_pred=c_cls, cnn_prob=round(float(c_prob), 3), hybrid_pred=hyb,
                             hole_src='hole' if hole_pred is not None else 'cnn',
                             correct_hole=int(hole_pred == s['class_name']), correct_hybrid=int(hyb == s['class_name'])))
        if si % 10 == 0:
            print(f"  {si + 1}/{len(sessions)} {s['session_dir']} {time.time() - t0:.0f}s")
    n = len(rows)
    judged = [r for r in rows if r['hole_pred'] is not None]
    res = dict(
        kind='cert_eval', tag=a.tag, started_at=started, finished_at=time.strftime('%Y-%m-%d %H:%M:%S'), elapsed_s=round(time.time() - t0, 1),
        split_file=os.path.relpath(a.split, DOOR), split_sha256=sha256(a.split), split_seed=split.get('seed'),
        n_sessions=len(sessions), n_frames=n, classes=classes,
        hole=dict(judged=len(judged), judged_rate=round(100.0 * len(judged) / max(1, n), 2), correct=sum(r['correct_hole'] for r in rows),
                  accuracy_all=round(100.0 * sum(r['correct_hole'] for r in rows) / max(1, n), 2),
                  accuracy_judged=round(100.0 * sum(r['correct_hole'] for r in judged) / max(1, len(judged)), 2),
                  unknown=sum(1 for r in judged if r['hole_pred'] == hc.UNKNOWN),
                  gates=dict(collections.Counter(r['gate'] for r in rows))),
        hybrid=dict(correct=sum(r['correct_hybrid'] for r in rows), accuracy=round(100.0 * sum(r['correct_hybrid'] for r in rows) / max(1, n), 2),
                    from_hole=sum(1 for r in rows if r['hole_src'] == 'hole'), from_cnn=sum(1 for r in rows if r['hole_src'] == 'cnn'),
                    cnn_correct_when_used=sum(r['correct_hybrid'] for r in rows if r['hole_src'] == 'cnn')),
        cnn_alone=dict(correct=sum(1 for r in rows if r['cnn_pred'] == r['gt']), accuracy=round(100.0 * sum(1 for r in rows if r['cnn_pred'] == r['gt']) / max(1, n), 2)),
        per_class={c: dict(n=sum(1 for r in rows if r['gt'] == c), hole_correct=sum(r['correct_hole'] for r in rows if r['gt'] == c),
                           hybrid_correct=sum(r['correct_hybrid'] for r in rows if r['gt'] == c)) for c in classes},
        env=dict(host=platform.node(), os=platform.platform(), python=platform.python_version(), torch=torch.__version__,
                 cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                 git=git_rev(), hole_model=os.path.relpath(hc.MODEL_PATH, DOOR), hole_model_sha256=sha256(hc.MODEL_PATH),
                 cnn_model=os.path.relpath(cnn_classifier.MODEL_PATH, DOOR), cnn_model_sha256=sha256(cnn_classifier.MODEL_PATH),
                 scale_mode=getattr(hc, 'SCALE_DEFAULT', None)))
    # 세션별
    sess = []
    for s in sessions:
        rr = [r for r in rows if r['session_dir'] == s['session_dir']]
        if not rr:
            continue
        vote = collections.Counter(r['hybrid_pred'] for r in rr).most_common(1)[0][0]
        sess.append(dict(session_dir=s['session_dir'], gt=s['class_name'], n=len(rr), hole_judged=sum(1 for r in rr if r['hole_pred'] is not None),
                         hole_correct=sum(r['correct_hole'] for r in rr), hybrid_correct=sum(r['correct_hybrid'] for r in rr),
                         session_vote=vote, session_correct=int(vote == s['class_name'])))
    res['sessions'] = dict(n=len(sess), correct=sum(x['session_correct'] for x in sess))
    # 혼동행렬
    cols_h, M_h = confusion(rows, 'hole_pred', classes, extra=[hc.UNKNOWN, '보류'])
    cols_y, M_y = confusion(rows, 'hybrid_pred', classes, extra=[hc.UNKNOWN])
    res['confusion'] = dict(hole=dict(rows=classes, cols=cols_h, M=M_h.tolist()), hybrid=dict(rows=classes, cols=cols_y, M=M_y.tolist()))
    plot_cms(res, out_dir)
    with open(os.path.join(out_dir, 'frames.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(os.path.join(out_dir, 'sessions.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(sess[0].keys())); w.writeheader(); w.writerows(sess)
    jp = os.path.join(out_dir, 'results.json')
    json.dump(res, open(jp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    open(jp + '.sha256', 'w').write(f"{sha256(jp)}  results.json\n")
    print(f"\n[{a.tag}] 테스트 {len(sess)}세션 {n}장 ({res['elapsed_s']}s)")
    print(f"  홀 판별기 단독: 인식율 {res['hole']['accuracy_all']}% (정답 {res['hole']['correct']}/{n}, 판정 {len(judged)} = {res['hole']['judged_rate']}%, 판정 정확도 {res['hole']['accuracy_judged']}%, unknown {res['hole']['unknown']})")
    print(f"  홀 + CNN 폴백   : 인식율 {res['hybrid']['accuracy']}% (정답 {res['hybrid']['correct']}/{n}, CNN 사용 {res['hybrid']['from_cnn']}장 중 정답 {res['hybrid']['cnn_correct_when_used']})")
    print(f"  CNN 단독(참고)  : {res['cnn_alone']['accuracy']}%   세션 다수결 {res['sessions']['correct']}/{res['sessions']['n']}")
    print(f"  저장 {out_dir}")


if __name__ == '__main__':
    main()
