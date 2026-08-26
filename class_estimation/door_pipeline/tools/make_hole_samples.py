"""홀 판별기 결과 샘플 이미지 생성 (성공 / 오판 / 보류) + 학습 곡선.

입력: attribute_models/hole_landmarks/eval_classifier.json (16_evaluate_hole_classifier.py 출력)
출력: report/hole_analysis/samples/{success,failure,abstain}_*.jpg, montage_*.png, training_curve.png
"""
import json, os, sqlite3, sys, collections
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hole_classifier import load_model, classify, CAD_D

DOOR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(DOOR, 'report', 'hole_analysis', 'samples')
os.makedirs(OUT, exist_ok=True)
E = json.load(open(os.path.join(DOOR, 'attribute_models', 'hole_landmarks', 'eval_classifier.json')))
rows = E['datasets_all']['rows'] + E.get('datasets_field', {}).get('rows', [])
COL = {'bolt': (255, 0, 0), 'corner_hinge': (0, 0, 255), 'corner_latch': (0, 140, 255)}


def draw(net, dev, row, title):
    f = os.path.join(DOOR, row['image']); rgb = cv2.imread(f)
    dp = f.replace('rgb_', 'depth_'); depth = cv2.imread(dp, cv2.IMREAD_UNCHANGED) if os.path.exists(dp) else None
    r = classify(net, dev, rgb, depth)
    v = rgb.copy()
    for c, pts in r['points'].items():
        for p in pts:
            cv2.circle(v, (int(p[0]), int(p[1])), 9, COL[c], 2)
            cv2.putText(v, f'{p[2]:.2f}', (int(p[0]) + 10, int(p[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL[c], 1)
    h, l = r['points']['corner_hinge'], r['points']['corner_latch']
    if h and l:
        cv2.line(v, (int(h[0][0]), int(h[0][1])), (int(l[0][0]), int(l[0][1])), (0, 0, 255), 2)
    cv2.rectangle(v, (0, 0), (900, 78), (0, 0, 0), -1)
    cv2.putText(v, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(v, f"GT {row['cls']} | pred {r['pred']} | D={r['D_mm'] and round(r['D_mm'])}mm (CAD {CAD_D.get(row['cls'])}) | gate={r['gate']}",
                (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    return v


def montage(imgs, cols, path, w=640):
    tiles = [cv2.resize(i, (w, int(i.shape[0] * w / i.shape[1]))) for i in imgs]
    H = max(t.shape[0] for t in tiles); tiles = [cv2.copyMakeBorder(t, 0, H - t.shape[0], 0, 6, cv2.BORDER_CONSTANT, value=(30, 30, 30)) for t in tiles]
    while len(tiles) % cols: tiles.append(np.full_like(tiles[0], 30))
    g = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(path, g); print('wrote', path, g.shape)


if __name__ == '__main__':
    net, dev = load_model()
    succ = [r for r in rows if r['pred'] and r['pred'] == r['cls']]
    fail = [r for r in rows if r['pred'] and r['pred'] != r['cls']]
    abst = [r for r in rows if not r['pred'] and r['cls'] in CAD_D]
    # 성공: 클래스별 1장 (마진 큰 것)
    by = collections.defaultdict(list)
    for r in succ: by[r['cls']].append(r)
    S = [max(v, key=lambda r: r.get('margin_mm') or 0) for c, v in sorted(by.items())]
    S_img = [draw(net, dev, r, f'SUCCESS {r["cls"]}') for r in S]
    for r, im in zip(S, S_img): cv2.imwrite(os.path.join(OUT, f"success_{r['cls']}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 80])
    montage(S_img, 4, os.path.join(OUT, 'montage_success.png'))
    # 오판: 전부
    F_img = [draw(net, dev, r, f'FAILURE {r["cls"]} -> {r["pred"]}') for r in fail]
    for r, im in zip(fail, F_img): cv2.imwrite(os.path.join(OUT, f"failure_{os.path.basename(os.path.dirname(r['image']))}_{os.path.basename(r['image'])[4:8]}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if F_img: montage(F_img, 4, os.path.join(OUT, 'montage_failure.png'))
    # 보류: 사유별 1장
    byg = {}
    for r in abst: byg.setdefault(r['gate'], r)
    A = list(byg.values()); A_img = [draw(net, dev, r, f'ABSTAIN [{r["gate"]}] {r["cls"]}') for r in A]
    for r, im in zip(A, A_img): cv2.imwrite(os.path.join(OUT, f"abstain_{r['gate']}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 80])
    montage(A_img, 3, os.path.join(OUT, 'montage_abstain.png'))
    # 학습 곡선 (DB training_metrics, 최신 hole_landmarks 세션)
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        from matplotlib import font_manager as fm
        fp = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
        if os.path.exists(fp): fm.fontManager.addfont(fp); plt.rcParams['font.family'] = fm.FontProperties(fname=fp).get_name()
        c = sqlite3.connect(os.path.join(DOOR, 'db', 'door_pipeline.db'))
        sid = c.execute("select s.id from training_sessions s join models m on m.id=s.model_id where m.name='hole_landmarks_resnet18' and s.status='completed' order by s.id desc limit 1").fetchone()[0]
        m = c.execute('select epoch, train_loss, val_accuracy from training_metrics where session_id=? order by epoch', (sid,)).fetchall()
        ep = [x[0] for x in m]; tl = [x[1] for x in m]; va = [(x[0], x[2]) for x in m if x[2] is not None]
        fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150); ax.plot(ep, tl, color='#2F6DB5', label='train loss (weighted MSE)'); ax.set_yscale('log'); ax.set_xlabel('epoch'); ax.set_ylabel('loss')
        ax2 = ax.twinx(); ax2.plot([v[0] for v in va], [v[1] for v in va], 'o-', color='#C0392B', label='holdout ≤8px (%)'); ax2.set_ylim(0, 105); ax2.set_ylabel('holdout ≤8px (%)')
        ax.set_title(f'홀 랜드마크 검출기 학습 곡선 (training_sessions.id={sid})'); ax.grid(alpha=.3)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, loc='center right', fontsize=8)
        plt.tight_layout(); plt.savefig(os.path.join(OUT, 'training_curve.png')); print('wrote training_curve.png (session', sid, ')')
    except Exception as e:
        print('training curve skipped:', e)
    print(f'success {len(succ)} / failure {len(fail)} / abstain {len(abst)}')
