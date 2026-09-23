"""4채널 홀 랜드마크 검출기 학습 — door_pipeline 정본(3채널) 가중치에서 브래킷 채널을 추가해 미세조정.

  python 16_train_hole_landmarks.py                       # 학습 + 평가 → attribute_models/hole_landmarks_bracket/
  python 16_train_hole_landmarks.py --epochs 8 --lr 5e-5
  python 16_train_hole_landmarks.py --eval_only

데이터(15_make_pseudo_labels.py 출력, labels/holes_pseudo/*.json):
  - 의사 라벨(현장 세션 프레임): split=train → 학습, val → 검증, test → 제외(17 에서 평가), **split 미지정 → 제외(기본, CNN 과 동일 기준; --include-unsplit 로 포함)**.
    6점은 정본 검출기, 브래킷은 규칙 판정.
  - 사람 라벨 134장(human_*.json): door_pipeline split.json 의 holdout 은 제외(6점 정밀도 평가용), 나머지 학습.
  - bracket_known=false 프레임은 브래킷 채널 손실을 마스크(가림·미정을 음성으로 가르치지 않음).
손실: 채널별 양성 가중 MSE(정본과 동일) × 채널 마스크. 증강: 정본과 동일(회전·스케일·플립·밝기).
평가(에폭마다): 사람 홀드아웃 6점 ≤8px 비율, val 세션 브래킷 판정 정확도(radar/none 프레임 GT 대비) → 두 값 평균이 최대인 에폭 저장.
"""
import argparse, collections, glob, json, math, os, random, sys, time
import cv2, numpy as np, torch
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
DOOR = os.path.normpath(os.path.join(HERE, '..', 'door_pipeline'))
sys.path.insert(0, DOOR); sys.path.insert(0, os.path.join(DOOR, 'db')); sys.path.insert(0, HERE)
import hole_classifier4 as hc4
from hole_classifier4 import base
import radar_check as rc

LAB_DIR = os.path.join(HERE, 'labels', 'holes_pseudo')
OUT_DIR = None   # args 파싱 뒤 설정
IN_W, IN_H, STRIDE, CH = base.IN_W, base.IN_H, base.STRIDE, hc4.CH
ap = argparse.ArgumentParser()
ap.add_argument('--epochs', type=int, default=10)
ap.add_argument('--bs', type=int, default=4)
ap.add_argument('--lr', type=float, default=5e-5)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--sigma', type=float, default=2.0)
ap.add_argument('--eval_only', action='store_true')
ap.add_argument('--max_per_session', type=int, default=20)
ap.add_argument('--workers', type=int, default=4)
ap.add_argument('--exclude-unsplit', dest='exclude_unsplit', action='store_true', default=True,
                help='(기본) DB split 미지정 세션을 학습·검증에서 제외 — CNN(datasets_factory_v2 뷰)과 같은 기준, 19_blind_eval.py 블라인드 집합으로 남김')
ap.add_argument('--include-unsplit', dest='exclude_unsplit', action='store_false',
                help='미지정 세션도 학습에 포함 (2026-09-23 첫 run 은 이 상태였음 — 블라인드 평가에 부적합)')
ap.add_argument('--out', default='hole_landmarks_bracket', help='attribute_models/<out>/ 산출 폴더')
args = ap.parse_args()
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
OUT_DIR = os.path.join(HERE, 'attribute_models', args.out); os.makedirs(OUT_DIR, exist_ok=True)
hc4.MODEL_PATH = os.path.join(OUT_DIR, 'model.pth')


def load_items():
    hold = set(json.load(open(os.path.join(DOOR, 'attribute_models', 'hole_landmarks', 'split.json')))['holdout'])
    items = []
    for f in sorted(glob.glob(os.path.join(LAB_DIR, '*.json'))):
        d = json.load(open(f)); P = d['points']
        pts = {'bolt': [P[k] for k in ('bolt_1', 'bolt_2', 'bolt_3', 'bolt_4') if P.get(k)],
               'corner_hinge': [P['corner_hinge']], 'corner_latch': [P['corner_latch']],
               'bracket': [P[k] for k in ('bracket_1', 'bracket_2') if P.get(k)]}
        img = os.path.normpath(os.path.join(HERE, d['image']))
        role = 'holdout' if (d['src'] == 'human+rule' and d.get('key') in hold) else ('test' if d['split'] == 'test' else 'val' if d['split'] == 'val' else 'train')
        if args.exclude_unsplit and d['src'] == 'pseudo' and d['split'] is None:
            role = 'blind'                                   # 학습·검증 모두 제외
        items.append(dict(key=os.path.basename(f)[:-5], image=img, cls=d['cls'], src=d['src'], pts=pts, mask=[1.0, 1.0, 1.0, 1.0 if d['bracket_known'] else 0.0],
                          radar_gt=d.get('radar_gt'), role=role, session=d.get('session_dir')))
    # 세션당 프레임 수 제한(중복 완화)
    per = collections.Counter(); out = []
    for it in items:
        if it['session']:
            if per[it['session']] >= args.max_per_session: continue
            per[it['session']] += 1
        out.append(it)
    return out


def letterbox_M(h, w):
    s = min(IN_W / w, IN_H / h)
    return np.array([[s, 0, (IN_W - w * s) / 2], [0, s, (IN_H - h * s) / 2]], np.float32)


def rand_aug_M(h, w):
    ang = random.uniform(-180, 180); sc = random.uniform(0.75, 1.15)
    R = cv2.getRotationMatrix2D((IN_W / 2, IN_H / 2), ang, sc)
    R[0, 2] += random.uniform(-60, 60); R[1, 2] += random.uniform(-40, 40)
    M = R @ np.vstack([letterbox_M(h, w), [0, 0, 1]])
    if random.random() < 0.5:
        Fm = np.array([[-1, 0, IN_W - 1], [0, 1, 0], [0, 0, 1]], np.float32)
        M = (Fm @ np.vstack([M, [0, 0, 1]]))[:2]
    return M.astype(np.float32)


def apply_M(M, p):
    return np.array([M[0, 0] * p[0] + M[0, 1] * p[1] + M[0, 2], M[1, 0] * p[0] + M[1, 1] * p[1] + M[1, 2]])


def heatmaps(pts_in, sigma):
    Hh, Ww = IN_H // STRIDE, IN_W // STRIDE
    hm = np.zeros((len(CH), Hh, Ww), np.float32); ys, xs = np.mgrid[0:Hh, 0:Ww]
    for ci, c in enumerate(CH):
        for p in pts_in[c]:
            x, y = p[0] / STRIDE, p[1] / STRIDE
            if x < -3 or y < -3 or x > Ww + 3 or y > Hh + 3: continue
            hm[ci] = np.maximum(hm[ci], np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma ** 2)))
    return hm


class DS(Dataset):
    def __init__(self, items, train): self.items, self.train = items, train
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        it = self.items[i]; im = cv2.imread(it['image']); h, w = im.shape[:2]
        M = rand_aug_M(h, w) if self.train else letterbox_M(h, w)
        x = cv2.warpAffine(im, M, (IN_W, IN_H), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
        if self.train:
            a = random.uniform(0.7, 1.3); b = random.uniform(-30, 30)
            x = np.clip(x.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
            if random.random() < 0.3: x = cv2.GaussianBlur(x, (3, 3), 0)
        hm = heatmaps({c: [apply_M(M, p) for p in it['pts'][c]] for c in CH}, args.sigma)
        x = torch.from_numpy(x[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.
        x = (x - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        return x, torch.from_numpy(hm), torch.tensor(it['mask'], dtype=torch.float32)


def masked_loss(pred, gt, mask):
    w = 1 + 20 * gt
    per = ((pred - gt) ** 2 * w).mean(dim=(2, 3))            # (B, C)
    return (per * mask).sum() / mask.sum().clamp(min=1.0)


@torch.no_grad()
def evaluate(net, dev, items, tag):
    """사람 홀드아웃: 6점 ≤8px 비율. 의사 라벨(val/test): 브래킷 판정(radar/none) 정확도 + 6점 오차."""
    net.eval(); errs = collections.defaultdict(list); rd = collections.Counter(); geom = hc4.geometry()
    for it in items:
        rgb = cv2.imread(it['image']); det = hc4.detect(net, dev, rgb)
        for c in ('bolt', 'corner_hinge', 'corner_latch', 'bracket'):
            for g in it['pts'][c]:
                if det[c]: errs[c].append(min(math.hypot(d[0] - g[0], d[1] - g[1]) for d in det[c]))
        if it['radar_gt'] in (0, 1) and it['mask'][3] > 0 and hc4.GROUP.get(it['cls']) != 'FRT':
            st, _ = hc4.radar_from_bracket(det, it['cls'], geom)
            gt = 'radar' if it['radar_gt'] == 1 else 'none'
            rd['judged' if st in ('radar', 'none') else 'unsure'] += 1
            if st in ('radar', 'none'): rd['correct' if st == gt else 'wrong'] += 1
            rd['n'] += 1
    six = np.array([e for c in ('bolt', 'corner_hinge', 'corner_latch') for e in errs[c]])
    br = np.array(errs['bracket'])
    r = dict(tag=tag, n=len(items), six_le8=float(np.mean(six <= 8)) if six.size else None, six_med=float(np.median(six)) if six.size else None,
             bracket_med=float(np.median(br)) if br.size else None, bracket_le8=float(np.mean(br <= 8)) if br.size else None,
             radar_n=rd['n'], radar_judged=rd['judged'], radar_correct=rd['correct'], radar_wrong=rd['wrong'],
             radar_acc=(rd['correct'] / rd['judged']) if rd['judged'] else None)
    print(f"  [{tag}] n={r['n']} 6점 ≤8px {100 * (r['six_le8'] or 0):.1f}% (med {r['six_med'] or 0:.1f}px) | 브래킷 med {r['bracket_med'] if r['bracket_med'] is not None else float('nan'):.1f}px "
          f"| 레이더 판정 {rd['judged']}/{rd['n']} 정확도 {100 * (r['radar_acc'] or 0):.1f}% (오판 {rd['wrong']})", flush=True)
    return r


if __name__ == '__main__':
    items = load_items()
    train = [i for i in items if i['role'] == 'train']; val = [i for i in items if i['role'] == 'val']
    hold = [i for i in items if i['role'] == 'holdout']; test = [i for i in items if i['role'] == 'test']
    print(f"라벨 {len(items)} → 학습 {len(train)} (사람 {sum(i['src'] != 'pseudo' for i in train)}) / val {len(val)} / 사람 홀드아웃 {len(hold)} / test(제외) {len(test)} / 블라인드(제외) {sum(i['role'] == 'blind' for i in items)}; "
          f"브래킷 양성 {sum(bool(i['pts']['bracket']) for i in train)} 음성 {sum(not i['pts']['bracket'] and i['mask'][3] > 0 for i in train)} 마스크 {sum(i['mask'][3] == 0 for i in train)}")
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = hc4.Net().to(dev); mp = hc4.MODEL_PATH
    if not args.eval_only:
        hc4.init_from_base(net)
        from db_log import DBLog
        db = DBLog(); mid = db.register_model(name=f'{args.out}_resnet18', architecture='ResNet18-FPN-heatmap', in_channels=3, num_classes=len(CH),
                                              weights_path=os.path.relpath(mp, DOOR), input_size=f'{IN_W}x{IN_H}',
                                              description='door_partno/16_train_hole_landmarks.py (bolt/corner_hinge/corner_latch/bracket), 정본 3채널에서 미세조정, 의사 라벨')
        sess = db.start_training(dataset_name='hole_labels', model_id=mid, optimizer='AdamW', learning_rate=args.lr, batch_size=args.bs, max_epochs=args.epochs,
                                 early_stop_patience=None, train_ratio=0.85, train_count=len(train), test_count=len(hold) + len(val), gpu_device=str(dev), loss_function='masked weighted MSE')
        dl = DataLoader(DS(train, True), batch_size=args.bs, shuffle=True, num_workers=args.workers, drop_last=True, persistent_workers=True)
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
        warm = len(dl); total = args.epochs * len(dl)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, i / total))))
        t0 = time.time(); best = None; hist = []
        for ep in range(args.epochs):
            net.train(); tl = 0
            for x, y, m in dl:
                x, y, m = x.to(dev), y.to(dev), m.to(dev)
                loss = masked_loss(net(x), y, m)
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); tl += loss.item()
            tl /= len(dl)
            print(f'ep{ep + 1} loss={tl:.5f} [{time.time() - t0:.0f}s]', flush=True)
            rh = evaluate(net, dev, hold, 'holdout'); rv = evaluate(net, dev, val, 'val')
            score = ((rh['six_le8'] or 0) + (rv['radar_acc'] or 0)) / 2
            hist.append(dict(epoch=ep + 1, loss=tl, holdout=rh, val=rv, score=score))
            db.log_epoch(sess, ep + 1, tl, None, round(score * 100, 2), sched.get_last_lr()[0], round(time.time() - t0, 1))
            if best is None or score >= best:
                best = score; torch.save(net.state_dict(), mp); print(f'  → 저장 (score {score:.4f})', flush=True)
        db.finish_training(sess, status='completed', actual_epochs=args.epochs, best_val_accuracy=round((best or 0) * 100, 2), total_time_sec=round(time.time() - t0, 1)); db.close()
        json.dump(hist, open(os.path.join(OUT_DIR, 'train_hist.json'), 'w'), indent=1, default=float)
    net.load_state_dict(torch.load(mp, map_location=dev))
    res = dict(holdout=evaluate(net, dev, hold, 'holdout'), val=evaluate(net, dev, val, 'val'), test=evaluate(net, dev, test, 'test'))
    json.dump(res, open(os.path.join(OUT_DIR, 'eval.json'), 'w'), indent=1, default=float)
    print('→', os.path.join(OUT_DIR, 'eval.json'))
