"""홀 랜드마크 히트맵 검출기 학습/평가 (labels/holes/*.json).

채널 3개: bolt(래치 볼트홀 4개, 순서 무관) / corner_hinge / corner_latch
입력 letterbox 1280×768, 출력 stride 4 히트맵. ResNet18(ImageNet) 인코더 + 경량 디코더.
증강: 임의 회전·스케일·플립·밝기 (도어가 바닥에 아무 각도로 놓이는 상황).

실행:
  python 15_train_hole_landmarks.py                 # 학습 + 홀드아웃 평가
  python 15_train_hole_landmarks.py --eval_only     # 저장 모델로 평가만
산출: attribute_models/hole_landmarks/{model.pth, split.json, eval.json, eval_*.jpg}
"""
import argparse, glob, json, math, os, random, sys, time
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models

DOOR = os.path.dirname(os.path.abspath(__file__))
LAB_DIR = os.path.join(DOOR, 'labels', 'holes')
OUT_DIR = os.path.join(DOOR, 'attribute_models', 'hole_landmarks')
IN_W, IN_H, STRIDE = 1280, 768, 4
CH = ['bolt', 'corner_hinge', 'corner_latch']
BOLT_KEYS = ['bolt_tl', 'bolt_tr', 'bolt_bl', 'bolt_br']
K_DEPTH = 0.8235   # depth(근사 intrinsics) mm → 실제 mm 보정 (볼트 피치로 1회 캘리브레이션)
CAD_D = {'E25_door_LH_FRT': 724, 'E30_door_LH_FRT': 765, 'E38_door_LH_FRT': 812, 'E25_door_LH_RR': 1037,
         'E30_door_LH_RR': 1158, 'E38_door_LH_RR': 1352, 'E25_door_RH': 886, 'E30_E38_door_RH': 1087}

ap = argparse.ArgumentParser()
ap.add_argument('--epochs', type=int, default=80)
ap.add_argument('--bs', type=int, default=4)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--holdout', type=float, default=0.2)
ap.add_argument('--eval_only', action='store_true')
ap.add_argument('--sigma', type=float, default=2.0)
args = ap.parse_args()
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
os.makedirs(OUT_DIR, exist_ok=True)


# ── 데이터 ────────────────────────────────────────────────
def load_labels():
    items = []
    for f in sorted(glob.glob(f'{LAB_DIR}/*.json')):
        d = json.load(open(f)); P = d['points']
        pts = {'bolt': [P[k] for k in BOLT_KEYS if P.get(k)],
               'corner_hinge': [P['corner_hinge']] if P.get('corner_hinge') else [],
               'corner_latch': [P['corner_latch']] if P.get('corner_latch') else []}
        if sum(len(v) for v in pts.values()) == 0:
            continue
        items.append(dict(key=os.path.basename(f)[:-5], image=os.path.join(DOOR, d['image']), cls=d['cls'], src=d['src'], pts=pts))
    return items


def split_items(items):
    sp = os.path.join(OUT_DIR, 'split.json')
    if os.path.exists(sp):
        s = json.load(open(sp)); hold = set(s['holdout'])
    else:
        by = {}
        for it in items:
            by.setdefault(it['cls'], []).append(it['key'])
        hold = set()
        rng = random.Random(args.seed)
        for c, ks in by.items():
            ks = sorted(ks); rng.shuffle(ks); n = max(1, int(round(len(ks) * args.holdout)))
            hold.update(ks[:n])
        json.dump(dict(holdout=sorted(hold)), open(sp, 'w'), indent=1)
    return [i for i in items if i['key'] not in hold], [i for i in items if i['key'] in hold]


def letterbox_M(h, w):
    s = min(IN_W / w, IN_H / h)
    M = np.array([[s, 0, (IN_W - w * s) / 2], [0, s, (IN_H - h * s) / 2]], np.float32)
    return M, s


def rand_aug_M(h, w):
    """원본 → 입력 캔버스 임의 아핀 (회전·스케일·플립·이동)."""
    ang = random.uniform(-180, 180); sc = random.uniform(0.75, 1.15)
    base, s0 = letterbox_M(h, w)
    R = cv2.getRotationMatrix2D((IN_W / 2, IN_H / 2), ang, sc)
    R[0, 2] += random.uniform(-60, 60); R[1, 2] += random.uniform(-40, 40)
    M = R @ np.vstack([base, [0, 0, 1]])
    if random.random() < 0.5:  # 좌우 플립
        Fm = np.array([[-1, 0, IN_W - 1], [0, 1, 0], [0, 0, 1]], np.float32)
        M = (Fm @ np.vstack([M, [0, 0, 1]]))[:2]
    return M.astype(np.float32)


def apply_M(M, p):
    return np.array([M[0, 0] * p[0] + M[0, 1] * p[1] + M[0, 2], M[1, 0] * p[0] + M[1, 1] * p[1] + M[1, 2]])


def heatmaps(pts_in, sigma):
    Hh, Ww = IN_H // STRIDE, IN_W // STRIDE
    hm = np.zeros((len(CH), Hh, Ww), np.float32)
    ys, xs = np.mgrid[0:Hh, 0:Ww]
    for ci, c in enumerate(CH):
        for p in pts_in[c]:
            x, y = p[0] / STRIDE, p[1] / STRIDE
            if x < -3 or y < -3 or x > Ww + 3 or y > Hh + 3:
                continue
            g = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma ** 2))
            hm[ci] = np.maximum(hm[ci], g)
    return hm


class HoleDS(Dataset):
    def __init__(self, items, train):
        self.items, self.train = items, train
        self.cache = {}

    def __len__(self):
        return len(self.items) * (4 if self.train else 1)

    def img(self, path):
        if path not in self.cache:
            self.cache[path] = cv2.imread(path)
        return self.cache[path]

    def __getitem__(self, i):
        it = self.items[i % len(self.items)]
        im = self.img(it['image']); h, w = im.shape[:2]
        M = rand_aug_M(h, w) if self.train else letterbox_M(h, w)[0]
        x = cv2.warpAffine(im, M, (IN_W, IN_H), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
        if self.train:
            a = random.uniform(0.7, 1.3); b = random.uniform(-30, 30)
            x = np.clip(x.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
            if random.random() < 0.3:
                x = cv2.GaussianBlur(x, (3, 3), 0)
        pts_in = {c: [apply_M(M, p) for p in it['pts'][c]] for c in CH}
        hm = heatmaps(pts_in, args.sigma)
        x = torch.from_numpy(x[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.
        x = (x - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        return x, torch.from_numpy(hm)


# ── 모델 ──────────────────────────────────────────────────
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        r = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)          # /2, 64
        self.l1 = nn.Sequential(r.maxpool, r.layer1)               # /4, 64
        self.l2, self.l3, self.l4 = r.layer2, r.layer3, r.layer4   # /8 128, /16 256, /32 512
        self.lat4 = nn.Conv2d(512, 128, 1); self.lat3 = nn.Conv2d(256, 128, 1)
        self.lat2 = nn.Conv2d(128, 128, 1); self.lat1 = nn.Conv2d(64, 128, 1)
        self.head = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                  nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                  nn.Conv2d(64, len(CH), 1))

    def forward(self, x):
        c1 = self.l1(self.stem(x)); c2 = self.l2(c1); c3 = self.l3(c2); c4 = self.l4(c3)
        p = self.lat4(c4)
        p = F.interpolate(p, size=c3.shape[2:], mode='bilinear', align_corners=False) + self.lat3(c3)
        p = F.interpolate(p, size=c2.shape[2:], mode='bilinear', align_corners=False) + self.lat2(c2)
        p = F.interpolate(p, size=c1.shape[2:], mode='bilinear', align_corners=False) + self.lat1(c1)
        return self.head(p)


def focal_mse(pred, gt):
    """양성 픽셀 가중 MSE (히트맵이 거의 0이라 단순 MSE는 붕괴)."""
    w = 1 + 20 * gt
    return ((pred - gt) ** 2 * w).mean()


# ── 피크 추출 / 평가 ────────────────────────────────────────
def peaks(hm, k, thr=0.15, nms=5):
    """hm: (H,W) → 최대 k개 (x,y,score) 서브픽셀."""
    h = torch.from_numpy(hm)[None, None]
    mx = F.max_pool2d(h, nms * 2 + 1, stride=1, padding=nms)
    cand = ((h == mx) & (h > thr))[0, 0].nonzero().numpy()
    out = []
    for y, x in cand:
        sc = hm[y, x]
        dx = dy = 0.0
        if 0 < x < hm.shape[1] - 1: dx = 0.5 * (hm[y, x + 1] - hm[y, x - 1]) / max(1e-6, (2 * hm[y, x] - hm[y, x + 1] - hm[y, x - 1]))
        if 0 < y < hm.shape[0] - 1: dy = 0.5 * (hm[y + 1, x] - hm[y - 1, x]) / max(1e-6, (2 * hm[y, x] - hm[y + 1, x] - hm[y - 1, x]))
        out.append((float(x + np.clip(dx, -1, 1)), float(y + np.clip(dy, -1, 1)), float(sc)))
    out.sort(key=lambda t: -t[2])
    return out[:k]


def depth_mm_distance(depth, pts_all, pa, pb, h):
    """검출점들의 볼록껍질 내부 depth로 도어 평면 피팅 → 두 점의 평면상 mm 거리 (K_DEPTH 보정)."""
    K = dict(fx=1065.0, fy=1065.0, cx=960.0, cy=h / 2.0)
    P = np.array([(p[0], p[1]) for p in pts_all], np.float32)
    if len(P) < 3: return None
    hull = cv2.convexHull(P).astype(np.int32)
    m = np.zeros(depth.shape, np.uint8); cv2.fillConvexPoly(m, hull, 255)
    m = cv2.erode(m, np.ones((15, 15), np.uint8))
    rows, cols = np.where((m > 0) & (depth > 0))
    if len(rows) < 500: return None
    if len(rows) > 40000:
        sel = np.random.default_rng(0).choice(len(rows), 40000, replace=False); rows, cols = rows[sel], cols[sel]
    z = depth[rows, cols].astype(np.float64)
    X = (cols - K['cx']) * z / K['fx']; Y = (rows - K['cy']) * z / K['fy']
    Q = np.stack([X, Y, z], 1); c = Q.mean(0); _, _, Vt = np.linalg.svd(Q - c, full_matrices=False); n = Vt[2]
    d = np.abs((Q - c) @ n); keep = d < 10
    if keep.sum() > 200:
        c = Q[keep].mean(0); _, _, Vt = np.linalg.svd(Q[keep] - c, full_matrices=False); n = Vt[2]
    def to3d(p):
        # 픽셀 → 광선 → 평면 교점
        r = np.array([(p[0] - K['cx']) / K['fx'], (p[1] - K['cy']) / K['fy'], 1.0])
        t = np.dot(c, n) / np.dot(r, n)
        return r * t
    return float(np.linalg.norm(to3d(pa) - to3d(pb))) * K_DEPTH


def bolt_frame(bolts):
    """볼트 4점 → (center, ex(157mm축 단위벡터), ey(96mm축), s px/mm). 형상 불일치면 None."""
    if len(bolts) < 4: return None
    P = np.array([(b[0], b[1]) for b in bolts[:4]], float)
    c = P.mean(0); Q = P - c
    # 최장 변 방향 = 157축
    pairs = sorted(((np.linalg.norm(P[a] - P[b]), a, b) for a in range(4) for b in range(a + 1, 4)), key=lambda t: t[0])
    _, i, j = pairs[2]   # [s,s,l,l,d,d] → 3번째 = 장변(157mm)
    ex = P[j] - P[i]; ex /= np.linalg.norm(ex); ey = np.array([-ex[1], ex[0]])
    u = Q @ ex; w = Q @ ey
    long_ = (u.max() - u.min()); short = (w.max() - w.min())
    s1, s2 = long_ / 157.0, short / 96.0
    if abs(s1 - s2) / max(s1, s2) > 0.18: return None
    return dict(c=c, ex=ex, ey=ey, s=(s1 + s2) / 2)


def geometry_gate(fr, hinge, latch, img_shape, margin=20):
    """모서리 홀 쌍의 기하 일관성: 래치홀이 볼트 프레임 기준 (|u|≈160, |w|≈190)mm, 힌지홀이 같은 w측·같은 선상, D 범위."""
    if fr is None or hinge is None or latch is None: return 'no_frame'
    h, w = img_shape[:2]
    for p in (hinge, latch):
        if p[0] < margin or p[1] < margin or p[0] > w - margin or p[1] > h - margin: return 'near_border'
    s = fr['s']
    vl = (np.array(latch[:2]) - fr['c']); ul, wl = vl @ fr['ex'] / s, vl @ fr['ey'] / s
    vh = (np.array(hinge[:2]) - fr['c']); uh, wh = vh @ fr['ex'] / s, vh @ fr['ey'] / s
    if not (100 <= abs(ul) <= 230 and 120 <= abs(wl) <= 270): return 'latch_offset'
    if abs(wh - wl) > 70: return 'not_collinear'          # 같은 상단 선 위 (하단 홀이면 w 부호 반대·거리 큼)
    if np.sign(uh) == np.sign(ul): return 'same_side'
    Dmm = abs(uh - ul)
    if not (600 <= Dmm <= 1500): return 'D_range'
    return 'ok'


def bolt_scale_from(bolts):
    """볼트 4점 → 157×96 직사각형 피팅 → px/mm (형상 불일치면 None)."""
    if len(bolts) < 4: return None
    P = np.array([(b[0], b[1]) for b in bolts[:4]])
    D = np.linalg.norm(P[:, None] - P[None], axis=2)
    d = sorted(D[np.triu_indices(4, 1)])
    short, long_, diag = np.mean(d[:2]), np.mean(d[2:4]), np.mean(d[4:])
    s1, s2 = long_ / 157.0, short / 96.0
    if abs(s1 - s2) / max(s1, s2) > 0.08 or abs(diag - math.hypot(157, 96) * (s1 + s2) / 2) / diag > 0.08:
        return None
    return (s1 + s2) / 2


@torch.no_grad()
def evaluate(net, items, dev, tag='eval', draw=True):
    net.eval(); rows = []
    for it in items:
        im = cv2.imread(it['image']); h, w = im.shape[:2]
        M, s0 = letterbox_M(h, w)
        x = cv2.warpAffine(im, M, (IN_W, IN_H), borderValue=(114, 114, 114))
        t = torch.from_numpy(x[:, :, ::-1].copy()).permute(2, 0, 1).float()[None] / 255.
        t = (t - torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]) / torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
        hm = net(t.to(dev))[0].float().cpu().numpy()
        Mi = cv2.invertAffineTransform(M)
        det = {}
        for ci, c in enumerate(CH):
            pk = peaks(hm[ci], 4 if c == 'bolt' else 1)
            det[c] = [(*apply_M(Mi, (px * STRIDE, py * STRIDE)), sc) for px, py, sc in pk]   # 원본 좌표
        r = dict(key=it['key'], cls=it['cls'], src=it['src'])
        gt = it['pts']
        # 점 오차 (원본 px)
        for c in CH:
            errs = []
            for g in gt[c]:
                if det[c]:
                    errs.append(float(min(math.hypot(d[0] - g[0], d[1] - g[1]) for d in det[c])))
            r[f'{c}_err_px'] = errs
            r[f'{c}_n_det'] = len(det[c]); r[f'{c}_n_gt'] = len(gt[c])
            r[f'{c}_score'] = [round(d[2], 3) for d in det[c]]
        # D 판정: 1순위 depth 평면 거리, 2순위 볼트 피치 스케일
        if det['corner_hinge'] and det['corner_latch']:
            dpath = it['image'].replace('rgb_', 'depth_')
            depth = cv2.imread(dpath, cv2.IMREAD_UNCHANGED) if os.path.exists(dpath) else None
            Dd = depth_mm_distance(depth, det['bolt'] + det['corner_hinge'] + det['corner_latch'], det['corner_hinge'][0], det['corner_latch'][0], h) if depth is not None else None
            s = bolt_scale_from(det['bolt'])
            Dpx = math.hypot(det['corner_hinge'][0][0] - det['corner_latch'][0][0], det['corner_hinge'][0][1] - det['corner_latch'][0][1])
            if Dd: r['D_mm'] = Dd; r['D_src'] = 'depth'
            elif s: r['D_mm'] = Dpx / s; r['D_src'] = 'bolt'
            if s: r['D_bolt_mm'] = Dpx / s
            fr = bolt_frame(det['bolt'])
            r['gate'] = geometry_gate(fr, det['corner_hinge'][0], det['corner_latch'][0], im.shape)
            if 'D_mm' in r and r['gate'] == 'ok': r['pred'] = min(CAD_D, key=lambda k: abs(CAD_D[k] - r['D_mm']))
        else:
            r['gate'] = 'no_corner'
        # GT D (GT 볼트로 스케일)
        sg = bolt_scale_from([(p[0], p[1], 1) for p in gt['bolt']])
        if sg and gt['corner_hinge'] and gt['corner_latch']:
            r['D_gt_mm'] = math.hypot(gt['corner_hinge'][0][0] - gt['corner_latch'][0][0], gt['corner_hinge'][0][1] - gt['corner_latch'][0][1]) / sg
        rows.append(r)
        if draw:
            v = im.copy()
            for c, col in zip(CH, [(255, 0, 0), (0, 0, 255), (0, 140, 255)]):
                for g in gt[c]: cv2.circle(v, (int(g[0]), int(g[1])), 12, (0, 255, 0), 1)
                for d in det[c]:
                    cv2.circle(v, (int(d[0]), int(d[1])), 7, col, 2); cv2.putText(v, f'{d[2]:.2f}', (int(d[0]) + 8, int(d[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
            txt = f"{it['cls']} D={r.get('D_mm', 0):.0f} (gt {r.get('D_gt_mm', 0):.0f}) -> {r.get('pred', '-')}"
            cv2.putText(v, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(OUT_DIR, f'{tag}_{it["key"]}.jpg'), v, [cv2.IMWRITE_JPEG_QUALITY, 80])
    # 집계
    def stat(name):
        e = np.array([x for r in rows for x in r[f'{name}_err_px']])
        return f"{name:13s} n={len(e):3d} med={np.median(e) if e.size else 0:5.1f}px  ≤4px {np.mean(e <= 4) * 100 if e.size else 0:4.0f}%  ≤8px {np.mean(e <= 8) * 100 if e.size else 0:4.0f}%  >20px {np.mean(e > 20) * 100 if e.size else 0:4.0f}%"
    lines = [f"[{tag}] {len(rows)}장"] + [stat(c) for c in CH]
    valid = [r for r in rows if r['cls'] in CAD_D]
    judged = [r for r in valid if 'pred' in r]
    acc = np.mean([r['pred'] == r['cls'] for r in judged]) * 100 if judged else 0
    import collections
    gates = collections.Counter(r.get('gate', '-') for r in valid)
    lines.append(f"D 판정: 판정 {len(judged)}/{len(valid)} (보류 사유 {dict(gates)})  판정 중 정확도 {acc:.1f}%  전체 대비 정답 {sum(r['pred'] == r['cls'] for r in judged)}/{len(valid)}")
    de = np.array([abs(r['D_mm'] - r['D_gt_mm']) for r in rows if 'D_mm' in r and 'D_gt_mm' in r])
    if de.size: lines.append(f"D 오차 |det-gt| med={np.median(de):.1f}mm  ≤10mm {np.mean(de <= 10) * 100:.0f}%")
    print('\n'.join(lines), flush=True)
    return rows, lines


# ── 메인 ──────────────────────────────────────────────────
if __name__ == '__main__':
    items = load_labels(); train_items, hold_items = split_items(items)
    print(f'라벨 {len(items)}장 → 학습 {len(train_items)} / 홀드아웃 {len(hold_items)}')
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Net().to(dev)
    mp = os.path.join(OUT_DIR, 'model.pth')
    if not args.eval_only:
        from db.db_log import DBLog
        db = DBLog(); mid = db.register_model(name='hole_landmarks_resnet18', architecture='ResNet18-FPN-heatmap', in_channels=3,
                                              num_classes=len(CH), weights_path=os.path.relpath(mp, DOOR), input_size=f'{IN_W}x{IN_H}',
                                              description='15_train_hole_landmarks.py (bolt/corner_hinge/corner_latch)')
        sess = db.start_training(dataset_name='hole_labels', model_id=mid, optimizer='AdamW', learning_rate=args.lr, batch_size=args.bs,
                                 max_epochs=args.epochs, early_stop_patience=None, train_ratio=1 - args.holdout,
                                 train_count=len(train_items), test_count=len(hold_items), gpu_device=str(dev), loss_function='weighted MSE')
        dl = DataLoader(HoleDS(train_items, True), batch_size=args.bs, shuffle=True, num_workers=4, drop_last=True)
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
        warm = 3 * len(dl); total = args.epochs * len(dl)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, i / total))))
        t0 = time.time(); best = None
        for ep in range(args.epochs):
            net.train(); tl = 0
            for x, y in dl:
                x, y = x.to(dev), y.to(dev)
                loss = focal_mse(net(x), y)
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); tl += loss.item()
            tl /= len(dl)
            if (ep + 1) % 5 == 0 or ep == 0:
                rows, lines = evaluate(net, hold_items, dev, tag='ep', draw=False)
                e = np.array([x for r in rows for c in CH for x in r[f'{c}_err_px']])
                score = float(np.mean(e <= 8)) if e.size else 0
                print(f'ep{ep + 1} loss={tl:.5f} holdout ≤8px={score * 100:.1f}%  [{time.time() - t0:.0f}s]', flush=True)
                db.log_epoch(sess, ep + 1, tl, None, round(score * 100, 2), sched.get_last_lr()[0], round(time.time() - t0, 1))
                if best is None or score >= best:
                    best = score; torch.save(net.state_dict(), mp)
            else:
                db.log_epoch(sess, ep + 1, tl, None, None, sched.get_last_lr()[0], round(time.time() - t0, 1))
        db.finish_training(sess, status='completed', actual_epochs=args.epochs, best_val_accuracy=round((best or 0) * 100, 2),
                           total_time_sec=round(time.time() - t0, 1)); db.close()
    net.load_state_dict(torch.load(mp, map_location=dev))
    rows, lines = evaluate(net, hold_items, dev, tag='eval', draw=True)
    rows_tr, lines_tr = evaluate(net, train_items, dev, tag='train', draw=False)
    json.dump(dict(holdout=rows, train=rows_tr, summary=lines, summary_train=lines_tr), open(os.path.join(OUT_DIR, 'eval.json'), 'w'), indent=1, default=float)
    print('\n'.join(lines_tr))
