"""CNN 품번 학습 — 형상군 9클래스 + 레이더 이진 헤드(다중과제, 기본) 또는 품번 14클래스 단일 헤드.

  python 02_train_multitask.py --mode multi --image_size 448 --seed 42            # 권장 주안
  python 02_train_multitask.py --mode part14 --image_size 448 --seed 42           # 비교 기준선
  python 02_train_multitask.py --mode multi --image_size 640 --seed 42            # 해상도 변형

입력: door_pipeline 과 동일한 RGBE(RGB + Canny) 전체 프레임 레터박스, NoAuxResNet18 백본. 초기화 기본 seed916 정식 run(백본+9클래스 헤드), --init imagenet 가능.
데이터: labels/partno_manifest.json (partno_labels.py) — train/val/test 는 DB 세션 단위 split 그대로.
손실: multi = CE(클래스, sqrt 역빈도 가중) + w·BCE(레이더, GT 있는 RR/RH 프레임만) / part14 = CE(품번 14, sqrt 가중; 품번 미정 프레임 제외)
선택: val 복합 점수(클래스 정확도와 레이더 정확도 평균 / part14 는 품번 정확도) 최대 에폭(동점이면 val 손실 최소), 조기 종료 patience.
산출: artifacts/<run>/{model.pth, split_info.json, train_log.json} + DB(models/training_sessions).
"""
import argparse, json, os, random, sys, time
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DOOR = os.path.normpath(os.path.join(HERE, '..', 'door_pipeline'))
sys.path.insert(0, DOOR); sys.path.insert(0, os.path.join(DOOR, 'db')); sys.path.insert(0, HERE)
from rgbe_utils import RGBETransform, RGBE_IN_CHANNELS
from cnn_classifier import NoAuxResNet18, DEFAULT_RUN
import partno_labels as pl

ap = argparse.ArgumentParser()
ap.add_argument('--mode', choices=['multi', 'part14'], default='multi')
ap.add_argument('--image_size', type=int, default=448)
ap.add_argument('--epochs', type=int, default=40); ap.add_argument('--patience', type=int, default=8)
ap.add_argument('--seed', type=int, default=42); ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--radar_weight', type=float, default=1.0)
ap.add_argument('--init', choices=['seed916', 'imagenet'], default='seed916')
ap.add_argument('--bs', type=int, default=0, help='0=자동(448→32, 640→16)')
ap.add_argument('--workers', type=int, default=4)
ap.add_argument('--tag', default='')
args = ap.parse_args()
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
RUN = f"partno_{args.mode}_{args.image_size}_seed{args.seed}{'_' + args.init if args.init != 'seed916' else ''}{args.tag}"
RUN_DIR = os.path.join(HERE, 'artifacts', RUN); os.makedirs(RUN_DIR, exist_ok=True)


class MTNet(nn.Module):
    """NoAuxResNet18 백본 + 클래스 헤드(9) [+ 레이더 헤드(1)] 또는 품번 헤드(14)."""
    def __init__(self, mode, n_cls=9, n_part=14):
        super().__init__()
        self.mode = mode
        self.base = NoAuxResNet18(n_cls if mode == 'multi' else n_part, in_channels=RGBE_IN_CHANNELS, pretrained=(args.init == 'imagenet'))
        self.radar = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.base.backbone_features, 64), nn.ReLU(), nn.Linear(64, 1)) if mode == 'multi' else None

    def forward(self, x):
        f = self.base.backbone(x)
        return self.base.classifier(f), (self.radar(f).squeeze(1) if self.radar is not None else None)


class DS(Dataset):
    def __init__(self, rows, train):
        self.rows, self.tf = rows, RGBETransform(args.image_size, is_train=train)
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]; im = Image.open(os.path.join(HERE, r['path'])).convert('RGB')
        return self.tf(im), r['cls_idx'], r['radar'], r['part_idx']


def sqrt_weights(labels, n):
    cnt = np.bincount([l for l in labels if l >= 0], minlength=n).astype(float); tot = cnt.sum()
    inv = np.where(cnt > 0, tot / (n * np.maximum(cnt, 1)), 0.0)
    return torch.tensor(np.sqrt(inv), dtype=torch.float32)


def run_epoch(net, dl, dev, opt=None, w_cls=None, w_part=None):
    train = opt is not None; net.train(train)
    tot = dict(loss=0.0, n=0, c_ok=0, c_n=0, r_ok=0, r_n=0, p_ok=0, p_n=0)
    ce_c = nn.CrossEntropyLoss(weight=w_cls); ce_p = nn.CrossEntropyLoss(weight=w_part); bce = nn.BCEWithLogitsLoss()
    with torch.set_grad_enabled(train):
        for x, c, r, p in dl:
            x, c, r, p = x.to(dev, non_blocking=True), c.to(dev), r.to(dev), p.to(dev)
            logits, rl = net(x)
            if args.mode == 'multi':
                loss = ce_c(logits, c); m = r >= 0
                if m.any(): loss = loss + args.radar_weight * bce(rl[m], r[m].float())
                pred = logits.argmax(1); tot['c_ok'] += (pred == c).sum().item(); tot['c_n'] += len(c)
                if m.any():
                    rp = (rl[m] > 0).long(); tot['r_ok'] += (rp == r[m]).sum().item(); tot['r_n'] += int(m.sum())
                    # 품번 = 클래스 예측 × 레이더 예측 (레이더 GT 있는 프레임) / FRT 는 클래스만
                pv = torch.tensor([pl.part_index(pl.CLASSES[int(pc)], (int(rl_i > 0) if pl.GROUP[pl.CLASSES[int(pc)]] != 'FRT' else None))
                                   for pc, rl_i in zip(pred, rl)], device=dev)
                mp = p >= 0; tot['p_ok'] += (pv[mp] == p[mp]).sum().item(); tot['p_n'] += int(mp.sum())
            else:
                m = p >= 0
                loss = ce_p(logits[m], p[m])
                pred = logits.argmax(1); tot['p_ok'] += (pred[m] == p[m]).sum().item(); tot['p_n'] += int(m.sum())
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            tot['loss'] += loss.item() * len(c); tot['n'] += len(c)
    acc = lambda a, b: 100.0 * tot[a] / max(1, tot[b])
    return dict(loss=tot['loss'] / max(1, tot['n']), cls_acc=acc('c_ok', 'c_n'), radar_acc=acc('r_ok', 'r_n'), part_acc=acc('p_ok', 'p_n'),
                n=tot['n'], radar_n=tot['r_n'], part_n=tot['p_n'])


if __name__ == '__main__':
    man = pl.load(); rows = man['rows']
    tr = [r for r in rows if r['split'] == 'train']; va = [r for r in rows if r['split'] == 'val']; te = [r for r in rows if r['split'] == 'test']
    if args.mode == 'part14':
        tr = [r for r in tr if r['part_idx'] >= 0]; va = [r for r in va if r['part_idx'] >= 0]
    print(f"[{RUN}] train {len(tr)} / val {len(va)} / test {len(te)}  (레이더 GT 있는 train {sum(r['radar'] >= 0 for r in tr)})")
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = MTNet(args.mode, len(pl.CLASSES), len(pl.PARTS)).to(dev)
    if args.init == 'seed916':
        sd = torch.load(os.path.join(DEFAULT_RUN, 'model.pth'), map_location=dev); own = net.base.state_dict()
        keep = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        own.update(keep); net.base.load_state_dict(own); print(f"  seed916 초기화: {len(keep)}/{len(own)} 텐서 복사 (헤드 {'포함' if args.mode == 'multi' else '제외'})")
    bs = args.bs or (32 if args.image_size <= 448 else 16)
    dl_tr = DataLoader(DS(tr, True), batch_size=bs, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=True)
    dl_va = DataLoader(DS(va, False), batch_size=bs, shuffle=False, num_workers=args.workers, pin_memory=True, persistent_workers=True)
    w_cls = sqrt_weights([r['cls_idx'] for r in tr], len(pl.CLASSES)).to(dev); w_part = sqrt_weights([r['part_idx'] for r in tr], len(pl.PARTS)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr); sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=3, factor=0.5)
    from db_log import DBLog
    db = DBLog(); mid = db.register_model(name=RUN, architecture='MTNet(NoAuxResNet18)', in_channels=RGBE_IN_CHANNELS,
                                          num_classes=len(pl.CLASSES) if args.mode == 'multi' else len(pl.PARTS), pretrained_base=args.init,
                                          weights_path=os.path.relpath(os.path.join(RUN_DIR, 'model.pth'), DOOR), input_size=f'{args.image_size}x{args.image_size}',
                                          description=f'door_partno/02_train_multitask.py mode={args.mode} radar_weight={args.radar_weight} seed={args.seed}')
    sess = db.start_training(dataset_name='door_factory_collect', model_id=mid, optimizer='Adam', learning_rate=args.lr, batch_size=bs, max_epochs=args.epochs,
                             early_stop_patience=args.patience, train_ratio=0.7, train_count=len(tr), test_count=len(te), gpu_device=str(dev),
                             loss_function='CE(sqrt)+BCE(radar)' if args.mode == 'multi' else 'CE14(sqrt)')
    json.dump(dict(run=RUN, mode=args.mode, image_size=args.image_size, seed=args.seed, init=args.init, class_names=pl.CLASSES, parts=pl.PARTS,
                   train_paths=[r['path'] for r in tr], val_paths=[r['path'] for r in va], test_paths=[r['path'] for r in te]),
              open(os.path.join(RUN_DIR, 'split_info.json'), 'w'), ensure_ascii=False, indent=1)
    best, best_loss, best_ep, wait, hist, t0, status = None, None, None, 0, [], time.time(), 'completed'
    try:
        for ep in range(args.epochs):
            a = run_epoch(net, dl_tr, dev, opt, w_cls, w_part); b = run_epoch(net, dl_va, dev, None, w_cls, w_part)
            sched.step(b['loss'])
            score = (b['cls_acc'] + b['radar_acc']) / 2 if args.mode == 'multi' else b['part_acc']
            hist.append(dict(epoch=ep + 1, train=a, val=b, score=score, lr=opt.param_groups[0]['lr']))
            db.log_epoch(sess, ep + 1, round(a['loss'], 6), round(b['loss'], 6), round(score, 4), opt.param_groups[0]['lr'], round(time.time() - t0, 1))
            # 선택: 복합 점수 최대, 동점이면 val 손실 최소(2026-09-23: 점수가 일찍 100% 에 닿아 덜 수렴한 2~5에폭이 저장되던 문제)
            improved = best is None or score > best or (score == best and b['loss'] < best_loss)
            if improved: best, best_loss, best_ep, wait = score, b['loss'], ep + 1, 0; torch.save(net.state_dict(), os.path.join(RUN_DIR, 'model.pth'))
            else: wait += 1
            print(f"  ep{ep + 1:3d} train L{a['loss']:.4f} cls {a['cls_acc']:.1f} radar {a['radar_acc']:.1f} part {a['part_acc']:.1f} | "
                  f"val L{b['loss']:.4f} cls {b['cls_acc']:.1f} radar {b['radar_acc']:.1f} part {b['part_acc']:.1f} | score {score:.2f} best {best:.2f}@{best_ep} {'*' if improved else ''} [{time.time() - t0:.0f}s]", flush=True)
            if wait >= args.patience: print('  조기 종료'); break
    except KeyboardInterrupt:
        status = 'stopped'; raise
    finally:
        db.finish_training(sess, status=status, actual_epochs=len(hist), best_val_accuracy=best, best_epoch=best_ep, total_time_sec=round(time.time() - t0, 1)); db.close()
        json.dump(dict(run=RUN, epochs=hist, best=best, best_epoch=best_ep, minutes=round((time.time() - t0) / 60, 1)), open(os.path.join(RUN_DIR, 'train_log.json'), 'w'), indent=1)
    print(f"완료 {RUN}: best {best:.2f} @ep{best_ep}  {(time.time() - t0) / 60:.1f}분 → {RUN_DIR}")
