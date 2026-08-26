import json, sys, numpy as np, cv2, os
r=json.load(open(sys.argv[1])); CAD={k:v['D'] for k,v in json.load(open('holes/corner_small_hole_distances.json')).items()}
BASE='/home/koceti/parts_deep/class_estimation/door_pipeline/datasets'
def cropped(cls, idx):
    m=cv2.imread(f'{BASE}/{cls}/mask_{idx}.png',0); 
    return bool(m[0,:].any() or m[-1,:].any() or m[:,0].any() or m[:,-1].any())
valid={}; sb=[]
for c,v in r.items():
    valid[c]=[]
    for m in v:
        if m.get('D_px') is None or m.get('dy_px') is None: continue
        if abs(m['dy_px'])>10*m['s_bbox']: continue
        if cropped(c,m['idx']): continue
        valid[c].append(m)
        if m.get('s_bolt'): sb.append(m['s_bolt'])
s_warp=float(np.median(sb)) if sb else float('nan')
print(f"전역 warp 스케일 s_warp (볼트피치 기반 중앙값, n={len(sb)}): {s_warp:.4f}   [s_bbox 전체 중앙값 {np.median([m['s_bbox'] for v in r.values() for m in v if m.get('s_bbox')]):.4f}]\n")
print(f"{'클래스':16s} {'전체':>4s} {'유효':>4s} | {'D_px med':>8s} {'D_mm=px/s':>9s} {'std':>5s} | {'CAD':>5s} {'차이':>5s} | {'D_bbox med':>10s} | 이웃 CAD와 간격")
cads=sorted(CAD.values())
for c in sorted(CAD):
    v=valid[c]; d=np.array([m['D_px'] for m in v]); dmm=d/s_warp; db=np.array([m['D_bbox_mm'] for m in v])
    med=np.median(dmm) if d.size else float('nan')
    gap=min(abs(CAD[c]-x) for x in cads if x!=CAD[c])
    print(f"{c:16s} {len(r[c]):4d} {len(v):4d} | {np.median(d) if d.size else 0:8.1f} {med:9.1f} {dmm.std() if d.size else 0:5.1f} | {CAD[c]:5d} {med-CAD[c]:+5.0f} | {np.median(db) if db.size else 0:10.1f} | {gap}")
# 분류 정확도 (전역 스케일 기준 최근접 CAD)
tot=ok=0
for c in CAD:
    for m in valid[c]:
        pred=min(CAD,key=lambda k:abs(CAD[k]-m['D_px']/s_warp)); tot+=1; ok+=(pred==c)
print(f"\n유효 이미지 최근접-CAD 분류 정확도: {ok}/{tot} = {100*ok/max(1,tot):.1f}%")
