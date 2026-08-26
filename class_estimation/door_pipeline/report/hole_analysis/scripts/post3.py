import json, sys, numpy as np
R=json.load(open(sys.argv[1])); CAD={k:v['D'] for k,v in json.load(open('holes/corner_small_hole_distances.json')).items()}
thr=float(sys.argv[2]) if len(sys.argv)>2 else 0.0
for tag in ('round0','round1'):
    r=R[tag]; sb=np.median([m['s_bbox'] for v in r.values() for m in v])
    print(f"\n== {tag}  (전역 스케일 s={sb:.4f}, NCC 점수 임계 {thr}, |dy|<=10s, 잘린 이미지 제외)")
    print(f"{'클래스':16s} {'n':>4s} | {'D med':>7s} {'p25':>5s} {'p75':>5s} {'std':>5s} | {'CAD':>5s} {'차이':>5s} | 정답률")
    tot=ok=0
    for c in sorted(CAD):
        v=[m for m in r[c] if m.get('D_px') and not m['cropped'] and abs(m['dy_px'])<=10*m['s_bbox'] and min(m['scL'],m['scR'])>=thr]
        d=np.array([m['D_px']/sb for m in v])
        if d.size==0: print(f"{c:16s}    0"); continue
        q=np.percentile(d,[25,50,75]); pred=[min(CAD,key=lambda k:abs(CAD[k]-x)) for x in d]; acc=np.mean([p==c for p in pred])*100
        tot+=d.size; ok+=sum(p==c for p in pred)
        print(f"{c:16s} {d.size:4d} | {q[1]:7.1f} {q[0]:5.0f} {q[2]:5.0f} {d.std():5.1f} | {CAD[c]:5d} {q[1]-CAD[c]:+5.0f} | {acc:5.1f}%")
    print(f"전체 최근접-CAD 정확도: {ok}/{tot} = {100*ok/max(1,tot):.1f}%")
