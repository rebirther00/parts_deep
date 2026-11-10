"""Frozen, label-free feature extraction. Limited extracts are always pilots."""
import shutil
import time
import cv2
import numpy as np
import torch
from .core import REPO, check_source, dump, jsonl, now, outpath, partition, read_json, read_jsonl, require_fresh_dir, sha256, snapshot
from .geometry import bolt_frame, depth_measure, geometry_gate, mask_descriptor
from .learning import HoleNet, letterbox, load_run, tensor_image


def peaks(hm,count,threshold,M,stride):
    # Local maxima, then deterministic suppression of a plateau/nearby peaks.
    a=np.asarray(hm,np.float32).copy();inv=cv2.invertAffineTransform(M);found=[]
    for _ in range(count):
        i=int(a.argmax());y,x=np.unravel_index(i,a.shape);score=float(a[y,x])
        if score<threshold:break
        p=inv@np.array([x*stride,y*stride,1.])
        found.append([float(p[0]),float(p[1]),score]);a[max(0,y-4):y+5,max(0,x-4):x+5]=-np.inf
    return found


def sync(dev):
    if dev.type=='cuda':torch.cuda.synchronize()


def extract(name,kind,feature_set,run=None,limit=0,fold='0'):
    sp,summary,rows,splits=snapshot(name);cfg=read_json(sp/'protocol.json');dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if limit:rows=partition(rows,splits,fold)['train'][:limit]
    dst=require_fresh_dir('features/'+feature_set);meta={'kind':kind,'snapshot':name,'snapshot_sha256':sha256(sp/'COMPLETE.json'),
        'pilot':bool(limit),'status':'running','created_at':now(),'code_sha256':sha256(__file__),
        'geometry_code_sha256':sha256(__import__('door8.geometry',fromlist=['x']).__file__),
        'timing_scope':'Single-image wall time: preprocessing, inference, geometry; disk decoding excluded. GPU synchronized. Warmup excluded.',
        'device':str(dev),'torch':torch.__version__,'gpu':torch.cuda.get_device_name() if dev.type=='cuda' else None}
    if kind=='holes':
        if not run:raise ValueError('Hole extraction requires a freshly trained research run')
        rp,rm=load_run(run,sp,allow_pilot=bool(limit))
        if rm['model_kind']!='holes' or rm['training_regime']!='human_manual_lab_only':raise ValueError('Only LAB-only hole models are allowed')
        net=HoleNet(pretrained=False).to(dev);net.load_state_dict(torch.load(rp/'model.pth',map_location=dev,weights_only=True));net.eval()
        meta.update(run=run,weights_sha256=rm['weights_sha256'],pilot=bool(limit or rm['pilot']))
        hp=cfg['holes']
        with torch.inference_mode():
            for _ in range(3):net(torch.zeros(1,3,hp['height'],hp['width'],device=dev))
    elif kind=='mobile_sam':
        from mobile_sam import sam_model_registry, SamPredictor
        src=REPO/'class_estimation/door/sam_models/mobile_sam.pt';dstweight=outpath('pretrained/mobile_sam.pt')
        if not dstweight.exists():dstweight.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dstweight)
        sam=sam_model_registry['vit_t'](checkpoint=str(dstweight)).to(dev).eval();predictor=SamPredictor(sam)
        meta.update(initialization='public MobileSAM checkpoint; no task-specific fine-tuning',weights_sha256=sha256(dstweight))
        dummy=np.zeros((1200,1920,3),np.uint8)
        with torch.inference_mode():
            predictor.set_image(dummy);predictor.predict(box=np.array([150,60,1770,1140]),multimask_output=True)
    else:raise ValueError(kind)
    dump(dst/'features.json',meta);out=[]
    with torch.inference_mode():
        for i,r in enumerate(rows):
            check_source(r);im=cv2.imread(r['rgb']);depth=cv2.imread(r['depth'],cv2.IMREAD_UNCHANGED)
            if im is None or depth is None:raise RuntimeError('Unreadable frozen source')
            h,w=im.shape[:2]
            if depth.shape!=(h,w):raise ValueError('RGB/depth registration shape mismatch')
            intr=r['intrinsics'];f={'frame_id':r['frame_id'],'rgb_sha256':r['rgb_sha256']}
            sync(dev);t=time.perf_counter()
            if kind=='holes':
                small,M=letterbox(im,hp['width'],hp['height']);x=tensor_image(small)[None].to(dev)
                with torch.autocast(device_type=dev.type,enabled=dev.type=='cuda'):heat=net(x)
                heat=heat[0].float().cpu().numpy()
                det={k:peaks(heat[c],n,hp['peak_threshold'],M,hp['stride']) for c,(k,n) in enumerate([('bolt',4),('hinge',1),('latch',1)])}
                hinge=det['hinge'][0] if det['hinge'] else None;latch=det['latch'][0] if det['latch'] else None
                bf=bolt_frame(det['bolt']);gate=geometry_gate(bf,hinge,latch,im.shape)
                f.update(points=det,gate=gate,span_normalized=None)
                if hinge is not None and latch is not None:
                    # Same isotropic approximation as the deployed pixel metric; per-camera scale is refit on TRAIN.
                    f['span_normalized']=float(np.linalg.norm(np.asarray(hinge[:2])-latch[:2])/intr['fx'])
                    dm=depth_measure(depth,det['bolt']+[hinge,latch],hinge,latch,intr)
                    if dm:f.update(dm)
            else:
                bbox=np.asarray(cfg['mask']['bbox_fraction'])*np.array([w,h,w,h])
                predictor.set_image(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));masks,sc,_=predictor.predict(box=bbox,multimask_output=True)
                # Selection never consults class, GT mask, CAD size, or classification correctness.
                eligible=[j for j,m in enumerate(masks) if float(m.mean())>=cfg['mask']['min_area_fraction']]
                if eligible:
                    j=max(eligible,key=lambda j:float(sc[j]));m=masks[j]
                    f.update(descriptor=mask_descriptor(m,intr),mask_score=float(sc[j]),mask_area_fraction=float(m.mean()))
                else:f.update(descriptor=None,mask_score=None,mask_area_fraction=None)
            sync(dev);f['inference_ms']=(time.perf_counter()-t)*1000;out.append(f)
            if (i+1)%100==0:print(f'{kind}: {i+1}/{len(rows)}',flush=True)
    jsonl(dst/'rows.jsonl',out)
    meta.update(status='complete',finished_at=now(),n=len(out),rows_sha256=sha256(dst/'rows.jsonl'))
    times=[r['inference_ms'] for r in out];meta['latency_ms']={'p50':float(np.median(times)),'p95':float(np.quantile(times,.95))}
    dump(dst/'features.json',meta);print(meta,flush=True);return meta


def load_features(feature_set,sp,allow_pilot=False):
    p=outpath('features/'+feature_set);meta=read_json(p/'features.json')
    if meta['status']!='complete':raise ValueError('Incomplete feature extraction')
    if meta.get('pilot') and not allow_pilot:raise ValueError('Pilot features cannot enter publication evaluation')
    if meta['snapshot_sha256']!=sha256(sp/'COMPLETE.json'):raise ValueError('Feature/snapshot mismatch')
    if meta['rows_sha256']!=sha256(p/'rows.jsonl'):raise ValueError('Feature content changed')
    rows=read_jsonl(p/'rows.jsonl');idx={r['frame_id']:r for r in rows}
    if len(idx)!=len(rows):raise ValueError('Duplicate feature IDs')
    return meta,idx
