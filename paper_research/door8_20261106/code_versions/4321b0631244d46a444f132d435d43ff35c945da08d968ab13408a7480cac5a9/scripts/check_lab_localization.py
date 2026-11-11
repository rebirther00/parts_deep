"""Localization diagnostic on the LAB checkpoint-selection set, never a test claim."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from door8.core import dump,jsonl,read_json,read_jsonl,setup_runtime,snapshot
setup_runtime()
import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from door8.features import peaks
from door8.learning import HoleNet,letterbox,load_run,tensor_image,POINT_KEYS


if __name__=='__main__':
    sp,_,_,_=snapshot();rp,meta=load_run('holes_lab_s42',sp);cfg=read_json(sp/'protocol.json');hp=cfg['holes']
    ids=set(read_json(sp/'annotations_lab_split.json')['val'])
    rows=[r for r in read_jsonl(sp/'annotations_lab.jsonl') if r['annotation_id'] in ids]
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');net=HoleNet(pretrained=False).to(dev).eval()
    net.load_state_dict(torch.load(rp/'model.pth',map_location=dev,weights_only=True));all_points=[]
    with torch.inference_mode():
        for r in rows:
            im=cv2.imread(r['rgb']);small,M=letterbox(im,hp['width'],hp['height'])
            with torch.autocast(device_type=dev.type,enabled=dev.type=='cuda'):heat=net(tensor_image(small)[None].to(dev))
            heat=heat[0].float().cpu().numpy()
            for ch,keys in enumerate([POINT_KEYS[:4],['corner_hinge'],['corner_latch']]):
                keys=[k for k in keys if r['points'].get(k) is not None and r.get('visible',{}).get(k,True)]
                gt=np.asarray([r['points'][k] for k in keys]);det=peaks(heat[ch],4 if ch==0 else 1,hp['peak_threshold'],M,hp['stride'])
                matched={}
                if len(gt) and len(det):
                    errors=np.linalg.norm(gt[:,None,:]-np.asarray(det)[None,:,:2],axis=2)
                    ii,jj=linear_sum_assignment(errors);matched={int(i):float(errors[i,j]) for i,j in zip(ii,jj)}
                for i,k in enumerate(keys):all_points.append({'annotation_id':r['annotation_id'],'key':k,'error_original_px':matched.get(i),
                    'matched_above_threshold':i in matched})
    errors=[r['error_original_px'] for r in all_points if r['error_original_px'] is not None]
    result={'run':'holes_lab_s42','subset':'LAB checkpoint-selection validation','not_independent_test':True,
        'images':len(rows),'visible_gt_points':len(all_points),'matched_points':len(errors),
        'median_error_original_px_when_matched':float(np.median(errors)) if errors else None,
        'pck8_original_px_all_visible':sum(e<=8 for e in errors)/len(all_points),
        'pck16_original_px_all_visible':sum(e<=16 for e in errors)/len(all_points),
        'note':'Missing detections count as failures in PCK. Unordered bolt detections are matched by minimum-cost assignment. Diagnostic only; not used to retune the frozen first-fold model.'}
    dump('reports/lab_localization_diagnostic.json',result);jsonl('reports/lab_localization_points.jsonl',all_points);print(result)
