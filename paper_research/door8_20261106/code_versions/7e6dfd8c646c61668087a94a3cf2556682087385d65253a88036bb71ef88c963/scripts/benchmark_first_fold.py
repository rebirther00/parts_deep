"""Offline processing-time diagnostic on the same 128 TRAIN images.

Does not measure camera acquisition, disk decoding, networking or dashboard time.
Hole feature time includes both pixel and depth calculations in this implementation.
"""
import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from door8.core import ROOT,check_source,dump,partition,read_json,read_jsonl,setup_runtime,snapshot
setup_runtime()
import cv2
import numpy as np
import torch
from door8.data import uniform_sample
from door8.features import load_features
from door8.learning import Classifier,letterbox,load_run,tensor_image


def timing(values):return {'n':len(values),'p50_ms':float(np.median(values)),'p95_ms':float(np.quantile(values,.95))}


if __name__=='__main__':
    sp,_,allrows,splits=snapshot();cfg=read_json(sp/'protocol.json');train=partition(allrows,splits,'0')['train']
    rows=[]
    for c in cfg['classes']:rows+=uniform_sample([r for r in train if r['class_name']==c],16)
    ids={r['frame_id'] for r in rows};result={}
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def sync():
        if dev.type=='cuda':torch.cuda.synchronize()
    for kind in ('rgb','rgbe'):
        rp,meta=load_run(f'{kind}_f0_s42',sp);net=Classifier(kind,pretrained=False).to(dev).eval()
        net.load_state_dict(torch.load(rp/'model.pth',map_location=dev,weights_only=True));times=[]
        def predict(im):
            small,_=letterbox(im,meta['image_size'],meta['image_size']);x=tensor_image(small)
            if kind=='rgbe':
                edge=cv2.Canny(cv2.cvtColor(small,cv2.COLOR_BGR2GRAY),80,200)
                x=torch.cat([x,torch.from_numpy(edge.copy()).float()[None]/127.5-1],0)
            with torch.autocast(device_type=dev.type,enabled=dev.type=='cuda'):logits=net(x[None].to(dev))
            return int(logits.float().softmax(1).cpu().numpy()[0].argmax())
        with torch.inference_mode():
            warm=cv2.imread(rows[0]['rgb'])
            for _ in range(10):predict(warm)
            for r in rows:
                check_source(r)
                im=cv2.imread(r['rgb']);sync();t=time.perf_counter();predict(im);sync();times.append((time.perf_counter()-t)*1000)
        result[kind]=timing(times)
        del net
    for feature_set,label in [('holes_lab_s42_full','holes_shared_pixel_and_depth_features'),('mobile_sam_full','mobile_sam_dimensions_features')]:
        meta,f=load_features(feature_set,sp)
        times=[x['inference_ms'] for x in f.values() if x['frame_id'] in ids]
        if len(times)!=len(rows):raise ValueError('Timing input mismatch')
        result[label]=timing(times)
    output={'scope':__doc__,'batch_size':1,'gpu':torch.cuda.get_device_name() if dev.type=='cuda' else None,
        'train_frame_ids':[r['frame_id'] for r in rows],'timings':result,
        'limitations':['CNN times include argmax; feature times exclude final nearest-class lookup.',
            'Hole feature path computes pixel span and depth plane together; these are not separate optimized runtimes.',
            'Offline timings are not an end-to-end live-system latency claim.']}
    dump('reports/offline_processing_time.json',output);print(result)
