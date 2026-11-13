"""Post-v1 field adaptation. All evaluations here are TRAIN/VAL development only."""
from collections import Counter
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn

from .core import (ROOT, capture_code, check_source, dump, jsonl, now, partition,
                   read_json, read_jsonl, require_fresh_dir, sha256, snapshot)
from .field_data import field_partitions, load_field_labels, nested_budget
from .features import peaks
from .geometry import bolt_frame, classify_measurement, depth_measure, fit_scale, geometry_gate
from .learning import (HoleData, HoleNet, POINT_KEYS, heatmap_loss, letterbox,
                       load_run, make_loader, pretrained_resnet, seed_all, tensor_image)
from .metrics import summarize

PROTOCOL = ROOT / 'config/field_adaptation_v2_protocol.json'


def development_inputs(fold='0', budget=24):
    plan = read_json(PROTOCOL)
    if str(fold) != plan['development_fold'] or budget not in plan['field_train_budgets']:
        raise ValueError('This protocol authorizes only its recorded development fold and budgets')
    sp, _, frames, splits = snapshot(plan['base_snapshot'])
    lp, _, labels = load_field_labels(plan['labelset'], sp)
    parts = field_partitions(labels, frames, splits, fold)
    selected = nested_budget(parts['train'], read_json(sp/'protocol.json')['classes'], budget,
                             str(fold), plan['budget_selection_seed'])
    lab = read_jsonl(sp/'annotations_lab.jsonl')
    ls = read_json(sp/'annotations_lab_split.json')
    tr = [r for r in lab if r['annotation_id'] in ls['train']] + selected
    lv = [r for r in lab if r['annotation_id'] in ls['val']]
    va = parts['val']
    if len(tr) != 94+budget or len(va) != 8 or len(lv) != 24:
        raise ValueError('Supervision count mismatch')
    for r in tr+va+lv:
        if r['source'] not in ('human_manual_lab', 'human_manual_field') or sha256(r['rgb']) != r['rgb_sha256']:
            raise ValueError('Annotation source mismatch')
    if ({r['rgb_sha256'] for r in tr} & {r['rgb_sha256'] for r in va+lv}
            or {r['group_id'] for r in selected} & {r['group_id'] for r in parts['val']+parts['test']}):
        raise ValueError('Training/held-out supervision leakage')
    return plan, sp, lp, frames, splits, tr, va, lv, selected


@torch.no_grad()
def validation_loss(net, loader, dev):
    net.eval(); values=[]
    for x,y,m in loader:
        with torch.autocast(device_type=dev.type, enabled=dev.type=='cuda'):
            loss=heatmap_loss(net(x.to(dev)),y.to(dev),m.to(dev))
        values.append(float(loss))
    return float(np.mean(values))


def train_field(run, budget, fold='0', seed=42, workers=2, pilot_epochs=0, memory_fraction=.22):
    plan,sp,lp,frames,splits,tr,va,lv,selected=development_inputs(fold,budget)
    if seed not in plan['seeds']:
        raise ValueError('Seed is outside this development protocol')
    hp=plan['training'];total_epochs=pilot_epochs or hp['epochs']
    dst=require_fresh_dir('runs/'+run);seed_all(seed)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dev.type=='cuda':
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        torch.cuda.reset_peak_memory_stats()
    meta=dict(model_kind='holes',training_regime='human_manual_lab_plus_field',
              snapshot=plan['base_snapshot'],snapshot_sha256=sha256(sp/'COMPLETE.json'),
              labelset=plan['labelset'],labelset_sha256=sha256(lp/'COMPLETE.json'),
              development_protocol_sha256=sha256(PROTOCOL),fold=str(fold),seed=seed,
              field_train_budget=budget,field_val_images=len(va),lab_train_images=94,lab_val_images=24,
              train_annotation_ids=[r['annotation_id'] for r in tr],val_annotation_ids=[r['annotation_id'] for r in va],
              lab_monitor_annotation_ids=[r['annotation_id'] for r in lv],
              train_group_ids=sorted({r['group_id'] for r in selected}),val_group_ids=sorted({r['group_id'] for r in va}),
              initialization='ImageNet1K_V1',initialization_sha256=sha256(pretrained_resnet()),
              pilot=bool(pilot_epochs),development_only=True,requested_epochs=total_epochs,
              checkpoint_selection=plan['checkpoint_selection'],hyperparameters=hp,
              gpu_memory_fraction=memory_fraction,status='running',started_at=now(),
              note='Field VAL selects checkpoint; LAB VAL is monitored only. No TEST coordinates or images used.')
    meta.update(capture_code());dump(dst/'run.json',meta);dump(dst/'field_protocol.json',plan)
    t0=time.perf_counter()
    try:
        net=HoleNet().to(dev);opt=torch.optim.AdamW(net.parameters(),lr=hp['learning_rate'])
        dl=make_loader(HoleData(tr,hp,True),hp['batch_size'],True,workers,seed)
        vl=make_loader(HoleData(va,hp,False),hp['batch_size'],False,workers,seed)
        ll=make_loader(HoleData(lv,hp,False),hp['batch_size'],False,workers,seed)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,total_epochs)
        scaler=torch.amp.GradScaler('cuda',enabled=dev.type=='cuda');best=float('inf');history=[]
        for epoch in range(total_epochs):
            net.train();losses=[];start=time.perf_counter()
            for x,y,m in dl:
                x,y,m=x.to(dev),y.to(dev),m.to(dev);opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=dev.type,enabled=dev.type=='cuda'):
                    loss=heatmap_loss(net(x),y,m)
                scaler.scale(loss).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_(net.parameters(),1.)
                scaler.step(opt);scaler.update();losses.append(float(loss.detach()))
            field_loss=validation_loss(net,vl,dev);lab_loss=validation_loss(net,ll,dev)
            record=dict(epoch=epoch+1,train_loss=float(np.mean(losses)),field_val_loss=field_loss,
                        lab_val_loss_monitor_only=lab_loss,seconds=time.perf_counter()-start)
            if not np.isfinite(field_loss):
                raise ValueError('Nonfinite field validation loss')
            history.append(record)
            if field_loss<best:
                best=field_loss;meta.update(best_epoch=epoch+1,best_field_val_loss=best)
                torch.save(net.state_dict(),dst/'model.pth')
            schedule.step();dump(dst/'history.json',history);dump(dst/'run.json',meta)
            print(json.dumps(dict(run=run,**record)),flush=True)
        meta.update(status='complete',finished_at=now(),seconds=time.perf_counter()-t0,
                    weights_sha256=sha256(dst/'model.pth'),optimizer_steps=len(dl)*total_epochs,
                    max_cuda_memory_allocated_bytes=torch.cuda.max_memory_allocated() if dev.type=='cuda' else None)
    except Exception as exc:
        meta.update(status='failed',failed_at=now(),error=repr(exc));dump(dst/'run.json',meta);raise
    dump(dst/'run.json',meta)
    return meta


def check_adaptation_provenance(rm,sp,frames,splits,fold):
    if rm['model_kind']!='holes' or rm['training_regime']!='human_manual_lab_plus_field' or rm['fold']!=str(fold):
        raise ValueError('Expected this fold\'s human field adaptation model')
    lp,_,labels=load_field_labels(rm['labelset'],sp)
    if sha256(lp/'COMPLETE.json')!=rm['labelset_sha256']:
        raise ValueError('Field labels changed since training')
    roles=field_partitions(labels,frames,splits,fold)
    train_ids={r['annotation_id'] for r in roles['train']};val_ids={r['annotation_id'] for r in roles['val']}
    used={r for r in rm['train_annotation_ids'] if r.startswith('field_')}
    if not used<=train_ids or len(used)!=rm['field_train_budget'] or set(rm['val_annotation_ids'])!=val_ids:
        raise ValueError('Field supervision crossed fold roles')
    return roles


def extract_development(run,feature_set,fold='0'):
    sp,_,frames,splits=snapshot();cfg=read_json(sp/'protocol.json');hp=cfg['holes']
    rp,rm=load_run(run,sp);check_adaptation_provenance(rm,sp,frames,splits,fold)
    if rm.get('pilot'):raise ValueError('Pilot cannot enter full development evaluation')
    parts=partition(frames,splits,fold);rows=parts['train']+parts['val']
    test_ids={r['frame_id'] for r in parts['test']}
    if test_ids & {r['frame_id'] for r in rows}:raise ValueError('Test entered development extraction')
    dst=require_fresh_dir('features/'+feature_set);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net=HoleNet(pretrained=False).to(dev);net.load_state_dict(torch.load(rp/'model.pth',map_location=dev,weights_only=True));net.eval()
    meta=dict(status='running',kind='holes',run=run,fold=str(fold),subset='train_val_only',pilot=False,
              snapshot_sha256=sha256(sp/'COMPLETE.json'),weights_sha256=rm['weights_sha256'],
              labelset_sha256=rm['labelset_sha256'],started_at=now(),test_frames_extracted=0,
              code_sha256=sha256(__file__),geometry_code_sha256=sha256(ROOT/'door8/geometry.py'))
    dump(dst/'features.json',meta);features=[]
    with torch.inference_mode():
        for i,row in enumerate(rows):
            check_source(row);im=cv2.imread(row['rgb']);depth=cv2.imread(row['depth'],cv2.IMREAD_UNCHANGED)
            if im is None or depth is None or depth.shape!=im.shape[:2]:raise ValueError('Unreadable/unaligned source pair')
            small,M=letterbox(im,hp['width'],hp['height'])
            with torch.autocast(device_type=dev.type,enabled=dev.type=='cuda'):
                hm=net(tensor_image(small)[None].to(dev))[0].float().cpu().numpy()
            det={k:peaks(hm[ch],n,hp['peak_threshold'],M,hp['stride']) for ch,(k,n) in enumerate([('bolt',4),('hinge',1),('latch',1)])}
            hinge=det['hinge'][0] if det['hinge'] else None;latch=det['latch'][0] if det['latch'] else None
            bf=bolt_frame(det['bolt']);f=dict(frame_id=row['frame_id'],rgb_sha256=row['rgb_sha256'],points=det,
                                           gate=geometry_gate(bf,hinge,latch,im.shape),span_normalized=None)
            if hinge is not None and latch is not None:
                f['span_normalized']=float(np.linalg.norm(np.asarray(hinge[:2])-latch[:2])/row['intrinsics']['fx'])
                measured=depth_measure(depth,det['bolt']+[hinge,latch],hinge,latch,row['intrinsics'])
                if measured:f.update(measured)
            features.append(f)
            if (i+1)%250==0:print(f'{run}: development features {i+1}/{len(rows)}',flush=True)
    jsonl(dst/'rows.jsonl',features);meta.update(status='complete',finished_at=now(),n=len(features),rows_sha256=sha256(dst/'rows.jsonl'))
    dump(dst/'features.json',meta)
    return meta


LOCALIZATION_SCOPE={'val':'Eight field checkpoint-selection VAL images; missing detections fail PCK; not independent test.',
                    'test':'Field TEST-role annotated images of the split; missing detections fail PCK; never used for training or selection.'}


def localization(labels,features,role='val'):
    points=[]
    for row in labels:
        for channel,keys in [('bolt',POINT_KEYS[:4]),('hinge',['corner_hinge']),('latch',['corner_latch'])]:
            keys=[k for k in keys if row['visible'][k] and row['points'][k] is not None]
            gt=np.asarray([row['points'][k] for k in keys]);pred=features[row['frame_id']]['points'][channel];matched={}
            if len(gt) and len(pred):
                errors=np.linalg.norm(gt[:,None,:]-np.asarray(pred)[None,:,:2],axis=2)
                ii,jj=linear_sum_assignment(errors);matched={int(i):float(errors[i,j]) for i,j in zip(ii,jj)}
            for i,key in enumerate(keys):points.append(dict(frame_id=row['frame_id'],key=key,channel=channel,error_px=matched.get(i)))
    def summary(rr):
        ee=[r['error_px'] for r in rr if r['error_px'] is not None]
        return dict(visible_points=len(rr),matched=len(ee),pck8=sum(e<=8 for e in ee)/len(rr) if rr else None,
                    pck16=sum(e<=16 for e in ee)/len(rr) if rr else None,
                    median_error_px_when_matched=float(np.median(ee)) if ee else None,
                    p95_error_px_when_matched=float(np.quantile(ee,.95)) if ee else None)
    return dict(all=summary(points),per_channel={c:summary([r for r in points if r['channel']==c]) for c in ('bolt','hinge','latch')},
                scope=LOCALIZATION_SCOPE[role]),points


def evaluate_development(feature_set,report,fold='0',labelset='field40_20261109_v1',baseline=False):
    sp,_,frames,splits=snapshot();cfg=read_json(sp/'protocol.json');parts=partition(frames,splits,fold)
    lp,_,labels=load_field_labels(labelset,sp);roles=field_partitions(labels,frames,splits,fold)
    fp=ROOT/'features'/feature_set;fm=read_json(fp/'features.json')
    if fm['status']!='complete' or fm.get('pilot') or fm['snapshot_sha256']!=sha256(sp/'COMPLETE.json'):
        raise ValueError('Invalid development feature artifact')
    if sha256(fp/'rows.jsonl')!=fm['rows_sha256']:raise ValueError('Feature content changed')
    rr=read_jsonl(fp/'rows.jsonl');all_idx={r['frame_id']:r for r in rr}
    if len(all_idx)!=len(rr):raise ValueError('Duplicate feature IDs')
    required={r['frame_id'] for r in parts['train']+parts['val']}
    rp,rm=load_run(fm['run'],sp)
    if baseline:
        if rm['training_regime']!='human_manual_lab_only':raise ValueError('Expected zero-field LAB control')
        if set(all_idx)!={r['frame_id'] for r in frames}:raise ValueError('Incomplete v1 control features')
    else:
        check_adaptation_provenance(rm,sp,frames,splits,fold)
        if fm['subset']!='train_val_only' or fm['fold']!=str(fold) or set(all_idx)!=required:
            raise ValueError('Development features must contain exactly TRAIN and VAL, no TEST')
    if fm['weights_sha256']!=rm['weights_sha256']:raise ValueError('Feature/model provenance mismatch')
    features={k:all_idx[k] for k in required}
    if any(features[r['frame_id']]['rgb_sha256']!=r['rgb_sha256'] for r in parts['train']+parts['val']):
        raise ValueError('Feature/image mismatch')
    dst=require_fresh_dir('reports/'+report);cad=dict(zip(cfg['classes'],cfg['cad_hole_distance_mm']))
    loc,points=localization(roles['val'],features);dump(dst/'localization.json',loc);jsonl(dst/'localization_points.jsonl',points)
    result=dict(run=fm['run'],field_train_budget=0 if baseline else rm['field_train_budget'],
                fold=str(fold),subset='validation',development_only=True,test_frames_evaluated=0,
                created_at=now(),feature_set=feature_set,feature_rows_sha256=fm['rows_sha256'],
                labelset_sha256=sha256(lp/'COMPLETE.json'),checkpoint_field_validation=not baseline,
                validation_frames=len(parts['val']),validation_sessions=len({r['group_id'] for r in parts['val']}),
                localization=loc,gates={role:dict(Counter(features[r['frame_id']]['gate'] for r in parts[role])) for role in ('train','val')},
                modes={})
    for mode in ('pixel','depth'):
        failure=None
        try:cal=fit_scale(parts['train'],features,cad,mode)
        except ValueError as exc:
            if not str(exc).startswith('No valid TRAIN measurements'):raise
            failure=str(exc);cal={'failure':failure,'train_frame_ids':[r['frame_id'] for r in parts['train']]}
        if set(cal['train_frame_ids'])!={r['frame_id'] for r in parts['train']}:raise ValueError('Calibration leakage')
        dump(dst/(mode+'_calibration.json'),cal);predictions=[]
        for row in parts['val']:
            pred,value,reason=(None,None,'training_calibration_failure') if failure else classify_measurement(row,features[row['frame_id']],cal,cfg)
            predictions.append({k:row[k] for k in ['frame_id','session_dir','group_id','class_name']}|
                               dict(pred=pred,measurement=value,reason=reason))
        metrics=summarize(predictions,cfg['classes']);jsonl(dst/(mode+'_predictions.jsonl'),predictions)
        dump(dst/(mode+'_metrics.json'),metrics)
        result['modes'][mode]=dict(frame={k:metrics['frame'][k] for k in ['n','correct','abstained','coverage','accuracy_all','macro_f1']},
                                  session={k:metrics['session'][k] for k in ['n','correct','abstained','coverage','accuracy_all','macro_f1']},
                                  failures=dict(Counter(r['reason'] for r in predictions)),calibration_failure=failure)
    dump(dst/'development.json',result)
    print(json.dumps(dict(report=report,budget=result['field_train_budget'],modes=result['modes'],localization=loc),ensure_ascii=False),flush=True)
    return result
