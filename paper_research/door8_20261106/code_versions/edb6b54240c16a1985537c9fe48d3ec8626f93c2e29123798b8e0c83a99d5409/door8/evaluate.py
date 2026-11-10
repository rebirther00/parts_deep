from collections import Counter
import copy
import numpy as np
import torch
from .core import dump, jsonl, now, outpath, partition, read_json, read_jsonl, require_fresh_dir, sha256, snapshot
from .features import load_features
from .geometry import classify_measurement, fit_mask, fit_scale, predict_mask
from .learning import CNNData, Classifier, cnn_predict, load_run, make_loader
from .metrics import paired_session_comparison, summarize


def save_report(dst,pred,cfg,meta):
    jsonl(dst/'predictions.jsonl',pred);dump(dst/'metrics.json',summarize(pred,cfg['classes']))
    meta.update(status='complete',created_at=now(),n=len(pred),predictions_sha256=sha256(dst/'predictions.jsonl'),
                evaluation_code_sha256=sha256(__file__))
    dump(dst/'report.json',meta)
    print({k:v for k,v in summarize(pred,cfg['classes'])['session'].items() if k in ['n','accuracy_all','coverage','macro_f1']},flush=True)


def evaluate_cnn(name,run,report,workers=4,pilot=False):
    sp,_,rows,splits=snapshot(name);cfg=read_json(sp/'protocol.json');rp,rm=load_run(run,sp,allow_pilot=pilot)
    if rm['model_kind'] not in ('rgb','rgbe'):raise ValueError('Expected CNN model')
    if bool(rm['pilot'])!=pilot:raise ValueError('Explicit pilot flag must match training')
    parts=partition(rows,splits,rm['fold'])
    if not pilot:
        if set(rm['train_frame_ids'])!={r['frame_id'] for r in parts['train']} or set(rm['val_frame_ids'])!={r['frame_id'] for r in parts['val']}:
            raise ValueError('Training IDs do not match the frozen fold')
    subset='pilot_validation' if pilot else 'test'
    rr=[r for r in parts['val'] if r['frame_id'] in rm['val_frame_ids']] if pilot else parts['test']
    dst=require_fresh_dir('reports/'+report);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net=Classifier(rm['model_kind'],len(cfg['classes']),pretrained=False).to(dev)
    net.load_state_dict(torch.load(rp/'model.pth',map_location=dev,weights_only=True))
    loader=make_loader(CNNData(rr,rm['model_kind'],rm['image_size'],False),cfg['cnn']['batch_size'],False,workers,rm['seed'])
    pred,_=cnn_predict(net,loader,rr,cfg['classes'],dev)
    # Prespecified secondary operating point: retain >=95% of VAL frames, without looking at test.
    vp=read_jsonl(rp/'val_predictions.jsonl');conf=sorted(r['confidence'] for r in vp)
    threshold=conf[int(.05*len(conf))] if conf else 0.
    rejected=[dict(r,pred=r['pred'] if r['confidence']>=threshold else None) for r in pred]
    jsonl(dst/'predictions_val95coverage.jsonl',rejected)
    dump(dst/'metrics_val95coverage.json',summarize(rejected,cfg['classes']))
    meta={'method':rm['model_kind'],'run':run,'fold':rm['fold'],'seed':rm['seed'],'subset':subset,'pilot':pilot,
          'snapshot_sha256':sha256(sp/'COMPLETE.json'),'weights_sha256':rm['weights_sha256'],
          'val95coverage_threshold':threshold,'scope':cfg['scope']}
    save_report(dst,pred,cfg,meta)


def evaluate_geometry(name,feature_set,fold,report,mode='pixel',gates=True,pose=False):
    sp,_,rows,splits=snapshot(name);cfg=read_json(sp/'protocol.json');parts=partition(rows,splits,fold)
    fm,features=load_features(feature_set,sp)
    if set(features)!={r['frame_id'] for r in rows}:raise ValueError('Features must cover every frozen frame')
    if any(features[r['frame_id']]['rgb_sha256']!=r['rgb_sha256'] for r in rows):raise ValueError('Feature/source mismatch')
    calibration_failure=None
    try:
        if mode=='mask':
            if fm['kind']!='mobile_sam':raise ValueError('Expected segmentation features')
            cal=fit_mask(parts['train'],features,cfg['classes'])
        else:
            if fm['kind']!='holes':raise ValueError('Expected landmark features')
            cal=fit_scale(parts['train'],features,dict(zip(cfg['classes'],cfg['cad_hole_distance_mm'])),mode)
    except ValueError as error:
        if not str(error).startswith(('No valid TRAIN measurements','Missing class in TRAIN mask features')):raise
        # A frozen method that cannot calibrate on TRAIN must not quietly drop
        # that fold or borrow held-out examples. Score it as total abstention.
        calibration_failure=str(error)
        cal={'failure':calibration_failure,'train_frame_ids':[r['frame_id'] for r in parts['train']],
             'train_group_ids':sorted({r['group_id'] for r in parts['train']})}
    # Never let a helper return calibration provenance covering the test set.
    if set(cal['train_frame_ids'])!={r['frame_id'] for r in parts['train']}:raise ValueError('Calibration leakage')
    dst=require_fresh_dir('reports/'+report);dump(dst/'calibration.json',cal);pred=[]
    for r in parts['test']:
        f=features[r['frame_id']]
        if calibration_failure:p,value,reason=None,None,'training_calibration_failure'
        else:p,value,reason=predict_mask(f,cal) if mode=='mask' else classify_measurement(r,f,cal,cfg,gates,pose)
        pred.append({k:r[k] for k in ['frame_id','session_dir','group_id','class_name']}|
                    {'pred':p,'measurement':value,'reason':reason,'inference_ms':f['inference_ms']})
    dump(dst/'failure_counts.json',dict(Counter(p['reason'] for p in pred)))
    if mode!='mask':
        cad=dict(zip(cfg['classes'],cfg['cad_hole_distance_mm']));residual={}
        for c in cfg['classes']:
            values=[abs(p['measurement']-cad[c]) for p in pred if p['class_name']==c and p['measurement'] is not None]
            residual[c]={'n_measurable':len(values),'n_all':sum(p['class_name']==c for p in pred),
                'median_abs_cad_residual_mm':float(np.median(values)) if values else None,
                'p95_abs_cad_residual_mm':float(np.quantile(values,.95)) if values else None}
        dump(dst/'cad_residuals.json',{'per_class':residual,
            'scope':'Only measurements surviving requested geometry/pose/range gates; CAD reference is not measured physical ground truth.'})
    save_report(dst,pred,cfg,{'method':('mobile_sam_dimensions' if mode=='mask' else 'holes_'+mode),
        'fold':str(fold),'feature_set':feature_set,'features_sha256':fm['rows_sha256'],
        'weights_sha256':fm['weights_sha256'],'subset':'test','pilot':False,
        'snapshot_sha256':sha256(sp/'COMPLETE.json'),'gates':gates,'pose_gate':pose,
        'calibration_failure':calibration_failure,'scope':cfg['scope']})


def compare(name,reports,output):
    sp,_,_,_=snapshot(name);cfg=read_json(sp/'protocol.json');loaded=[];table=[]
    for report in reports:
        p=outpath('reports/'+report);meta=read_json(p/'report.json')
        if meta['pilot'] or meta['subset']!='test' or meta['status']!='complete':raise ValueError('Comparison requires completed nonpilot test evaluations')
        if meta['snapshot_sha256']!=sha256(sp/'COMPLETE.json') or meta['predictions_sha256']!=sha256(p/'predictions.jsonl'):raise ValueError('Report provenance mismatch')
        pred=read_jsonl(p/'predictions.jsonl');loaded.append((report,meta,pred));met=summarize(pred,cfg['classes'])
        table.append({'report':report,'method':meta['method'],'fold':meta['fold'],
            'frame_accuracy_all':met['frame']['accuracy_all'],'frame_macro_f1':met['frame']['macro_f1'],
            'frame_coverage':met['frame']['coverage'],'session_accuracy_all':met['session']['accuracy_all'],
            'session_macro_f1':met['session']['macro_f1'],'session_coverage':met['session']['coverage']})
    tests=[]
    for i,(a,ma,pa) in enumerate(loaded):
        for b,mb,pb in loaded[i+1:]:
            if ma['fold']!=mb['fold']:raise ValueError('Pairwise comparisons require the same fold')
            tests.append({'a':a,'b':b,**paired_session_comparison(pa,pb,cfg['classes'])})
    # Control familywise error across all pairs requested in this comparison.
    previous=0.;total=len(tests)
    for rank,i in enumerate(sorted(range(total),key=lambda i:tests[i]['exact_mcnemar_p_session'])):
        previous=max(previous,min(1.,(total-rank)*tests[i]['exact_mcnemar_p_session']))
        tests[i]['holm_adjusted_p']=previous
    dst=require_fresh_dir('reports/'+output);dump(dst/'comparison.json',{'snapshot_sha256':sha256(sp/'COMPLETE.json'),'table':table,'paired_session_tests':tests,
          'note':'Exploratory session-level comparisons; raw and Holm-adjusted p values are reported. Physical-instance independence is unverified.'})
    from .core import write_csv
    write_csv(dst/'comparison.csv',table)
