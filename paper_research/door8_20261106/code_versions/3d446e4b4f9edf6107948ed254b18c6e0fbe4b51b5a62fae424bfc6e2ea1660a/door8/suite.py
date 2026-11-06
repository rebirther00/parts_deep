"""Sequential, restartable experiment matrix. Never runs on import."""
import shlex
import subprocess
import sys
from .core import ROOT, dump, outpath, read_json, sha256, snapshot


def suite(name='v1',execute=False,temporal=False):
    sp,_,_,_=snapshot(name);cfg=read_json(sp/'protocol.json');jobs=[]
    def add(args,marker):jobs.append({'args':args,'completion_marker':marker})
    add(['extract','--kind','mobile_sam','--features','mobile_sam_full'],'features/mobile_sam_full/features.json')
    folds=[str(i) for i in range(cfg['folds'])]+(['temporal'] if temporal else [])
    for seed in cfg['seeds']:
        hole=f'holes_lab_s{seed}';feat=f'{hole}_full'
        add(['train-holes','--seed',str(seed),'--run',hole],f'runs/{hole}/run.json')
        add(['extract','--kind','holes','--run',hole,'--features',feat],f'features/{feat}/features.json')
        for fold in folds:
            reports=[]
            for kind in ('rgb','rgbe'):
                run=f'{kind}_f{fold}_s{seed}';report=run+'_test';reports.append(report)
                add(['train-cnn','--kind',kind,'--fold',fold,'--seed',str(seed),'--run',run],f'runs/{run}/run.json')
                add(['eval-cnn','--run',run,'--report',report],f'reports/{report}/report.json')
            for mode in ('pixel','depth'):
                report=f'holes_{mode}_f{fold}_s{seed}';reports.append(report)
                add(['eval-geometry','--features',feat,'--mode',mode,'--fold',fold,'--report',report],f'reports/{report}/report.json')
            mask=f'mask_f{fold}'
            if seed==cfg['seeds'][0]:
                add(['eval-geometry','--features','mobile_sam_full','--mode','mask','--fold',fold,'--report',mask],f'reports/{mask}/report.json')
            reports.append(mask);comp=f'comparison_f{fold}_s{seed}'
            add(['compare','--reports',*reports,'--output',comp],f'reports/{comp}/comparison.json')
    for seed in cfg['seeds']:
        pooled=[]
        for kind in ('rgb','rgbe','holes_pixel','holes_depth'):
            reports=[f'{kind}_f{f}_s{seed}'+('_test' if kind in ('rgb','rgbe') else '') for f in range(cfg['folds'])]
            output=f'{kind}_oof_s{seed}';pooled.append(output)
            add(['collect-oof','--reports',*reports,'--output',output],f'reports/{output}/report.json')
        if seed==cfg['seeds'][0]:
            add(['collect-oof','--reports',*[f'mask_f{f}' for f in range(cfg['folds'])],'--output','mask_oof'],'reports/mask_oof/report.json')
        pooled.append('mask_oof');output=f'comparison_oof_s{seed}'
        add(['compare','--reports',*pooled,'--output',output],f'reports/{output}/comparison.json')
    dump('reports/experiment_matrix.json',{'snapshot':name,'snapshot_sha256':sha256(sp/'COMPLETE.json'),'jobs':jobs,
          'scope':'All CV; optional retrospective temporal split. No future prospective results.',
          'resume_policy':'Skip completed matching artifacts only; incomplete directories cause a stop, never overwrite.'})
    for job in jobs:
        cmd=[sys.executable,str(ROOT/'run.py'),'--snapshot',name,*job['args']]
        print(shlex.join(cmd),flush=True)
        if execute:
            marker=outpath(job['completion_marker'])
            if marker.exists():
                meta=read_json(marker)
                if marker.name=='comparison.json':
                    if meta.get('snapshot_sha256')!=sha256(sp/'COMPLETE.json'):raise ValueError('Comparison snapshot mismatch')
                elif meta.get('status')!='complete' or meta.get('pilot') or meta.get('snapshot_sha256')!=sha256(sp/'COMPLETE.json'):
                    raise ValueError('Existing artifact is incomplete, pilot, or from another snapshot: '+str(marker))
                # Downstream loaders additionally verify model / prediction / feature hashes.
                continue
            subprocess.run(cmd,cwd=ROOT,check=True)


def collect_oof(name,reports,output):
    """Pool the five mutually exclusive test folds once, never pool seed repeats."""
    from .core import jsonl, require_fresh_dir
    from .metrics import summarize
    sp,_,rows,splits=snapshot(name);cfg=read_json(sp/'protocol.json');metas=[];pred=[]
    from .core import read_jsonl
    for report in reports:
        p=outpath('reports/'+report);m=read_json(p/'report.json')
        if m.get('pilot') or m['subset']!='test' or m.get('status')!='complete':raise ValueError('Not an official test report')
        if m['snapshot_sha256']!=sha256(sp/'COMPLETE.json') or m['predictions_sha256']!=sha256(p/'predictions.jsonl'):raise ValueError('Provenance mismatch')
        metas.append(m);pred+=read_jsonl(p/'predictions.jsonl')
    if {m['fold'] for m in metas}!={str(i) for i in range(cfg['folds'])} or len(metas)!=cfg['folds']:raise ValueError('Need exactly five distinct test folds')
    if len({m['method'] for m in metas})!=1 or len({m.get('seed') for m in metas})!=1 or len({m.get('feature_set') for m in metas})!=1:
        raise ValueError('Cannot mix methods or seeds')
    if len(pred)!=len(rows) or {r['frame_id'] for r in pred}!={r['frame_id'] for r in rows}:raise ValueError('OOF rows missing or repeated')
    # Verify every report actually contains its assigned fold, not merely a disjoint set.
    from .core import partition
    for report,m in zip(reports,metas):
        ids={r['frame_id'] for r in read_jsonl(outpath('reports/'+report)/'predictions.jsonl')}
        if ids!={r['frame_id'] for r in partition(rows,splits,m['fold'])['test']}:raise ValueError('Wrong test-fold membership')
    dst=require_fresh_dir('reports/'+output);jsonl(dst/'predictions.jsonl',pred);dump(dst/'metrics.json',summarize(pred,cfg['classes']))
    dump(dst/'report.json',{'status':'complete','pilot':False,'subset':'test','fold':'oof','method':metas[0]['method'],
        'seed':metas[0].get('seed'),'feature_set':metas[0].get('feature_set'),'source_reports':reports,
        'snapshot_sha256':sha256(sp/'COMPLETE.json'),'predictions_sha256':sha256(dst/'predictions.jsonl'),
        'note':'One OOF prediction per frame and seed. CV fold models share training data; CI is descriptive.'})
