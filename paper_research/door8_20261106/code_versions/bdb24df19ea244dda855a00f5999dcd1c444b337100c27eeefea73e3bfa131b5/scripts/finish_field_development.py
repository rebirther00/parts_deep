"""Sequential, restartable development jobs. Never evaluates the TEST partition."""
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from door8.core import ROOT,dump,now,read_json,sha256


def job(command,artifact,stage,expected):
    if artifact.exists():
        meta=read_json(artifact)
        if any(meta.get(k)!=v for k,v in expected.items()):
            raise RuntimeError(f'Existing artifact needs review, not overwrite: {artifact}')
        print('SKIP '+stage,flush=True);return
    log=ROOT/'reports/field_v2_development_logs'/(stage+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    print('START '+stage,flush=True)
    with log.open('x',encoding='utf8') as stream:
        subprocess.run([sys.executable,*command],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    meta=read_json(artifact)
    if any(meta.get(k)!=v for k,v in expected.items()):raise RuntimeError('Incomplete stage: '+stage)
    print('DONE '+stage,flush=True)


if __name__=='__main__':
    status_path=ROOT/'reports/field_v2_workflow_status.json'
    status=dict(started_at=now(),status='running',scope='fold0 seed42 TRAIN/VAL only',budgets=[24,8,16],completed=[])
    dump(status_path,status)
    try:
        for budget in status['budgets']:
            run=f'holes_field{budget}_f0_s42_v2';feature=run+'_development';report=run+'_validation'
            status.update(current_budget=budget,current_stage='train');dump(status_path,status)
            job(['scripts/run_field_adaptation.py','train','--run',run,'--budget',str(budget)],
                ROOT/'runs'/run/'run.json',run+'_train',
                {'status':'complete','pilot':False,'development_protocol_sha256':sha256(ROOT/'config/field_adaptation_v2_protocol.json')})
            status.update(current_stage='extract');dump(status_path,status)
            job(['scripts/run_field_adaptation.py','extract','--run',run,'--features',feature],
                ROOT/'features'/feature/'features.json',run+'_extract',{'status':'complete','subset':'train_val_only','pilot':False})
            status.update(current_stage='evaluate');dump(status_path,status)
            job(['scripts/run_field_adaptation.py','evaluate','--features',feature,'--report',report],
                ROOT/'reports'/report/'development.json',run+'_evaluate',{'development_only':True,'test_frames_evaluated':0})
            status['completed'].append(budget);dump(status_path,status)
        status.update(status='complete',finished_at=now(),current_stage='complete');dump(status_path,status)
    except Exception as exc:
        status.update(status='failed',error=repr(exc),failed_at=now());dump(status_path,status);raise
