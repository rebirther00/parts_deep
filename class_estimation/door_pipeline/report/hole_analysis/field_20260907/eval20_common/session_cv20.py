"""미리 만든 뷰(<view>/<class>/rgb_<date>_s_<sess>_<idx>.png)로 세션 그룹 K-fold CV. DB 미변경·DB 로그 없음.
사용: python session_cv20.py <view_dir> <out_root> [K]"""
import os, sys, json, glob, subprocess, collections, shutil, time
from pathlib import Path
DOOR=Path('/home/koceti/parts_deep/class_estimation/door_pipeline'); PY=os.path.expanduser('~/parts_deep/venv/parts_deep/bin/python')
VIEW=Path(sys.argv[1]); OUT=Path(sys.argv[2]); K=int(sys.argv[3]) if len(sys.argv)>3 else 5
files=sorted(glob.glob(str(VIEW/'*'/'rgb_*.png')))
by=collections.defaultdict(lambda: collections.defaultdict(list))   # cls -> sess -> files
for f in files:
    cls=Path(f).parent.name; sess=Path(f).name[4:-4].rsplit('_',1)[0]; by[cls][sess].append(f)
fold_of={}
for cls,ss in by.items():
    keys=sorted(ss)   # 파일명이 날짜_세션이라 시간순
    for i,k in enumerate(keys): fold_of[(cls,k)]=(i%K) if len(keys)>=3 else -1
def build(root,k):
    if root.exists(): shutil.rmtree(root)
    n=collections.Counter()
    for cls,ss in by.items():
        for sess,fs in ss.items():
            t='test' if fold_of[(cls,sess)]==k else 'train'; d=root/t/cls; d.mkdir(parents=True,exist_ok=True)
            for f in fs:
                os.symlink(os.path.realpath(f), d/Path(f).name); dp=f.replace('rgb_','depth_')
                if os.path.exists(dp): os.symlink(os.path.realpath(dp), d/Path(dp).name)
                n[t]+=1
    return n
OUT.mkdir(parents=True,exist_ok=True); json.dump({'K':K,'fold_of':{f'{c}/{s}':v for (c,s),v in fold_of.items()}},open(OUT/'folds.json','w'),ensure_ascii=False,indent=1)
results={}
for k in range(K):
    root=OUT/f'fold{k}'; n=build(root,k); print(f'\n=== fold {k}: test {n["test"]}장 train {n["train"]}장',flush=True)
    run=DOOR/'artifacts'/f'rgbe_noaux_448_seed42_fold{k}'
    if run.exists(): shutil.rmtree(run)
    env=dict(os.environ,DOOR_DB_LOG='0'); t0=time.time()
    subprocess.run([PY,'-u',str(DOOR/'02_train.py'),'--model_type','rgbe','--no_aux','--image_size','448','--seed','42','--dataset_dir',str(root),'--presplit'],env=env,check=True,stdout=open(OUT/f'train_fold{k}.log','w'),stderr=subprocess.STDOUT,cwd=DOOR)
    subprocess.run([PY,'-u',str(DOOR/'04_evaluate_factory.py'),'--model',str(run/'model.pth'),'--dataset_dir',str(root/'test')],env=env,check=True,stdout=open(OUT/f'eval_fold{k}.log','w'),stderr=subprocess.STDOUT,cwd=DOOR)
    res=json.load(open(run/'factory_eval_results.json'))
    results[k]=dict(accuracy=res['accuracy'],total=res['total_samples'],correct=res['correct_samples'],class_acc=res['class_accuracies'],minutes=round((time.time()-t0)/60,1))
    print(f'fold {k}: {res["accuracy"]}% ({res["correct_samples"]}/{res["total_samples"]}) {results[k]["minutes"]}분',flush=True)
    json.dump(results,open(OUT/'cv_results.json','w'),ensure_ascii=False,indent=1)
tot=sum(r['total'] for r in results.values()); cor=sum(r['correct'] for r in results.values())
print(f'\n[{K}-fold 세션 CV] {cor}/{tot} = {100*cor/tot:.2f}%  fold별 {[r["accuracy"] for r in results.values()]}')
