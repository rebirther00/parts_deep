"""Select 40 images without consulting model scores: one class x one CV group."""
import random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from door8.core import dump,jsonl,read_json,sha256,snapshot


if __name__=='__main__':
    sp,_,rows,splits=snapshot();cfg=read_json(sp/'protocol.json');rng=random.Random(cfg['split_seed']);out=[]
    keys=['bolt_tl','bolt_tr','bolt_bl','bolt_br','corner_hinge','corner_latch']
    for f in range(5):
        for c in cfg['classes']:
            candidates=[r for r in rows if r['class_name']==c and splits['fold_of'][r['group_id']]==f]
            groups=sorted({r['group_id'] for r in candidates});g=rng.choice(groups)
            rr=sorted([r for r in candidates if r['group_id']==g],key=lambda r:r['frame_id']);r=rr[len(rr)//2]
            roles={fold:next(k for k in ['train','val','test'] if g in spec[k]) for fold,spec in splits['partitions'].items()}
            out.append({k:r[k] for k in ['frame_id','session_id','session_dir','group_id','class_name','rgb','rgb_sha256']}|
                {'annotation_id':f"field_{r['session_id']}_{Path(r['rgb']).stem}",'snapshot_sha256':sha256(sp/'COMPLETE.json'),
                 'cv_group':f,'roles_by_fold':roles,'points':{k:None for k in keys},'visible':{k:True for k in keys},
                 'annotation_status':'unannotated','source':'pending_human_manual_field','annotator':''})
    assert len(out)==40 and len({r['frame_id'] for r in out})==40
    jsonl('data/field40_v2_queue.jsonl',out)
    dump('config/field_adaptation_v2_plan.json',{'status':'planned_human_annotations_required','selection_seed':cfg['split_seed'],
         'n_images':40,'selection':'One randomly selected session per class and CV group; middle sampled frame. No model scores used.',
         'per_fold':{'field_train_images':24,'field_validation_images':8,'field_test_localization_images':8},
         'label_budgets_train':[0,8,16,24],'scope':'Retrospective follow-up motivated by the failed first LAB-only transfer trial.',
         'rules':['No VLM or teacher-generated coordinates. A human clicks the original image.',
                  'Never train a shared model on all 40 and call the downstream folds independent.',
                  'For each fold, field point training and checkpoint selection follow the frozen train/val groups.',
                  'The 8 test images with point annotations are for localization evaluation only.',
                  'The LAB-only v1 remains a recorded baseline. Do not overwrite its failed results.',
                  'This plan is not implemented training and is not a completed experiment.']})
    print('Prepared 40 unannotated human-label candidates; no coordinates generated.')
