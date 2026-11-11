from __future__ import annotations

import collections
import concurrent.futures
import random
import sqlite3
import subprocess
from pathlib import Path

from .core import ROOT, REPO, SOURCE, check_source, digest, dump, jsonl, now, outpath, partition, protocol, read_json, read_jsonl, require_fresh_dir, sha256, snapshot, write_csv


def uniform_sample(rows, n):
    if len(rows) <= n: return list(rows)
    if n == 1: return [rows[len(rows)//2]]
    return [rows[round(i*(len(rows)-1)/(n-1))] for i in range(n)]


def make_splits(sessions, cfg):
    by = collections.defaultdict(list)
    for s in sessions: by[s['class_name']].append(s['group_id'])
    by = {k:sorted(set(v)) for k,v in by.items()}
    k = cfg['folds']; rng = random.Random(cfg['split_seed']); assignment = {}
    for c in cfg['classes']:
        groups = by[c][:]
        if len(groups) < k: raise ValueError(f'{c}: {len(groups)} groups cannot support {k} folds')
        rng.shuffle(groups)
        for i,g in enumerate(groups): assignment[g] = i % k
    parts = {}
    for f in range(k):
        parts[str(f)] = {p:[] for p in ('train','val','test')}
        for g,j in sorted(assignment.items()):
            role = 'test' if j == f else 'val' if j == (f+1)%k else 'train'
            parts[str(f)][role].append(g)
    # Time split is an explicitly retrospective sensitivity analysis.
    cutoff = cfg['temporal_test_since']; times = collections.defaultdict(list)
    for s in sessions: times[s['group_id']].append(s['session_dir'][:8])
    temporal = {'train':[], 'val':[], 'test':[]}
    for c,groups in by.items():
        # A repeated/duplicate group crossing the boundary stays wholly in test.
        early = sorted(g for g in groups if max(times[g]) < cutoff)
        late = sorted(set(groups)-set(early))
        if len(early)<2 or not late: raise ValueError(f'{c}: cannot construct temporal train/val/test')
        rng.shuffle(early); nval = max(1,round(len(early)*0.15))
        temporal['val'] += early[:nval]; temporal['train'] += early[nval:]; temporal['test'] += late
    parts['temporal'] = temporal
    return {'seed':cfg['split_seed'], 'k':k, 'cv_rule':'test=k; val=(k+1)%K; train=remaining; session groups',
            'temporal_label':'retrospective temporal holdout, previously observed during project development',
            'partitions':parts, 'fold_of':assignment}


def freeze(name='v1'):
    cfg = protocol(); dst = require_fresh_dir('data/snapshots/'+name)
    srcdb = SOURCE/'db/door_pipeline.db'
    # Online backup provides one consistent read-only source snapshot, even with WAL.
    source = sqlite3.connect(srcdb.resolve().as_uri()+'?mode=ro', uri=True)
    target = sqlite3.connect(str(outpath(dst/'source.sqlite')))
    try: source.backup(target)
    finally: target.close(); source.close()
    con = sqlite3.connect((dst/'source.sqlite').as_uri()+'?mode=ro', uri=True); con.row_factory = sqlite3.Row
    records = [dict(x) for x in con.execute('''SELECT s.id,s.session_dir,s.class_name original_label,s.started_at,s.notes,
      c.name current_label,COUNT(i.id) n_images,SUM(i.is_valid) n_valid,SUM(i.synced_local AND i.is_valid) n_local
      FROM capture_sessions s JOIN images i ON i.session_id=s.id JOIN classes c ON c.id=i.class_id
      WHERE s.dataset_id=? GROUP BY s.id,c.name ORDER BY s.session_dir''',(cfg['dataset_id'],))]
    excluded=[]; sessions=[]; frames=[]
    multi=collections.Counter(s['id'] for s in records)
    for s in records:
        reasons=[]
        if s['n_valid'] != s['n_images']: reasons.append('previously_invalid_or_partially_invalid_session')
        if s['current_label']=='E23_door_LH_FRT': reasons.append('E23_out_of_scope')
        elif s['current_label'] not in cfg['classes']: reasons.append('class_out_of_scope')
        if s['original_label'] != s['current_label']: reasons.append('human_confirmed_worker_class_error')
        if multi[s['id']] != 1: reasons.append('multiple_labels_in_session')
        if not s['n_local']: reasons.append('no_local_sample')
        if reasons:
            excluded.append(dict(session_id=s['id'],session_dir=s['session_dir'],original_label=s['original_label'],
                current_label=s['current_label'],reason=';'.join(reasons),n_images=s['n_images'],notes=s['notes'] or ''))
            continue
        candidates=[dict(x) for x in con.execute('SELECT id,rgb_path,depth_path,rgb_filename FROM images WHERE session_id=? AND is_valid AND synced_local ORDER BY rgb_filename',(s['id'],))]
        selected=uniform_sample(candidates,cfg['max_frames_per_session'])
        sm=dict(session_id=s['id'],session_dir=s['session_dir'],class_name=s['original_label'],group_id=s['session_dir'],
                source_pairs=s['n_images'],local_pairs=s['n_local'],selected_frames=len(selected),label_source='worker_original',
                physical_instance_id='',production_lot_id='')
        sessions.append(sm)
        for row in selected:
            rgb=(SOURCE/'datasets_factory_collect'/row['rgb_path']).resolve()
            dep=(SOURCE/'datasets_factory_collect'/row['depth_path']).resolve()
            base=(SOURCE/'datasets_factory_collect').resolve()
            if not rgb.is_relative_to(base) or not dep.is_relative_to(base): raise ValueError('Source path escape')
            mp=rgb.parent/'meta.json'; meta=read_json(mp) if mp.exists() else {}
            intr=meta.get('intrinsics')
            if intr is None:
                intr=dict(serial=54910212,width=1920,height=1200,fx=1269.746,fy=1269.746,cx=961.669,cy=598.594)
                provenance='fixed_hardware_intrinsics_from_project_camera_record'
            else: provenance='session_meta_intrinsics'
            f=dict(frame_id=f"{s['id']}:{row['rgb_filename']}",session_id=s['id'],session_dir=s['session_dir'],
                   group_id=s['session_dir'],class_name=s['original_label'],class_index=cfg['classes'].index(s['original_label']),
                   rgb=str(rgb),depth=str(dep),intrinsics=intr,intrinsics_source=provenance,
                   meta_sha256=sha256(mp) if mp.exists() else None)
            for key,p in [('rgb',rgb),('depth',dep)]:
                st=p.stat(); f[key+'_bytes']=st.st_size; f[key+'_mtime_ns']=st.st_mtime_ns
            frames.append(f)
    con.close()
    if not frames: raise ValueError('No eligible frames')
    print(f'Hashing {len(frames)} RGB/depth pairs from {len(sessions)} sessions',flush=True)
    def hashes(f):
        f=dict(f)
        for key in ('rgb','depth'): f[key+'_sha256']=sha256(f[key])
        return f
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: frames=list(pool.map(hashes,frames))
    # Exact duplicate RGB images join session groups, rather than leaking across folds.
    parent={s['group_id']:s['group_id'] for s in sessions}
    def find(g):
        while parent[g]!=g: parent[g]=parent[parent[g]];g=parent[g]
        return g
    seen={}; joins=[]
    for f in frames:
        old=seen.get(f['rgb_sha256'])
        if old:
            if old['class_name']!=f['class_name']: raise ValueError('Identical RGB has conflicting original labels; human audit required')
            a,b=find(old['group_id']),find(f['group_id'])
            if a!=b: parent[max(a,b)]=min(a,b); joins.append([a,b])
        else: seen[f['rgb_sha256']]=f
    for r in sessions+frames:r['group_id']=find(r['group_id'])
    splits=make_splits(sessions,cfg)
    counts={}
    for f in splits['partitions']:
        parts=partition(frames,splits,f)
        counts[f]={k:dict(frames=len(v),sessions=len({r['session_id'] for r in v}),
                          classes=dict(collections.Counter(r['class_name'] for r in v))) for k,v in parts.items()}
    # Copy manual label CONTENT and provenance; never use automatic/pseudo coordinates.
    annotations=[]; annotation_excluded=[]
    for p in sorted((SOURCE/'labels/holes').glob('*.json')):
        d=read_json(p)
        if d.get('src')!='datasets' or d['cls'] not in cfg['classes']:
            annotation_excluded.append({'source':str(p),'reason':'non_lab_or_out_of_scope'});continue
        rgb=(SOURCE/d['image']).resolve()
        annotations.append(dict(annotation_id=p.stem,class_name=d['cls'],rgb=str(rgb),rgb_sha256=sha256(rgb),
             points=d['points'],visible=d.get('visible',{}),source='human_manual_lab',source_label=str(p),source_label_sha256=sha256(p)))
    arng=random.Random(cfg['split_seed']); lab_val=[]
    for c in cfg['classes']:
        items=[r['annotation_id'] for r in annotations if r['class_name']==c];arng.shuffle(items)
        lab_val+=items[:max(1,round(len(items)*0.2))]
    lab_split={'train':[r['annotation_id'] for r in annotations if r['annotation_id'] not in lab_val], 'val':sorted(lab_val),
               'test':[], 'note':'Auxiliary LAB checkpoint-selection split, NOT an independent physical-instance test. Field sessions supply downstream test.'}
    # A separate annotation queue enables a later field-label ablation; no invented coordinates.
    queue=[]
    for s in sessions:
        rr=[r for r in frames if r['session_id']==s['session_id']]
        for r in uniform_sample(rr,2):
            queue.append(dict(frame_id=r['frame_id'],session_id=r['session_id'],group_id=r['group_id'],class_name=r['class_name'],rgb=r['rgb'],
                              points={k:None for k in ['bolt_tl','bolt_tr','bolt_bl','bolt_br','corner_hinge','corner_latch']},
                              annotation_status='unannotated',annotator='',annotation_time=''))
    source_files=[SOURCE/'hole_classifier.py', SOURCE/'16_train_hole_landmarks.py', SOURCE/'db/decisions.csv', SOURCE/'02_train.py']
    provenance={'created_at':now(),'source_db':str(srcdb),'snapshot_db_sha256':sha256(dst/'source.sqlite'),
                'source_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                'source_files':{str(p):sha256(p) for p in source_files},
                'human_review':'User states all worker-vs-hole class disagreements and radar options were visually reviewed. Corrected class sessions excluded from this study.',
                'source_policy':'Original files are referenced for reading only; no operational module is imported or executed.'}
    summary={'created_at':now(),'snapshot':name,'protocol_sha256':digest(cfg),'eligible_sessions':len(sessions),
             'excluded_sessions':len({r['session_id'] for r in excluded}),'source_pairs':sum(s['source_pairs'] for s in sessions),
             'local_pairs_before_sampling':sum(s['local_pairs'] for s in sessions),'selected_frames':len(frames),
             'classes':{c:dict(sessions=sum(s['class_name']==c for s in sessions),frames=sum(f['class_name']==c for f in frames)) for c in cfg['classes']},
             'auxiliary_lab_annotations':len(annotations),'excluded_nonlab_annotations':len(annotation_excluded),
             'exact_duplicate_group_joins':joins,'split_counts':counts,
             'limitations':['Physical instance IDs and lots are unavailable; sessions are not guaranteed independent physical parts.',
               'This is retrospectively reconstructed data. Prior method development has seen this field period.',
               'Uniform sampling applies to the locally mirrored sample, not necessarily the complete NAS capture.']}
    dump(dst/'protocol.json',cfg);jsonl(dst/'frames.jsonl',frames);write_csv(dst/'sessions.csv',sessions)
    write_csv(dst/'excluded_sessions.csv',excluded);dump(dst/'splits.json',splits)
    jsonl(dst/'annotations_lab.jsonl',annotations);dump(dst/'annotations_lab_split.json',lab_split)
    dump(dst/'excluded_annotations.json',annotation_excluded);jsonl(dst/'field_annotation_queue.jsonl',queue)
    dump(dst/'provenance.json',provenance);dump(dst/'summary.json',summary)
    tracked=[p for p in dst.iterdir() if p.is_file() and p.name!='source.sqlite']
    dump(dst/'COMPLETE.json',{'created_at':now(),'files':{p.name:sha256(p) for p in tracked}})
    print(f'Frozen {name}: {len(sessions)} sessions, {len(frames)} sampled frames, {len(annotations)} LAB labels',flush=True)
    return summary


def audit(name='v1',full_hash=False):
    p,s,rows,splits=snapshot(name)
    if len({r['frame_id'] for r in rows})!=len(rows):raise ValueError('Duplicate frame IDs')
    for f in splits['partitions']: partition(rows,splits,f)
    for r in rows: check_source(r,full_hash)
    annotations=read_jsonl(p/'annotations_lab.jsonl')
    if len({r['annotation_id'] for r in annotations})!=len(annotations):raise ValueError('Duplicate annotation IDs')
    if {a['rgb_sha256'] for a in annotations}&{r['rgb_sha256'] for r in rows}:raise ValueError('LAB and field overlap')
    result={'time':now(),'snapshot':name,'frames':len(rows),'sessions':s['eligible_sessions'],
            'partition_checks':len(splits['partitions']),'source_hash_check':full_hash,'passed':True}
    dump('reports/data_audit.json',result);print(result)
    return result
