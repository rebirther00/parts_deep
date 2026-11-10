"""Read-only review of v7 revisions; write only a new review directory."""
from pathlib import Path
from collections import Counter, defaultdict
import ast
import hashlib
import json
import math
import statistics as st
import subprocess
import sys
import zipfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'reports/논문초안_v7_리뷰_20261110'
sys.path.insert(0, str(ROOT))
from door8.geometry import classify_measurement, fit_scale

def read(p): return json.loads(p.read_text())
def rows(p): return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name, obj): (DEST / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def main():
    DEST.mkdir(exist_ok=False)
    names = [f'manuscript/논문초안_v7.{ext}' for ext in ['hwpx', 'pdf', 'md', 'docx']]
    names += [f'manuscript/v7_{s}.md' for s in ['검토답변', '작성확인사항', '차별성_검토']]
    hashes = {p: digest(ROOT / p) for p in names}
    save('원본해시.json', hashes)
    result = dict(original_sha256=hashes)
    # Native file, not just the companion Markdown.
    hp = 'http://www.hancom.co.kr/hwpml/2011/paragraph'
    with zipfile.ZipFile(ROOT / names[0]) as z:
        root = ET.fromstring(z.read('Contents/section0.xml'))
        paragraphs = [dict(id=p.get('id'), text=''.join(p.itertext())) for p in root.iter('{'+hp+'}p')]
        result['native_hwpx'] = dict(top_paragraphs=len(root.findall('{'+hp+'}p')), paragraphs=len(paragraphs),
            tables=len(list(root.iter('{'+hp+'}tbl'))), figures=len(list(root.iter('{'+hp+'}pic'))))
    save('HWPX문단.json', paragraphs)
    (DEST / 'HWPX본문추출.txt').write_text('\n\n'.join(p['text'] for p in paragraphs))
    native = ''.join(p['text'] for p in paragraphs)
    for s in ['측정 가능한 모든 시험 영상', '21,259건 중 1건', '서로 다른 실물 8개', '총 8장·48점', '95.0~95.1%', 'qj = pj / Σi≠c pi']:
        assert s in native, s
    v6, v7 = [(ROOT / f'manuscript/논문초안_v{n}.md').read_text() for n in (6,7)]
    assert [s for s in v6.splitlines() if s.startswith('|')] == [s for s in v7.splitlines() if s.startswith('|')]
    result['all_markdown_table_rows_unchanged'] = True
    subprocess.run(['pdftotext', '-layout', str(ROOT / names[1]), str(DEST / 'PDF본문추출.txt')], check=True)
    # Existing errata refers to immutable outputs: check every file and JSON pointer.
    err = read(ROOT / 'reports/errata_localization_scope_20261110.json')
    checked = []
    for a in err['affected_artifacts']:
        p = ROOT / a['file']; assert digest(p) == a['sha256']
        d = read(p)
        for pointer in a['json_paths']:
            node = d
            for k in pointer.strip('/').split('/'): node = node[k]
            assert node == err['wrong_string']
        checked.append(dict(file=a['file'], paths=len(a['json_paths']), hash_verified=True))
    logs = []
    for p in err['affected_logs']:
        f = ROOT / p
        assert err['wrong_string'] in f.read_text()
        logs.append(dict(file=p, sha256=digest(f), listed_as_string_only=True))
    # Execute only the pure localization function from the current module, no model loading.
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    tree = ast.parse((ROOT / 'door8/field_experiments.py').read_text())
    selected = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name == 'localization') or
        (isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id == 'LOCALIZATION_SCOPE' for t in n.targets))]
    labelrows = rows(ROOT / 'data/annotation_snapshots/field40_20261109_v1/annotations.jsonl')
    keys = ['bolt_tl','bolt_tr','bolt_br','bolt_bl','corner_hinge','corner_latch']
    # Read the authoritative channel ordering from the module assigning POINT_KEYS.
    lt = ast.parse((ROOT / 'door8/learning.py').read_text())
    for n in lt.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='POINT_KEYS' for t in n.targets):
            keys = ast.literal_eval(n.value)
    ns = dict(np=np, linear_sum_assignment=linear_sum_assignment, POINT_KEYS=keys)
    exec(compile(ast.Module(body=selected,type_ignores=[]), 'localization-reviewed', 'exec'), ns)
    result['scope_errata'] = dict(artifacts=checked, log_verification=logs, logs_have_hash_in_author_errata=False)
    # Recompute residual bands from stored points and TRAIN-only scales, no inference.
    sp = ROOT / 'data/snapshots/v1'
    cfg, splits, frames = read(sp/'protocol.json'), read(sp/'splits.json')['partitions'], rows(sp/'frames.jsonl')
    cad = dict(zip(cfg['classes'], cfg['cad_hole_distance_mm']))
    margin = read(ROOT / 'reports/extension_v5/margin.json')
    band = margin['bands']; details={}; outside=[]; exact_ties=[]; counterexamples=[]
    for fold in ['0','1','2','3','4','temporal']:
        files = (['holes_field8_f0_s42_v2_development','holes_field8_f0_s42_v2_f0_test_v3'] if fold=='0' else
                 [f'holes_field8_f{fold}_s42_v3_trainTest'] if fold!='temporal' else ['holes_field_temporal_s42_trainTest'])
        pool = {r['frame_id']:r for f in files for r in rows(ROOT/f'features/{f}/rows.jsonl')}
        parts = {role:[r for r in frames if r['group_id'] in set(gg)] for role,gg in splits[fold].items()}
        if fold=='0':
            labels = [r for r in labelrows if r['group_id'] in set(splits[fold]['test'])]
            val, vp = ns['localization'](labels, pool)
            test, tp = ns['localization'](labels, pool, role='test')
            assert vp == tp and {k:v for k,v in val.items() if k!='scope'} == {k:v for k,v in test.items() if k!='scope'}
            assert len(labels)==8 and test['all']['visible_points']==48 and test['all']['pck8']==1
            result['localization_role_test'] = dict(n_images=8, n_points=48, numeric_outputs_identical=True, scope=test['scope'])
        for mode in ['pixel','depth']:
            cal = fit_scale(parts['train'],pool,cad,mode)
            assert cal['camera_scales'] == margin['splits'][fold+':'+mode]['scale']
            for role in ['train','test']:
                grouped=defaultdict(list); measured=0; out=0
                for r in parts[role]:
                    pred,d,reason=classify_measurement(r,pool[r['frame_id']],cal,cfg)
                    residual=None if d is None else d/cad[r['class_name']]-1
                    grouped[r['group_id']].append((r,pred,residual))
                    if d is None: continue
                    measured+=1; lo,hi=band[r['class_name']]
                    ob=residual<lo or residual>hi;out+=ob
                    if mode=='pixel' and ob:
                        outside.append(dict(fold=fold, role=role, frame_id=r['frame_id'],session=r['session_dir'],truth=r['class_name'],pred=pred,measurement=d,residual=residual,band=[lo,hi],reason=reason))
                    distances=sorted(abs(v-d) for v in cad.values())
                    if math.isclose(distances[0],distances[1],rel_tol=0,abs_tol=1e-10):
                        exact_ties.append(dict(fold=fold,mode=mode,role=role,frame_id=r['frame_id']))
                saved=margin['splits'][fold+':'+mode][role]
                assert (measured,out)==(saved['frames_measured'],saved['frames_out_of_band'])
                details[fold+':'+mode+':'+role]=dict(measured=measured,outside=out)
                for rr in grouped.values():
                    truth=rr[0][0]['class_name']; residuals=[r[2] for r in rr if r[2] is not None]
                    if not residuals:continue
                    med=st.median(residuals);u=med/band[truth][1 if med>=0 else 0]
                    counts=Counter(r[1] for r in rr if r[1] is not None).most_common()
                    vote=counts[0][0] if counts and (len(counts)==1 or counts[0][1]>counts[1][1]) else None
                    if role=='test' and mode=='depth' and u>=1 and vote==truth:
                        c=sum(r[1]==truth for r in rr);a=sum(r[1] is None for r in rr)
                        counterexamples.append(dict(fold=fold,session=rr[0][0]['session_dir'],U=u,correct=c,wrong=len(rr)-c-a,abstained=a))
    assert len(outside)==1 and outside[0]['role']=='train'
    assert len(counterexamples)==11 and len({r['session'] for r in counterexamples})==8
    result.update(residual_counts=details,pixel_outside=outside,exact_nearest_CAD_ties=exact_ties,U_counterexamples=counterexamples)
    assert all(digest(ROOT/p)==h for p,h in hashes.items())
    save('근거대조.json',result)
    print(json.dumps(dict(native=result['native_hwpx'],errata_artifacts=len(checked),logs=len(logs),pixel_outside=outside,nearest_ties=len(exact_ties),U_cases=len(counterexamples)),ensure_ascii=False,indent=2))

if __name__=='__main__': main()
