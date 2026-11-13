"""Audit the v8 revisions without changing originals or rerunning experiments."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import re
import subprocess
import zipfile
from lxml import etree
from hwpx.document import HwpxDocument
from build_manuscript_v8_hwpx import parse, plain
from write_v4_review import manuscript_structure

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'reports/논문초안_v8_리뷰_20261110'
HP = 'http://www.hancom.co.kr/hwpml/2011/paragraph'
HH = 'http://www.hancom.co.kr/hwpml/2011/head'

def read(p): return json.loads(p.read_text())
def rows(p): return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name, value): (DEST/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

def main():
    if (DEST/'근거대조.json').exists(): raise FileExistsError(DEST/'근거대조.json')
    DEST.mkdir(exist_ok=True)
    names = [f'manuscript/논문초안_v8.{e}' for e in ['hwpx','pdf','md','docx']]
    names += [f'manuscript/v8_{s}.md' for s in ['검토답변','작성확인사항','차별성_검토']]
    names += ['reports/errata_localization_scope_20261110.json']
    hashes = {n:digest(ROOT/n) for n in names}
    if (DEST/'원본해시.json').exists():
        assert read(DEST/'원본해시.json')==hashes
    else: save('원본해시.json',hashes)
    old = read(ROOT/'reports/논문초안_v7_리뷰_20261110/원본해시.json')
    assert all(digest(ROOT/n)==h for n,h in old.items())
    v7,v8 = [(ROOT/f'manuscript/논문초안_v{v}.md').read_text() for v in (7,8)]
    assert [s for s in v7.splitlines() if s.startswith('|')] == [s for s in v8.splitlines() if s.startswith('|')]
    result = dict(original_sha256=hashes, v7_originals_unchanged=True, all_table_rows_unchanged=True)
    doc = HwpxDocument.open(ROOT/'manuscript/논문초안_v8.hwpx')
    assert not doc.validate().issues
    with zipfile.ZipFile(ROOT/'manuscript/논문초안_v8.hwpx') as z:
        root = etree.fromstring(z.read('Contents/section0.xml'))
        header = etree.fromstring(z.read('Contents/header.xml'))
    paragraphs = [dict(id=p.get('id'),text=''.join(p.itertext())) for p in root.iter('{'+HP+'}p')]
    save('HWPX문단.json',paragraphs)
    (DEST/'HWPX본문추출.txt').write_text('\n\n'.join(p['text'] for p in paragraphs))
    texts = {p['text'].strip() for p in paragraphs}
    parsed = parse(v8)
    missing = [b for b in parsed['blocks'] if b[0] in ('p','h1','h2','tabcap') and plain(b[1]).strip() not in texts]
    assert not missing, missing
    assert parsed['abstract'] in texts
    before,after = [manuscript_structure(ROOT/f'manuscript/논문초안_v{v}.hwpx') for v in (7,8)]
    assert before['images']==after['images']
    p7,p8=before['paragraphs'],after['paragraphs'];assert len(p7)==len(p8)
    native_changes=[dict(before=a[1],after=b[1]) for a,b in zip(p7,p8) if a[1]!=b[1]]
    assert len(native_changes)==3
    body_tables=[]
    capprops={p.get('id'):p for p in header.iter('{'+HH+'}paraPr')}
    top=root.findall('{'+HP+'}p')
    for index,p in enumerate(top):
        if not ''.join(p.itertext()).startswith('Table '):continue
        tables=list(top[index+1].iter('{'+HP+'}tbl'));assert len(tables)==1
        t=tables[0];prop=capprops[p.get('paraPrIDRef')]
        keep=next(prop.iter('{'+HH+'}breakSetting')).get('keepWithNext')
        assert t.get('pageBreak')=='NONE' and keep=='1'
        body_tables.append(dict(caption=''.join(p.itertext()),pageBreak=t.get('pageBreak'),keepWithNext=keep,
                                caption_paraPr=p.get('paraPrIDRef'),rows=t.get('rowCnt')))
    assert len(body_tables)==8
    result['native_hwpx']=dict(validation_issues=0,body_text_mismatches=missing,changed_paragraphs=native_changes,
        top_paragraphs=len(top),all_paragraphs=len(p8),tables=after['tables'],figures=after['figures'],
        abstract_words=len(parsed['abstract'].split()),unchanged_embedded_images=True,body_tables=body_tables)
    # Check the actual DOCX property, not only the builder source.
    W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
    with zipfile.ZipFile(ROOT/'manuscript/논문초안_v8.docx') as z:
        dr=etree.fromstring(z.read('word/document.xml'))
    dtables=[t for t in dr.findall('.//'+W+'tbl') if t.getprevious() is not None
             and ''.join(t.getprevious().itertext()).startswith('Table ')]
    assert len(dtables)==8
    for t in dtables:
        for row in t.findall(W+'tr')[:-1]:
            for p in row.findall('.//'+W+'p'):
                keep=p.find(W+'pPr/'+W+'keepNext');assert keep is not None and keep.get(W+'val','1')!='0'
    result['docx_tables_keep_next_verified']=8
    # Verify amendment against current files and the independent v7 audit hashes.
    err=read(ROOT/'reports/errata_localization_scope_20261110.json')
    previous=read(ROOT/'reports/논문초안_v7_리뷰_20261110/근거대조.json')
    previous_logs={r['file']:r['sha256'] for r in previous['scope_errata']['log_verification']}
    path_count=0
    for a in err['affected_artifacts']:
        assert digest(ROOT/a['file'])==a['sha256']
        original=read(ROOT/a['file'])
        for ptr in a['json_paths']:
            node=original
            for key in ptr.strip('/').split('/'):node=node[key]
            assert node==err['wrong_string'];path_count+=1
    for a in err['affected_logs']:
        assert digest(ROOT/a['file'])==a['sha256']==previous_logs[a['file']]
        assert err['wrong_string'] in (ROOT/a['file']).read_text()
    assert err['amendments']
    result['errata']=dict(artifacts_verified=len(err['affected_artifacts']),json_paths_verified=path_count,
        logs_verified=len(err['affected_logs']),all_logs_equal_v7_audit=True,amendment_record_present=True)
    # Check the newly discussed outlier using stored features, with no model execution.
    frames=rows(ROOT/'data/snapshots/v1/frames.jsonl')
    group=[r for r in frames if r['session_dir']=='20260831/E38_door_LH_FRT/s_183313']
    parts=read(ROOT/'data/snapshots/v1/splits.json')['partitions']['4']
    assert len(group)==20 and all(r['group_id'] in parts['train'] for r in group)
    pool={r['frame_id']:r for r in rows(ROOT/'features/holes_field8_f4_s42_v3_trainTest/rows.jsonl')}
    cal=read(ROOT/'reports/extension_v5/margin.json')['splits']['4:pixel']['scale']
    cfg=read(ROOT/'data/snapshots/v1/protocol.json')
    out=[]
    for r in group:
        f=pool[r['frame_id']];raw=f.get('span_normalized')
        raw_d=raw*cal[str(r['intrinsics']['serial'])] if raw is not None else None
        d=raw_d if f['gate']=='ok' else None
        ci=min(range(len(cfg['classes'])),key=lambda i:abs(cfg['cad_hole_distance_mm'][i]-d)) if d is not None else None
        pred=cfg['classes'][ci] if ci is not None and abs(cfg['cad_hole_distance_mm'][ci]-d)<=30 else None
        out.append(dict(frame_id=r['frame_id'],distance_mm=d,raw_distance_before_gate_mm=raw_d,gate=f['gate'],pred=pred,
                        hinge=f['points']['hinge'],latch=f['points']['latch']))
    bad=next(r for r in out if r['frame_id']=='58:rgb_0140.png')
    good=[r for r in out if r['pred']=='E38_door_LH_FRT']
    assert len(good)==6 and all(round(r['distance_mm'],1)==812.7 for r in good)
    assert round(bad['distance_mm'],1)==764.1 and round(bad['hinge'][0][2],3)==.155
    result['new_outlier_explanation']=dict(session=group[0]['session_dir'],role='train',fold='4',
        session_votes=dict(Counter(r['pred'] or 'abstained' for r in out)),gate_counts=dict(Counter(r['gate'] for r in out)),frames=out,
        author_claim_all_other_19_measured_812_7=False,
        note='Stored peak scores and measurements checked; no manual ground-truth point newly collected.')
    subprocess.run(['pdftotext','-layout',str(ROOT/'manuscript/논문초안_v8.pdf'),str(DEST/'PDF본문추출.txt')],check=True)
    pages=(DEST/'PDF본문추출.txt').read_text().split('\f')
    result['pdf']=dict(pages=sum(bool(p.strip()) for p in pages),table_captions_by_page={})
    for i,p in enumerate(pages,1):
        captions=re.findall(r'Table\s+([1-8])\s+[A-Z]',p)
        if captions:result['pdf']['table_captions_by_page'][str(i)]=captions
    assert result['pdf']['pages']==14
    assert all(digest(ROOT/n)==h for n,h in hashes.items())
    save('근거대조.json',result)
    print(json.dumps(dict(native_changed_paragraphs=len(native_changes),native_body_tables_unsplit=len(body_tables),
        errata=result['errata'],pdf=result['pdf'],outlier=bad,session_votes=result['new_outlier_explanation']['session_votes']),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
