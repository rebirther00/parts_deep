"""Submission-wide read-only v9 audit. Writes only new review artifacts."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import re
import subprocess
import zipfile
from lxml import etree
from hwpx.document import HwpxDocument
from build_manuscript_v9_hwpx import parse,plain
from write_v4_review import manuscript_structure

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'reports/논문초안_v9_리뷰_20261110'
HP='http://www.hancom.co.kr/hwpml/2011/paragraph'
HH='http://www.hancom.co.kr/hwpml/2011/head'
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(DEST/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')

def main():
    DEST.mkdir(exist_ok=False)
    names=[f'manuscript/논문초안_v9.{e}' for e in ['hwpx','pdf','md','docx']]
    names+=['manuscript/v8_검토답변.md','manuscript/v8_작성확인사항.md','reports/errata_localization_scope_20261110.json']
    hashes={n:digest(ROOT/n) for n in names};save('원본해시.json',hashes)
    result=dict(original_sha256=hashes)
    m8,m9=[(ROOT/f'manuscript/논문초안_v{v}.md').read_text() for v in (8,9)]
    assert m8.split('## ABSTRACT')[1].split('## 후 기')[0]==m9.split('## ABSTRACT')[1].split('## 후 기')[0]
    assert [l for l in m8.splitlines() if l.startswith('|')]==[l for l in m9.splitlines() if l.startswith('|')]
    result['research_body_abstract_tables_identical_to_v8']=True
    d=HwpxDocument.open(ROOT/names[0]);assert not d.validate().issues and not d.notes.memos
    with zipfile.ZipFile(ROOT/names[0]) as z:
        root=etree.fromstring(z.read('Contents/section0.xml'));header=etree.fromstring(z.read('Contents/header.xml'))
    pp=[dict(id=p.get('id'),text=''.join(p.itertext())) for p in root.iter('{'+HP+'}p')]
    save('HWPX문단.json',pp);(DEST/'HWPX본문추출.txt').write_text('\n\n'.join(p['text'] for p in pp))
    native={p['text'].strip() for p in pp};parsed=parse(m9)
    for b in parsed['blocks']:
        if b[0] in ('p','h1','h2','tabcap'):assert plain(b[1]).strip() in native,b
    assert parsed['abstract'] in native and parsed['ack'] in native
    text='\n'.join(native)
    for s in ['이민수*, 이진호, 신희영, 신영일','Minsu Lee*, Jinho Lee, Heeyoung Shin, Youngil Shin','minsu.lee@koceti.re.kr']:
        assert s in text,s
    old,new=[manuscript_structure(ROOT/f'manuscript/논문초안_v{v}.hwpx') for v in (8,9)]
    assert old['images']==new['images']
    props={p.get('id'):p for p in header.iter('{'+HH+'}paraPr')}
    top=root.findall('{'+HP+'}p');tables=[]
    for i,p in enumerate(top):
        if not ''.join(p.itertext()).startswith('Table '):continue
        t=next(top[i+1].iter('{'+HP+'}tbl'))
        keep=next(props[p.get('paraPrIDRef')].iter('{'+HH+'}breakSetting')).get('keepWithNext')
        assert t.get('pageBreak')=='NONE' and keep=='1'
        tables.append(dict(caption=''.join(p.itertext()),pageBreak=t.get('pageBreak'),keepWithNext=keep))
    assert len(tables)==8
    placeholders=sorted({match for p in native for match in re.findall(r'【[^】]+】',p)})
    result['native_hwpx']=dict(validation_issues=0,paragraphs=len(pp),top_paragraphs=len(top),tables=new['tables'],figures=new['figures'],
        unchanged_images=True,body_tables=tables,body_text_matches_md=True,author_text_matches_md=True,
        abstract_words=len(parsed['abstract'].split()),keywords=parsed['keywords'],placeholders=placeholders)
    # Count table 1 directly from the frozen manifest and inspect split independence.
    sp=ROOT/'data/snapshots/v1';frames=rows(sp/'frames.jsonl');split=read(sp/'splits.json')['partitions']
    cg=defaultdict(set);cf=Counter();hashgroups=defaultdict(set)
    for r in frames:
        cg[r['class_name']].add(r['group_id']);cf[r['class_name']]+=1;hashgroups[r['rgb_sha256']].add(r['group_id'])
    counts={c:dict(doors=len(cg[c]),images=cf[c]) for c in sorted(cf)}
    test_occurrences=Counter();partition_counts={}
    for k,part in split.items():
        tr,va,te=[set(part[role]) for role in ['train','val','test']]
        assert not (tr&va or tr&te or va&te)
        partition_counts[k]=dict(train=len(tr),val=len(va),test=len(te))
        if k!='temporal':test_occurrences.update(te)
    assert len(test_occurrences)==361 and set(test_occurrences.values())=={1}
    assert len(frames)==7104 and len(cf)==8
    result['manifest_and_splits']=dict(per_class=counts,frames=len(frames),physical_door_groups=len(test_occurrences),
        within_split_group_overlap=0,OOF_tests_per_group=1,partitions=partition_counts,
        identical_rgb_hashes_across_groups=sum(len(gg)>1 for gg in hashgroups.values()))
    # Existing independent audits support these unchanged tables; check the current collected values.
    c3=read(ROOT/'reports/extension_v3/collected.json');c6=read(ROOT/'reports/extension_v6/collected.json')
    def compact(v):return {k:{key:value for key,value in row.items() if key in ['session','session_abstained','frame','frame_abstained','macro_f1']} for k,row in v.items() if isinstance(row,dict)}
    assert c3['cv5']['hole8']['session']==[361,361] and c3['cv5']['rgb']['session']==[360,361] and c3['cv5']['rgbe']['session']==[359,361]
    assert not c6['missing_strict']
    result['current_aggregates']=dict(cv5=compact(c3['cv5']),temporal=compact(c3['temporal']),
        extension_v6_sha256=digest(ROOT/'reports/extension_v6/collected.json'),extension_v6_protocol_sha256=c6['protocol_sha256'])
    result['prior_full_audits']=['reports/논문초안_v6_리뷰_20261110/근거대조.json','reports/논문초안_v7_리뷰_20261110/근거대조.json','reports/논문초안_v8_리뷰_20261110/근거대조.json']
    for f in ['manuscript/v8_검토답변.md','manuscript/v8_작성확인사항.md']:
        s=(ROOT/f).read_text();assert '정답 6장' in s and '보류 13장' in s and '정정(2026-11-10' in s
    result['prior_outlier_reply_corrected']=True
    subprocess.run(['pdftotext','-layout',str(ROOT/names[1]),str(DEST/'PDF본문추출.txt')],check=True)
    pages=(DEST/'PDF본문추출.txt').read_text().split('\f')
    result['pdf']=dict(pages=sum(bool(p.strip()) for p in pages),draft_header_pages=sum('manuscript draft v9' in p for p in pages),
        placeholders=sorted(set(re.findall(r'【[^】]+】','\n'.join(pages)))),table_captions_by_page={str(i):re.findall(r'Table\s+([1-8])\s+[A-Z]',p) for i,p in enumerate(pages,1) if re.search(r'Table\s+[1-8]\s+[A-Z]',p)})
    for i,p in enumerate(pages,1):
        if not p.strip():continue
        subprocess.run(['pdftoppm','-f',str(i),'-l',str(i),'-scale-to','1500' if i in [1,8,10,11,12] else '1000','-png','-singlefile',str(ROOT/names[1]),str(DEST/f'원고_p{i:02}')],check=True)
    assert all(digest(ROOT/n)==h for n,h in hashes.items())
    save('근거대조.json',result)
    print(json.dumps(dict(native=result['native_hwpx'],pdf=result['pdf'],manifest=result['manifest_and_splits'],previous_reply_corrected=True),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
