"""Audit author v10 files read-only and save a separate review evidence bundle."""
from pathlib import Path
from collections import Counter, defaultdict
import difflib
import hashlib
import json
import re
import subprocess
import zipfile
from lxml import etree
from hwpx.document import HwpxDocument
from build_manuscript_v10_hwpx import parse, plain
from write_v4_review import manuscript_structure

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'reports/논문초안_v10_리뷰_20261110'
HP = 'http://www.hancom.co.kr/hwpml/2011/paragraph'
HH = 'http://www.hancom.co.kr/hwpml/2011/head'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    (DEST / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def read(path):
    return json.loads(path.read_text())


def main():
    DEST.mkdir(exist_ok=False)
    names = [f'manuscript/논문초안_v10.{ext}' for ext in ['hwpx', 'pdf', 'md', 'docx']]
    names += ['manuscript/v10_작성확인사항.md', 'figures/v10c_fig1_method.png', 'figures/v10c_fig1_method.pdf']
    names += ['scripts/build_manuscript_v10_hwpx.py', 'scripts/build_manuscript_v10_docx.py', 'scripts/make_figure1_v10col.py']
    hashes = {n: digest(ROOT / n) for n in names}
    save('원본해시.json', hashes)
    result = dict(original_sha256=hashes)
    old, new = [(ROOT / f'manuscript/논문초안_v{v}.md').read_text() for v in [9, 10]]
    (DEST / 'v9_v10差分.txt').write_text(''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile='v9', tofile='v10')))
    expected = old.replace('【우편번호】', '54004').replace(
        '현장 영상으로 학습하면 높은 정확도를 얻을 수 있다[2]',
        '외관이 유사한 산업 부품의 식별에는 2차원 영상과 3차원 점군을 이용한 방법이 비교 연구되었다[2]'
    ).replace('../figures/v3_fig1_method.png', '../figures/v10c_fig1_method.png')
    assert expected == new
    parsed = parse(new)
    result['md_changes'] = dict(only_expected_three_edits=True, abstract_unchanged=parse(old)['abstract'] == parsed['abstract'],
        tables_unchanged=[b for b in parse(old)['blocks'] if b[0] == 'table'] == [b for b in parsed['blocks'] if b[0] == 'table'],
        references_unchanged=parse(old)['refs'] == parsed['refs'])
    doc = HwpxDocument.open(ROOT / names[0])
    assert not doc.validate().issues and not doc.notes.memos
    with zipfile.ZipFile(ROOT / names[0]) as z:
        root = etree.fromstring(z.read('Contents/section0.xml'))
        header = etree.fromstring(z.read('Contents/header.xml'))
    paragraphs = [dict(id=p.get('id'), text=''.join(p.itertext())) for p in root.iter('{' + HP + '}p')]
    save('HWPX문단.json', paragraphs)
    (DEST / 'HWPX본문추출.txt').write_text('\n\n'.join(p['text'] for p in paragraphs))
    native = {p['text'].strip() for p in paragraphs}
    for block in parsed['blocks']:
        if block[0] in ['p', 'h1', 'h2', 'tabcap', 'fig']:
            assert plain(block[1]).strip() in native, block[:2]
        if block[0] == 'table':
            for row in block[1]:
                for cell in row:
                    if cell.strip():
                        assert plain(cell).strip() in native, cell
    assert parsed['abstract'] in native and parsed['ack'] in native
    props = {p.get('id'): p for p in header.iter('{' + HH + '}paraPr')}
    top = root.findall('{' + HP + '}p')
    tables = []
    for i, p in enumerate(top):
        if not ''.join(p.itertext()).startswith('Table '):
            continue
        table = next(top[i+1].iter('{' + HP + '}tbl'))
        keep = next(props[p.get('paraPrIDRef')].iter('{' + HH + '}breakSetting')).get('keepWithNext')
        assert table.get('pageBreak') == 'NONE' and keep == '1'
        tables.append(dict(caption=''.join(p.itertext()), pageBreak=table.get('pageBreak'), keepWithNext=keep))
    assert len(tables) == 8
    structure = manuscript_structure(ROOT / names[0])
    old_structure = manuscript_structure(ROOT / 'manuscript/논문초안_v9.hwpx')
    changed_images = [n for n, v in structure['images'].items() if old_structure['images'].get(n) != v]
    assert changed_images == ['BinData/BIN0001.png']
    embedded = {}
    for ext, prefix in [('hwpx', 'BinData/'), ('docx', 'word/media/')]:
        with zipfile.ZipFile(ROOT / f'manuscript/논문초안_v10.{ext}') as z:
            embedded[ext] = [n for n in z.namelist() if n.startswith(prefix) and hashlib.sha256(z.read(n)).hexdigest() == hashes['figures/v10c_fig1_method.png']]
        assert len(embedded[ext]) == 1
    placeholders = sorted({m for t in native for m in re.findall(r'【[^】]+】', t)})
    assert not placeholders
    assert any('Jeonbuk State 54004, Republic of Korea' in t for t in native)
    result['native_hwpx'] = dict(validation_issues=0, paragraphs=len(paragraphs), tables=structure['tables'], figures=structure['figures'],
        body_text_and_table_cells_match_md=True, body_tables=tables, abstract_words=len(parsed['abstract'].split()),
        placeholders=placeholders, figure1_embedded_matches=embedded, only_figure1_changed=True)
    # Independent recount of the frozen manifest and split membership.
    sp = ROOT / 'data/snapshots/v1'
    rows = [json.loads(line) for line in (sp / 'frames.jsonl').read_text().splitlines() if line.strip()]
    splits = read(sp / 'splits.json')['partitions']
    groups, file_groups = defaultdict(set), defaultdict(set)
    count = Counter()
    for row in rows:
        groups[row['class_name']].add(row['group_id'])
        count[row['class_name']] += 1
        file_groups[row['rgb_sha256']].add(row['group_id'])
    occurrence, partition_counts = Counter(), {}
    for name, split in splits.items():
        train, val, test = [set(split[k]) for k in ['train', 'val', 'test']]
        assert not (train & val or train & test or val & test)
        partition_counts[name] = dict(train=len(train), val=len(val), test=len(test))
        if name != 'temporal':
            occurrence.update(test)
    assert len(rows) == 7104 and len(groups) == 8 and len(occurrence) == 361 and set(occurrence.values()) == {1}
    result['manifest_and_splits'] = dict(frames=len(rows), groups=len(occurrence), classes=len(groups),
        counts={c: dict(doors=len(groups[c]), images=count[c]) for c in sorted(count)},
        within_split_overlap=0, OOF_tests_per_group=1, partitions=partition_counts,
        identical_rgb_hashes_across_groups=sum(len(g) > 1 for g in file_groups.values()))
    collected = read(ROOT / 'reports/extension_v3/collected.json')
    extension = read(ROOT / 'reports/extension_v6/collected.json')
    previous = read(ROOT / 'reports/논문초안_v9_리뷰_20261110/근거대조.json')
    assert not extension['missing_strict']
    assert digest(ROOT / 'reports/extension_v6/collected.json') == previous['current_aggregates']['extension_v6_sha256']
    for method, value in [('hole8', [361, 361]), ('rgb', [360, 361]), ('rgbe', [359, 361])]:
        assert collected['cv5'][method]['session'] == value
    for method, value in [('hole', [133, 133]), ('rgb', [132, 133]), ('rgbe', [132, 133]), ('hole_depth', [108, 133])]:
        assert collected['temporal'][method]['session'] == value
    result['current_aggregates'] = {k: collected[k] for k in ['cv5', 'temporal']}
    result['extension_v6_unchanged_since_independent_audits'] = True
    result['raw_predictions_reaggregated_this_turn'] = False
    result['previous_audit'] = 'reports/논문초안_v9_리뷰_20261110/근거대조.json'
    pdf = ROOT / 'manuscript/논문초안_v10.pdf'
    subprocess.run(['pdftotext', '-layout', str(pdf), str(DEST / 'PDF본문추출.txt')], check=True)
    pages = [p for p in (DEST / 'PDF본문추출.txt').read_text().split('\f') if p.strip()]
    assert len(pages) == 14 and all('manuscript draft' not in p and '【' not in p for p in pages)
    result['pdf'] = dict(pages=len(pages), draft_header_count=0, placeholders=[],
        table_captions_by_page={i: re.findall(r'Table\s+([1-8])\s+[A-Z]', p) for i, p in enumerate(pages, 1) if re.search(r'Table\s+[1-8]\s+[A-Z]', p)})
    for i in range(1, 15):
        subprocess.run(['pdftoppm', '-f', str(i), '-l', str(i), '-scale-to', '1500' if i in [1, 3, 7, 8, 10, 11, 12] else '1100',
                        '-png', '-singlefile', str(pdf), str(DEST / f'원고_p{i:02}')], check=True)
    assert all(digest(ROOT / n) == value for n, value in hashes.items())
    save('근거대조.json', result)
    print(json.dumps(dict(native=result['native_hwpx'], pdf=result['pdf'], manifest=result['manifest_and_splits'], original_files_unchanged=len(hashes)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
