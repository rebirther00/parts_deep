"""Build manuscript/논문초안_v3.hwpx on the KSMTE template (한국생산제조학회_새 논문 양식(한글).hwpx).

Run with the system python3 (python-hwpx installed there):  python3 scripts/build_manuscript_v3_hwpx.py

1) Template is copied and stripped: the front-matter table keeps rows 0-25 (header, titles, authors,
   ARTICLE INFO, ABSTRACT, keywords); the sample body tables and the reference-format guide are removed.
2) Body is appended in two columns (NEWSPAPER, gap 1920) with the template's own paraPr/charPr ids:
   section heading pPr46/cPr63, sub-heading pPr47/cPr52, body pPr45/cPr51, citation superscript cPr59,
   captions pPr48|pPr5/cPr68, references heading pPr50/cPr77, references pPr27/cPr46.
3) Figures use the column-width versions (figures/v3c_*), English captions; tables are English.
"""
import copy
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from lxml import etree
from hwpx.document import HwpxDocument

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / 'manuscript/한국생산제조학회_새 논문 양식(한글).hwpx'
SRC = ROOT / 'manuscript/논문초안_v3.md'
DST = ROOT / 'manuscript/논문초안_v3.hwpx'
SKILL = Path('/home/koceti/.claude/skills/synced/7e7f0965-91de-47cb-9ed6-c9c6c8b5fa28_c454563a-476d-4188-a8e9-727b7bd3e549/hwpx/scripts')
NS = {'hp': 'http://www.hancom.co.kr/hwpml/2011/paragraph'}
HP = '{%s}' % NS['hp']
FRONT_TABLE_ID, SAMPLE_BODY_ID = '1209247388', '1209798889'
KEEP_ROWS = 26
COL_GAP = 1920
COL_WIDTH = (59528 - 2 * 4819 - COL_GAP) // 2          # 23985 HWPUNIT = 84.6 mm
S = dict(h1=('46', '63'), h2=('47', '52'), body=('45', '51'), cite='59', fig=('48', '68'), tab=('5', '68'),
         refh=('50', '77'), ref=('27', '46'))


def stripped_template(workdir):
    unz = workdir / 'tpl'
    with zipfile.ZipFile(TEMPLATE) as z:
        z.extractall(unz)
    sec = unz / 'Contents/section0.xml'
    tree = etree.parse(str(sec)); root = tree.getroot()
    ps = root.findall('hp:p', NS)
    for t in ps[0].iter(HP + 'tbl'):
        if t.get('id') == SAMPLE_BODY_ID:
            t.getparent().remove(t)
            break
    front = next(t for t in ps[0].iter(HP + 'tbl') if t.get('id') == FRONT_TABLE_ID)
    rows = front.findall('hp:tr', NS)
    removed_h = 0
    for tr in rows[KEEP_ROWS:]:
        removed_h += max(int(tc.find('hp:cellSz', NS).get('height')) for tc in tr.findall('hp:tc', NS))
        front.remove(tr)
    front.set('rowCnt', str(KEEP_ROWS))
    sz = front.find('hp:sz', NS)
    if sz is not None:
        sz.set('height', str(max(int(sz.get('height')) - removed_h, 1000)))
    for p in ps[1:]:
        root.remove(p)
    tree.write(str(sec), xml_declaration=True, encoding='UTF-8', standalone=True)
    base = workdir / 'base.hwpx'
    subprocess.run([sys.executable, str(SKILL / 'pack.py'), str(unz), str(base)], check=True, capture_output=True)
    return base


# ---------------------------------------------------------------- markdown parsing
def parse(md):
    lines = md.splitlines()
    doc = dict(title_ko='', title_en='', abstract='', keywords=[], blocks=[], refs=[], ack='')
    i, section = 0, None
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('# ') and not doc['title_ko']:
            doc['title_ko'] = line[2:]; doc['title_en'] = lines[i + 2].strip(); i += 3; continue
        if line.startswith('## '):
            section = line[3:].strip()
            if section not in ('ABSTRACT', 'References', '후 기', '후기'):
                doc['blocks'].append(('h1', section))
            i += 1; continue
        if not line:
            i += 1; continue
        if section == 'ABSTRACT':
            if line.startswith('Key Words:'):
                doc['keywords'] = [k.strip() for k in line.split(':', 1)[1].split(',')]
            else:
                doc['abstract'] = line
        elif section == 'References':
            doc['refs'].append(re.sub(r'^\d+\.\s*', '', line))
        elif section in ('후 기', '후기'):
            doc['ack'] = line
        elif section is None:
            pass
        elif line.startswith('### '):
            doc['blocks'].append(('h2', line[4:]))
        elif line.startswith('!['):
            m = re.fullmatch(r'!\[(.+)\]\((.+)\)', line); doc['blocks'].append(('fig', m.group(1), m.group(2)))
        elif line.startswith('Table '):
            doc['blocks'].append(('tabcap', line))
        elif line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', c) for c in cells):
                    rows.append(cells)
                i += 1
            doc['blocks'].append(('table', rows)); continue
        else:
            doc['blocks'].append(('p', line))
        i += 1
    return doc


TOKEN = re.compile(r'(\*\*[^*]+\*\*|\^[^^]+\^|~[^~]+~|\[\d+(?:[,\-–]\d+)*\])')


def plain(t):
    return re.sub(r'\*\*|\^|~', '', t)


def add_rich(doc, text, ppr, cpr, styles):
    p = doc.add_paragraph('', para_pr_id_ref=ppr, char_pr_id_ref=cpr, include_run=False)
    for part in TOKEN.split(text):
        if not part:
            continue
        if part.startswith('**'):
            p.add_run(part[2:-2], char_pr_id_ref=styles['bold'])
        elif part.startswith('^'):
            p.add_run(part[1:-1], char_pr_id_ref=styles['sup'])
        elif part.startswith('~'):
            p.add_run(part[1:-1], char_pr_id_ref=styles['sub'])
        elif re.fullmatch(r'\[\d+(?:[,\-–]\d+)*\]', part):
            p.add_run(part, char_pr_id_ref=S['cite'])
        else:
            p.add_run(part, char_pr_id_ref=cpr)
    return p


def build():
    md = SRC.read_text(encoding='utf8'); d = parse(md)
    with tempfile.TemporaryDirectory() as tmp:
        base = stripped_template(Path(tmp))
        doc = HwpxDocument.open(str(base))
        front = next(t for t in doc.paragraphs[0].tables if t.row_count == KEEP_ROWS)
        front.set_cell_text(2, 0, d['title_ko'])
        front.set_cell_text(6, 0, d['title_en'])
        for r in (4, 8, 10):
            front.set_cell_text(r, 0, '(최종 파일 업로드 시 작성)')
        front.set_cell_text(15, 7, d['abstract'])
        front.set_cell_text(22, 0, '\n'.join(d['keywords']), split_paragraphs=True)
        styles = dict(bold=doc.ensure_run_style(bold=True, base_char_pr_id=S['body'][1]),
                      sup=doc.ensure_run_style(script='sup', base_char_pr_id=S['body'][1]),
                      sub=doc.ensure_run_style(script='sub', base_char_pr_id=S['body'][1]))
        cell_cpr = doc.ensure_run_style(size=8, base_char_pr_id=S['ref'][1])
        cell_bold = doc.ensure_run_style(size=8, bold=True, base_char_pr_id=S['ref'][1])
        lines_fill = doc.ensure_border_fill(border_color='#000000', border_width='0.12 mm', active_borders=['top', 'bottom'])
        head_fill = doc.ensure_border_fill(border_color='#000000', border_width='0.4 mm', active_borders=['top', 'bottom'],
                                           border_type='DOUBLE_SLIM')
        first = doc.add_paragraph('', para_pr_id_ref=S['body'][0], char_pr_id_ref=S['body'][1])
        doc.set_columns(2, same_gap=COL_GAP, paragraph=first)
        caption_next = None
        for b in d['blocks']:
            kind = b[0]
            if kind == 'h1':
                doc.add_paragraph(b[1], para_pr_id_ref=S['h1'][0], char_pr_id_ref=S['h1'][1])
            elif kind == 'h2':
                doc.add_paragraph(b[1], para_pr_id_ref=S['h2'][0], char_pr_id_ref=S['h2'][1])
            elif kind == 'p':
                add_rich(doc, b[1], S['body'][0], S['body'][1], styles)
            elif kind == 'fig':
                path = (SRC.parent / b[2].replace('/v3_fig', '/v3c_fig')).resolve()
                doc.add_picture(path.read_bytes(), 'png', width_mm=80, align='center', para_pr_id_ref=S['fig'][0])
                cap = plain(b[1])
                doc.add_paragraph(cap, para_pr_id_ref=S['fig'][0], char_pr_id_ref=S['fig'][1])
            elif kind == 'tabcap':
                caption_next = plain(b[1])
            elif kind == 'table':
                if caption_next:
                    doc.add_paragraph(caption_next, para_pr_id_ref=S['tab'][0], char_pr_id_ref=S['tab'][1]); caption_next = None
                rows = b[1]
                t = doc.add_table(len(rows), len(rows[0]), width=COL_WIDTH - 400, border_fill_id_ref=lines_fill,
                                  char_pr_id_ref=cell_cpr, para_pr_id_ref=S['tab'][0])
                for ri, cells in enumerate(rows):
                    for ci, v in enumerate(cells):
                        t.set_cell_text(ri, ci, plain(v))
                        cell = t.cell(ri, ci)
                        for par in cell.paragraphs:
                            for run in par.runs:
                                run.char_pr_id_ref = cell_bold if ri == 0 else cell_cpr
                        if ri == 0:
                            t.set_cell_border_fill(ri, ci, head_fill)
        if d['ack']:
            doc.add_paragraph('후 기', para_pr_id_ref=S['h1'][0], char_pr_id_ref=S['h1'][1])
            add_rich(doc, d['ack'], S['body'][0], S['body'][1], styles)
        refh = doc.add_paragraph('References', para_pr_id_ref=S['refh'][0], char_pr_id_ref=S['refh'][1])
        for i, r in enumerate(d['refs'], 1):
            doc.add_paragraph(f'[{i}] {r}', para_pr_id_ref=S['ref'][0], char_pr_id_ref=S['ref'][1])
        doc.save_to_path(str(DST))
    print(DST)


if __name__ == '__main__':
    build()
