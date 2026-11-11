"""Build manuscript/논문초안_v5.docx (KSMTE-like two-column layout) and 논문초안_v5.pdf (LibreOffice) from 논문초안_v5.md.

Approximates the journal template (A4, side margins 17 mm, two columns with a 6.8 mm gap, body 10 pt with 16 pt line pitch,
first-line indent 2.8 mm, section headings 12 pt bold centred, captions 9 pt bold, three-line tables, superscript [n]).
The authoritative journal file is the HWPX (scripts/build_manuscript_v5_hwpx.py); this DOCX/PDF is for reading and page count.

Inline markup: **bold**, ^superscript^, _{subscript}, [n] citation (superscript). '~' is a literal range sign.
Run: ../../venv/parts_deep/bin/python scripts/build_manuscript_v5_docx.py
"""
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from door8.core import ROOT

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Mm, Pt

SRC = ROOT / 'manuscript/논문초안_v5.md'
DST = ROOT / 'manuscript/논문초안_v5.docx'
FONT = 'Noto Serif CJK KR'
TOKEN = re.compile(r'(\*\*[^*]+\*\*|\^[^^]+\^|_\{[^}]+\}|\[\d+(?:[,\-–]\d+)*\])')
COL_W = Mm(84.6)


def runs(par, text, size=None, bold=False, cite_sup=True):
    for part in TOKEN.split(text):
        if not part:
            continue
        if part.startswith('**'):
            r = par.add_run(part[2:-2]); r.bold = True
        elif part.startswith('^'):
            r = par.add_run(part[1:-1]); r.font.superscript = True; r.bold = bold
        elif part.startswith('_{'):
            r = par.add_run(part[2:-1]); r.font.subscript = True; r.bold = bold
        elif re.fullmatch(r'\[\d+(?:[,\-–]\d+)*\]', part):
            r = par.add_run(part); r.font.superscript = cite_sup; r.bold = bold
        else:
            r = par.add_run(part); r.bold = bold
        if size:
            r.font.size = Pt(size)


def fmt(p, size=10, align=WD_ALIGN_PARAGRAPH.JUSTIFY, indent=0.0, before=0, after=0, exact=None, keep=False):
    pf = p.paragraph_format
    p.alignment = align; pf.first_line_indent = Mm(indent) if indent else None
    pf.space_before, pf.space_after = Pt(before), Pt(after)
    if exact:
        pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY; pf.line_spacing = Pt(exact)
    else:
        pf.line_spacing = 1.0
    pf.keep_with_next = keep
    pf.widow_control = True
    for r in p.runs:
        if r.font.size is None:
            r.font.size = Pt(size)
    return p


def set_fonts(doc):
    st = doc.styles['Normal']
    st.font.name = FONT; st.font.size = Pt(10)
    rf = st._element.get_or_add_rPr().get_or_add_rFonts()
    for k in ('w:ascii', 'w:hAnsi', 'w:eastAsia', 'w:cs'):
        rf.set(qn(k), FONT)


def columns(section, n, space_mm=6.8):
    cols = section._sectPr.find(qn('w:cols'))
    if cols is None:
        cols = OxmlElement('w:cols'); section._sectPr.append(cols)
    cols.set(qn('w:num'), str(n)); cols.set(qn('w:space'), str(int(space_mm / 25.4 * 1440)))


def border(el, side, val='single', sz=4):
    b = OxmlElement(f'w:{side}'); b.set(qn('w:val'), val); b.set(qn('w:sz'), str(sz)); b.set(qn('w:space'), '0')
    b.set(qn('w:color'), '000000'); el.append(b)


def column_widths(rows, total_mm=84.0, body_mm=1.45, head_mm=1.6, pad_mm=2.2):
    """Each column first gets its longest unbreakable token (body numbers or header word) so nothing breaks
    mid-token at 7.5 pt; the remaining width goes to columns with long cells in proportion to their extra length."""
    plain = lambda v: re.sub(r'_\{([^}]*)\}', r'\1', re.sub(r'\*\*|\^', '', v))
    n = len(rows[0]); need, extra = [], []
    for j in range(n):
        tok = max((len(t) for r in rows[1:] for t in plain(r[j]).split()), default=1)
        head = max(len(t) for t in plain(rows[0][j]).split())
        need.append(max(tok * body_mm, head * head_mm) + pad_mm)
        extra.append(max(max((len(plain(r[j])) for r in rows[1:]), default=0) * body_mm + pad_mm - need[-1], 0))
    left = total_mm - sum(need)
    if left <= 0:
        return [Mm(total_mm * x / sum(need)) for x in need]
    tot = sum(extra) or 1.0
    return [Mm(a + left * e / tot) if sum(extra) else Mm(a + left / n) for a, e in zip(need, extra)]


def three_line_table(doc, rows):
    n = len(rows[0])
    t = doc.add_table(rows=0, cols=n); t.alignment = WD_TABLE_ALIGNMENT.CENTER; t.autofit = True
    tblPr = t._tbl.tblPr
    bd = OxmlElement('w:tblBorders')
    border(bd, 'top', 'double', 4); border(bd, 'bottom', 'single', 6)
    for side in ('left', 'right', 'insideH', 'insideV'):
        border(bd, side, 'nil', 0)
    tblPr.append(bd)
    w = OxmlElement('w:tblW'); w.set(qn('w:w'), str(int(COL_W.twips))); w.set(qn('w:type'), 'dxa'); tblPr.append(w)
    mar = OxmlElement('w:tblCellMar')
    for side in ('left', 'right'):
        m = OxmlElement(f'w:{side}'); m.set(qn('w:w'), '40'); m.set(qn('w:type'), 'dxa'); mar.append(m)
    tblPr.append(mar)
    widths = column_widths(rows)
    lay = OxmlElement('w:tblLayout'); lay.set(qn('w:type'), 'fixed'); tblPr.append(lay)
    grid = t._tbl.tblGrid
    for gc, wd in zip(grid.findall(qn('w:gridCol')), widths):
        gc.set(qn('w:w'), str(int(wd.twips)))
    for i, cells in enumerate(rows):
        row = t.add_row()
        for cell, wd in zip(row.cells, widths):
            cell.width = wd
        trPr = row._tr.get_or_add_trPr(); cant = OxmlElement('w:cantSplit'); trPr.append(cant)
        if i == 0:
            trPr.append(OxmlElement('w:tblHeader'))
        for j, (cell, value) in enumerate(zip(row.cells, cells)):
            cell.text = ''
            p = cell.paragraphs[0]
            runs(p, value, size=7.5, bold=(i == 0))
            fmt(p, size=7.5, align=WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER, before=1, after=1)
            if i == 0:
                tcPr = cell._tc.get_or_add_tcPr(); b = OxmlElement('w:tcBorders'); border(b, 'bottom', 'single', 4); tcPr.append(b)
    return t


def rule(doc):
    p = doc.add_paragraph(); fmt(p, size=4, after=2)
    pPr = p._p.get_or_add_pPr(); bd = OxmlElement('w:pBdr'); border(bd, 'bottom', 'single', 6); pPr.append(bd)


def parse(md):
    lines = md.splitlines(); d = dict(title_ko='', title_en='', abstract='', keywords='', blocks=[], refs=[], ack='')
    i, section = 0, None
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('# ') and not d['title_ko']:
            d['title_ko'] = line[2:]; d['title_en'] = lines[i + 2].strip(); i += 3; continue
        if line.startswith('## '):
            section = line[3:].strip()
            if section not in ('ABSTRACT', 'References', '후 기'):
                d['blocks'].append(('h1', section))
            i += 1; continue
        if not line or section is None:
            i += 1; continue
        if section == 'ABSTRACT':
            if line.startswith('Key Words:'):
                d['keywords'] = line.split(':', 1)[1].strip()
            else:
                d['abstract'] = line
        elif section == 'References':
            cells = [c.strip() for c in line.strip('|').split('|')]
            if line.startswith('|') and cells[0].startswith('['):
                d['refs'].append(cells)            # [n], authors, year, title, source
        elif section == '후 기':
            d['ack'] = line
        elif line.startswith('### '):
            d['blocks'].append(('h2', line[4:]))
        elif line.startswith('!['):
            m = re.fullmatch(r'!\[(.+)\]\((.+)\)', line); d['blocks'].append(('fig', m.group(1), m.group(2)))
        elif line.startswith('Table '):
            d['blocks'].append(('tabcap', line))
        elif line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', c) for c in cells):
                    rows.append(cells)
                i += 1
            d['blocks'].append(('table', rows)); continue
        else:
            d['blocks'].append(('p', line))
        i += 1
    return d


def reference_form(doc, refs):
    """Journal reference form (양식 1): per entry, row 1 = [n] | authors | year | title, row 2 = source across all cells.
    Column widths follow the template table (1900 : 17020 : 5498 : 24902) with a wider number column; the journal hides borders."""
    widths = [Mm(10), Mm(60), Mm(16), Mm(90)]           # template ratio, number column widened so '[10]' fits
    t = doc.add_table(rows=0, cols=4); t.alignment = WD_TABLE_ALIGNMENT.CENTER
    tblPr = t._tbl.tblPr
    lay = OxmlElement('w:tblLayout'); lay.set(qn('w:type'), 'fixed'); tblPr.append(lay)
    for gc, wd in zip(t._tbl.tblGrid.findall(qn('w:gridCol')), widths):
        gc.set(qn('w:w'), str(int(wd.twips)))
    for num, authors, year, title, source in refs:
        r1 = t.add_row(); r2 = t.add_row()
        for cell, wd, text in zip(r1.cells, widths, (num, authors, year, title)):
            cell.width = wd; cell.text = ''
            p = cell.paragraphs[0]; runs(p, text, size=9, cite_sup=False); fmt(p, 9, WD_ALIGN_PARAGRAPH.LEFT, before=1)
        m = r2.cells[0].merge(r2.cells[3]); m.text = ''
        p = m.paragraphs[0]; runs(p, source, size=9); fmt(p, 9, WD_ALIGN_PARAGRAPH.LEFT, after=4)
        p.paragraph_format.left_indent = widths[0]
        for row in (r1, r2):
            row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))


def caption(doc, text):
    m = re.match(r'((?:Fig|Table)\.? ?\d+)\s*(.*)', text)
    p = doc.add_paragraph()
    r = p.add_run(m.group(1) + ' '); r.bold = True; r.font.size = Pt(9)
    runs(p, m.group(2), size=9)
    return p


def build():
    d = parse(SRC.read_text(encoding='utf8'))
    doc = Document(); set_fonts(doc)
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Mm(210), Mm(297)
    sec.top_margin, sec.bottom_margin = Mm(25), Mm(22); sec.left_margin = sec.right_margin = Mm(17)
    sec.header_distance, sec.footer_distance = Mm(12), Mm(10)
    hp = sec.header.paragraphs[0]; hp.text = 'Journal of the Korean Society of Manufacturing Technology Engineers (manuscript draft v5)'
    fmt(hp, size=7.5, align=WD_ALIGN_PARAGRAPH.LEFT)
    fp = sec.footer.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fld = OxmlElement('w:fldSimple'); fld.set(qn('w:instr'), 'PAGE'); fp._p.append(fld)
    # ---- front matter, one column
    p = doc.add_paragraph(); runs(p, d['title_ko'], size=16, bold=True); fmt(p, 16, WD_ALIGN_PARAGRAPH.CENTER, after=6)
    p = doc.add_paragraph(); runs(p, d['title_en'], size=12, bold=True); fmt(p, 12, WD_ALIGN_PARAGRAPH.CENTER, after=6)
    p = doc.add_paragraph(); runs(p, '(저자·소속·교신저자: 최종 파일 업로드 시 작성)', size=9); fmt(p, 9, WD_ALIGN_PARAGRAPH.CENTER, after=8)
    rule(doc)
    p = doc.add_paragraph(); runs(p, 'ABSTRACT', size=9, bold=True); fmt(p, 9, WD_ALIGN_PARAGRAPH.LEFT, after=2)
    p = doc.add_paragraph(); runs(p, d['abstract'], size=9); fmt(p, 9, after=4, exact=12.5)
    p = doc.add_paragraph(); r = p.add_run('Key Words: '); r.bold = True; r.font.size = Pt(9); runs(p, d['keywords'], size=9)
    fmt(p, 9, WD_ALIGN_PARAGRAPH.LEFT, after=2)
    rule(doc)
    body = doc.add_section(WD_SECTION.CONTINUOUS); columns(body, 2)
    # ---- body, two columns
    pending_cap = None
    for b in d['blocks']:
        kind = b[0]
        if kind == 'h1':
            p = doc.add_paragraph(); runs(p, b[1], size=12, bold=True); fmt(p, 12, WD_ALIGN_PARAGRAPH.CENTER, before=10, after=5, keep=True)
        elif kind == 'h2':
            p = doc.add_paragraph(); runs(p, b[1], size=10.5, bold=True); fmt(p, 10.5, WD_ALIGN_PARAGRAPH.LEFT, before=6, after=3, keep=True)
        elif kind == 'p':
            p = doc.add_paragraph(); runs(p, b[1], size=10); fmt(p, 10, indent=2.8, exact=16)
        elif kind == 'fig':
            path = (SRC.parent / re.sub(r'/(v\d)_fig', r'/\1c_fig', b[2])).resolve()
            p = doc.add_paragraph(); p.add_run().add_picture(str(path), width=Mm(80))
            fmt(p, align=WD_ALIGN_PARAGRAPH.CENTER, before=6, after=2, keep=True)
            c = caption(doc, b[1]); fmt(c, 9, WD_ALIGN_PARAGRAPH.JUSTIFY, after=8, exact=11.5)
        elif kind == 'tabcap':
            pending_cap = b[1]
        elif kind == 'table':
            if pending_cap:
                c = caption(doc, pending_cap); fmt(c, 9, WD_ALIGN_PARAGRAPH.JUSTIFY, before=6, after=3, exact=11.5, keep=True); pending_cap = None
            three_line_table(doc, b[1])
            p = doc.add_paragraph(); fmt(p, size=4, after=4)
    if d['ack']:
        p = doc.add_paragraph(); runs(p, '후 기', size=12, bold=True); fmt(p, 12, WD_ALIGN_PARAGRAPH.CENTER, before=10, after=5, keep=True)
        p = doc.add_paragraph(); runs(p, d['ack'], size=10); fmt(p, 10, indent=2.8, exact=16)
    refs = doc.add_section(WD_SECTION.NEW_PAGE); columns(refs, 1)        # template: references on a separate page
    p = doc.add_paragraph(); runs(p, 'References', size=12, bold=True); fmt(p, 12, WD_ALIGN_PARAGRAPH.LEFT, after=6)
    reference_form(doc, d['refs'])
    doc.save(DST)
    out = subprocess.run(['soffice', '--headless', '--convert-to', 'pdf', '--outdir', str(DST.parent), str(DST)],
                         capture_output=True, text=True, timeout=300)
    print(DST, DST.with_suffix('.pdf'), out.returncode)


if __name__ == '__main__':
    build()
