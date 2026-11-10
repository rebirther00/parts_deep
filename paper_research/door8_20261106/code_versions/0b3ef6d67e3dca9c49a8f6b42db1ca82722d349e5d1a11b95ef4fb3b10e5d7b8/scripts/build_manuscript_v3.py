"""Build manuscript/논문초안_v3.docx from 논문초안_v3.md (A4, figures and tables inline).

Inline markup handled: **bold**, ^superscript^, ~subscript~. Lines '![caption](path)' become figures.
The journal (KSMTE) template is not applied here; this is a clean A4 draft for review.
"""
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from door8.core import ROOT

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

SRC = ROOT / 'manuscript/논문초안_v3.md'
DST = ROOT / 'manuscript/논문초안_v3.docx'
TOKEN = re.compile(r'(\*\*[^*]+\*\*|\^[^^]+\^|~[^~]+~)')


def runs(par, text, size=None, bold=False):
    for part in TOKEN.split(text):
        if not part:
            continue
        if part.startswith('**'):
            r = par.add_run(part[2:-2]); r.bold = True
        elif part.startswith('^'):
            r = par.add_run(part[1:-1]); r.font.superscript = True; r.bold = bold
        elif part.startswith('~'):
            r = par.add_run(part[1:-1]); r.font.subscript = True; r.bold = bold
        else:
            r = par.add_run(part); r.bold = bold
        if size:
            r.font.size = Pt(size)


def set_fonts(doc):
    for name in ('Normal', 'Title', 'Heading 1', 'Heading 2', 'Heading 3'):
        st = doc.styles[name]
        st.font.name = 'Times New Roman'
        st._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'), '바탕')
        st.font.color.rgb = None
    doc.styles['Normal'].font.size = Pt(10)
    pf = doc.styles['Normal'].paragraph_format
    pf.line_spacing = 1.3; pf.space_after = Pt(3)
    doc.styles['Title'].font.size = Pt(16)
    doc.styles['Heading 1'].font.size = Pt(12)
    doc.styles['Heading 2'].font.size = Pt(10.5)


def add_table(doc, rows):
    t = doc.add_table(rows=0, cols=len(rows[0]))
    t.style = 'Table Grid'; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, cells in enumerate(rows):
        row = t.add_row()
        for cell, value in zip(row.cells, cells):
            cell.text = ''
            p = cell.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            runs(p, value, size=8.5, bold=(i == 0))
        if i == 0:
            row._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
    doc.add_paragraph()


def build():
    if DST.exists():
        raise FileExistsError(f'{DST} exists; use a new version name')
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.top_margin = sec.bottom_margin = Cm(2.5); sec.left_margin = sec.right_margin = Cm(2.2)
    set_fonts(doc)
    footer = sec.footer.paragraphs[0]; footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fld = OxmlElement('w:fldSimple'); fld.set(qn('w:instr'), 'PAGE'); footer._p.append(fld)
    lines = SRC.read_text(encoding='utf8').splitlines()
    table, first_para = [], True
    for raw in lines + ['']:
        line = raw.strip()
        if line.startswith('|'):
            cells = [c.strip() for c in line.strip('|').split('|')]
            if not all(re.fullmatch(r':?-+:?', c) for c in cells):
                table.append(cells)
            continue
        if table:
            add_table(doc, table); table = []
        if not line:
            continue
        m = re.fullmatch(r'!\[(.+)\]\((.+)\)', line)
        if m:
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.add_run().add_picture(str((SRC.parent / m.group(2)).resolve()), width=Cm(15.5))
            c = doc.add_paragraph(); c.alignment = WD_ALIGN_PARAGRAPH.CENTER
            runs(c, m.group(1), size=9)
            continue
        if line.startswith('# '):
            p = doc.add_paragraph(style='Title'); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; runs(p, line[2:])
        elif line.startswith('## '):
            doc.add_heading(line[3:], 1)
        elif line.startswith('### '):
            doc.add_heading(line[4:], 2)
        elif line.startswith('Table '):
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; runs(p, line, size=9, bold=True)
        else:
            p = doc.add_paragraph()
            if first_para:            # English title line under the Korean title
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER; runs(p, line, size=12, bold=True); first_para = False
            else:
                runs(p, line)
    doc.save(DST)
    print(DST)


if __name__ == '__main__':
    build()
