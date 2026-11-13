"""Write v9 review artifacts without changing author manuscripts."""
from pathlib import Path
import argparse
import re
import write_v5_review as engine

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'reports/논문초안_v9_리뷰_20261110'
engine.DEST = DEST
engine.SOURCE = ROOT / 'manuscript/논문초안_v9.hwpx'
engine.ANNOTATED = ROOT / 'manuscript/논문초안_v9_검토메모_20261110.hwpx'
engine.COMMENTS = [
    ('Fig. 1 (a) Field image', '제출 전 수정 1 — 캡션의 (b) 흐름도가 실제 그림에 없음',
     'HWPX와 DOCX에 삽입된 이미지의 해시를 확인했습니다. 두 파일은 사진만 있는 figures/v3c_fig1_method.png를 사용하지만, 캡션은 (a) 사진과 (b) 판정 흐름도를 설명합니다. 원래 figures/v3_fig1_method.png에는 두 패널이 있습니다. 핵심 기여인 분리 구조를 보여 주도록 흐름도를 복원하는 편을 권합니다. 사진만 유지한다면 캡션의 (a)/(b)와 decision pipeline을 없애고 3.1절의 그림 인용도 맞추세요. 이전 검토에서 놓친 편집 불일치이며, 새 실험은 필요하지 않습니다.'),
    ('Korea Construction Equipment Technology Institute, 36 Sandan-ro',
     '제출 전 수정 2 — 우편번호 자리표시를 54004로',
     '소속 주소의 【우편번호】가 HWPX·PDF·MD·DOCX에 남아 있습니다. 기관 공식 영문 안내 https://www.koceti.re.kr/home/en 의 36 Sandan-ro 주소 우편번호는 54004입니다. 해당 표시를 54004로 바꾸고 생성 원천에도 반영하세요. 저자 영문명은 파일 간 일치만 확인했습니다.'),
]


def report():
    from docx import Document
    from docx.shared import Cm, Pt
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.opc.constants import RELATIONSHIP_TYPE as RT

    engine.check_originals()
    target = DEST / '검토의견.docx'
    if target.exists():
        raise FileExistsError(target)
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.top_margin = sec.bottom_margin = Cm(1.8)
    sec.left_margin = sec.right_margin = Cm(2)
    for name in ['Normal', 'Title', 'Heading 1', 'Heading 2']:
        style = doc.styles[name]
        style.font.name = 'Noto Sans CJK KR'
        style._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), 'Noto Sans CJK KR')
    normal = doc.styles['Normal']
    normal.font.size = Pt(10)
    normal.paragraph_format.line_spacing = 1.1
    normal.paragraph_format.space_after = Pt(5)
    doc.styles['Title'].font.size = Pt(20)
    for name, size in [('Heading 1', 13), ('Heading 2', 11)]:
        style = doc.styles[name]
        style.font.size = Pt(size)
        style.paragraph_format.space_before = Pt(11)
        style.paragraph_format.space_after = Pt(5)
    sec.header.paragraphs[0].text = '논문초안 v9 전반 검토 | 2026-11-10 | 원고 원본 보존'
    sec.header.paragraphs[0].style = doc.styles['Normal']
    field = OxmlElement('w:fldSimple')
    field.set(qn('w:instr'), 'PAGE')
    sec.footer.paragraphs[0].alignment = 2
    sec.footer.paragraphs[0]._p.append(field)

    def inline(p, text):
        pattern = r'\[([^\]]+)\]\(([^)]+)\)|\*\*([^*]+)\*\*|`([^`]+)`'
        pos = 0
        for m in re.finditer(pattern, text):
            p.add_run(text[pos:m.start()])
            if m.group(1):
                link = OxmlElement('w:hyperlink')
                link.set(qn('r:id'), p.part.relate_to(m.group(2), RT.HYPERLINK, is_external=True))
                run = OxmlElement('w:r')
                props = OxmlElement('w:rPr')
                color = OxmlElement('w:color'); color.set(qn('w:val'), '0563C1')
                props.append(color); run.append(props)
                txt = OxmlElement('w:t'); txt.text = m.group(1)
                run.append(txt); link.append(run); p._p.append(link)
            else:
                run = p.add_run(m.group(3) or m.group(4))
                run.bold = bool(m.group(3))
            pos = m.end()
        p.add_run(text[pos:])

    table = None
    for line in (DEST / '검토의견.md').read_text().splitlines():
        line = line.strip()
        if line.startswith('|'):
            values = [v.strip() for v in line.strip('|').split('|')]
            if all(re.fullmatch(r'[:\- ]+', v) for v in values):
                continue
            if table is None:
                table = doc.add_table(rows=0, cols=len(values))
                table.style = 'Light Shading Accent 1'
                table.autofit = False
                for col, width in zip(table.columns, [3.5, 13.5]):
                    col.width = Cm(width)
            row = table.add_row()
            row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
            if len(table.rows) == 1:
                row._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
            for cell, value, width in zip(row.cells, values, [3.5, 13.5]):
                cell.width = Cm(width)
                inline(cell.paragraphs[0], value)
            continue
        table = None
        if not line:
            continue
        if line.startswith('# '):
            doc.add_heading(line[2:], 0)
        elif line.startswith('## '):
            doc.add_heading(line[3:], 1)
        else:
            inline(doc.add_paragraph(), line)
    for table in doc.tables:
        for row in table.rows[:-1]:
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.keep_with_next = True
    doc.save(target)
    assert len(Document(target).tables) == 2
    engine.check_originals()
    print(target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['memos', 'report'])
    args = parser.parse_args()
    {'memos': engine.memos, 'report': report}[args.action]()
