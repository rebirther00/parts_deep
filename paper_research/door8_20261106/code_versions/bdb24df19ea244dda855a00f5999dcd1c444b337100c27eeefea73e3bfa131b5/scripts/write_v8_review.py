"""Make v8 review artifacts, preserving author files and old reviews."""
from pathlib import Path
import argparse
import re
import write_v5_review as engine

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'reports/논문초안_v8_리뷰_20261110'
engine.DEST=DEST
engine.SOURCE=ROOT/'manuscript/논문초안_v8.hwpx'
engine.ANNOTATED=ROOT/'manuscript/논문초안_v8_검토메모_20261110.hwpx'
engine.COMMENTS=[
    ('학습 세션에 같은 판정을 적용하면', '본문 수치 유지 — 답변서의 학습 사례 설명만 정정',
     'v7의 네 지적은 해결됐습니다. 다만 v8 답변서 2.3과 작성확인사항의 “나머지 19장 모두 812.7 mm”는 저장 결과와 다릅니다. 폴드 4 학습 세션 E38 LH FRT s_183313의 실제 판정은 정답 6장(812.7047 mm), 오판 1장(764.1189 mm), 보류 13장(모서리 누락 11·볼트 배열 실패 2)입니다. 보류 제외 최빈값이 6 대 1이므로 도어 정답과 본문의 학습 초과 1건은 그대로 맞습니다. 원고 본문을 고치라는 메모가 아니며, 답변서·대응 기록의 새 사례 설명을 정정하라는 뜻입니다. 피크 응답 0.1554는 맞습니다. 별도 검토서에 대체 문장과 판정 근거를 남겼습니다.')
]

def report():
    from docx import Document
    from docx.shared import Cm,Pt
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    engine.check_originals()
    target=DEST/'검토의견.docx'
    if target.exists():raise FileExistsError(target)
    doc=Document();sec=doc.sections[0]
    sec.page_width,sec.page_height=Cm(21),Cm(29.7)
    sec.top_margin=sec.bottom_margin=Cm(1.9)
    sec.left_margin=sec.right_margin=Cm(2)
    for name in ['Normal','Title','Heading 1','Heading 2','Intense Quote']:
        style=doc.styles[name];style.font.name='Noto Sans CJK KR'
        style._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Noto Sans CJK KR')
    doc.styles['Normal'].font.size=Pt(10)
    doc.styles['Normal'].paragraph_format.line_spacing=1.12
    doc.styles['Normal'].paragraph_format.space_after=Pt(5)
    doc.styles['Title'].font.size=Pt(20)
    sec.header.paragraphs[0].text='논문초안 v8 재검토 | 2026-11-10 | 원고 원본 보존'
    field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE')
    sec.footer.paragraphs[0].alignment=2;sec.footer.paragraphs[0]._p.append(field)
    def plain(s):return s.replace('**','').replace('`','')
    table=None
    for line in (DEST/'검토의견.md').read_text().splitlines():
        line=line.strip()
        if line.startswith('|'):
            values=[plain(v.strip()) for v in line.strip('|').split('|')]
            if all(re.fullmatch(r'[:\- ]+',v) for v in values):continue
            if table is None:
                table=doc.add_table(rows=0,cols=len(values));table.style='Light Shading Accent 1'
                widths=[4.0,10.0,3.0] if values[0]=='v7 지적' else [6.4,1.5,9.1]
                for col,width in zip(table.columns,widths):col.width=Cm(width)
            row=table.add_row();row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
            if len(table.rows)==1:row._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
            for cell,value,width in zip(row.cells,values,widths):cell.text=value;cell.width=Cm(width)
            continue
        table=None
        if not line:continue
        if line.startswith('# '):doc.add_heading(plain(line[2:]),0)
        elif line.startswith('## '):doc.add_heading(plain(line[3:]),1)
        elif line.startswith('> '):doc.add_paragraph(plain(line[2:]),style='Intense Quote')
        else:doc.add_paragraph(plain(line))
    doc.save(target)
    assert len(Document(target).tables)==2
    engine.check_originals();print(target)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['memos','report']);a=p.parse_args()
    {'memos':engine.memos,'report':report}[a.action]()
