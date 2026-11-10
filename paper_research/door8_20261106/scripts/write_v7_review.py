"""Create a separate v7 review and a native memo copy; preserve all originals."""
from pathlib import Path
import argparse
import re
import write_v5_review as engine

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'reports/논문초안_v7_리뷰_20261110'
engine.DEST = DEST
engine.SOURCE = ROOT / 'manuscript/논문초안_v7.hwpx'
engine.ANNOTATED = ROOT / 'manuscript/논문초안_v7_검토메모_20261110.hwpx'
engine.COMMENTS = [
    ('대상은 굴착기 측면 도어 8개 형상군이다', '소규모 교정 — 판정 구간의 경계 동률',
     '측정 가능한 영상에 한정한 저자 답변의 동치 논리는 내부·외부에 대해 수용합니다. 다만 닫힌 구간 끝점은 동률 규칙이 필요합니다. 724와 765 mm의 중간값 744.5 mm에서 구현은 목록상 먼저 오는 E25를 선택하므로 E30의 하단 경계를 정답 영역에 포함할 수 없습니다. “중간값에서는 CAD 목록 순서의 동률 규칙을 적용하고 정확히 30 mm 차이는 수락한다”를 한 문장 보충하면 충분합니다. 현재 U<1과 구간 밖에 관한 본문 핵심은 유지할 수 있으며, 재계산한 측정에서는 해당 동률이 없었습니다. 수치 변경·재학습 요구가 아닙니다.'),
    ('학습 세션에 같은 판정을 적용하면', '확인 — 저자 추가 정정과 시험 범위 수용',
     '학습 픽셀 영상 1/21,259, 시험 0/7,087을 저장 검출점에서 재현했습니다. 학습 초과는 폴드 4의 E38 LH FRT s_183313, rgb_0140.png로 측정 764.119 mm입니다. 이전 검토에서 제가 제안한 “모든 영상”은 범위가 넓었습니다. v7의 “시험 영상” 수정과 1건 공개가 맞습니다. 깊이 반례 11건·실물 8개 및 세 사례 9/6/5, 9/6/5, 9/8/3도 재현되어 C1은 해결됐습니다.'),
    ('Table 7 Doors rejected when their class', '최종 편집 — 비교표 단 나눔 확인',
     '제공 PDF 10쪽에서는 Table 7의 첫 두 데이터 행만 왼쪽 단 하단에 놓이고 나머지가 오른쪽 단으로 이어집니다. 내용은 읽히고 머리행도 반복되지만, 핵심 비교표를 한 단에 모으면 읽기 쉽겠습니다. 캡션 설명을 4.4절로 옮긴 선택은 이전 권고 범위에 맞습니다. 이 PDF는 DOCX의 LibreOffice 근사본이므로 한글 정본의 실제 배치를 별도로 확인하세요.'),
]

def report():
    from docx import Document
    from docx.shared import Cm, Pt
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    engine.check_originals()
    target=DEST/'검토의견.docx'
    if target.exists(): raise FileExistsError(target)
    doc=Document(); sec=doc.sections[0]
    sec.page_width,sec.page_height=Cm(21),Cm(29.7)
    sec.top_margin=sec.bottom_margin=Cm(1.9)
    sec.left_margin=sec.right_margin=Cm(2)
    for name in ['Normal','Title','Heading 1','Heading 2','Intense Quote']:
        style=doc.styles[name]; style.font.name='Noto Sans CJK KR'
        style._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Noto Sans CJK KR')
    doc.styles['Normal'].font.size=Pt(10)
    doc.styles['Normal'].paragraph_format.line_spacing=1.15
    doc.styles['Normal'].paragraph_format.space_after=Pt(6)
    doc.styles['Title'].font.size=Pt(20)
    sec.header.paragraphs[0].text='논문초안 v7 검토 | 2026-11-10 | 원고 원본 보존'
    field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE')
    sec.footer.paragraphs[0].alignment=2;sec.footer.paragraphs[0]._p.append(field)
    def plain(s): return s.replace('**','').replace('`','')
    table=None
    for line in (DEST/'검토의견.md').read_text().splitlines():
        line=line.strip()
        if line.startswith('|'):
            values=[plain(v.strip()) for v in line.strip('|').split('|')]
            if all(re.fullmatch(r'[:\- ]+',v) for v in values):continue
            if table is None:table=doc.add_table(rows=0,cols=len(values));table.style='Light Shading Accent 1'
            row=table.add_row();row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
            if len(table.rows)==1:row._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
            for cell,value in zip(row.cells,values):cell.text=value
            continue
        table=None
        if not line:continue
        if line.startswith('# '):doc.add_heading(plain(line[2:]),0)
        elif line.startswith('## '):doc.add_heading(plain(line[3:]),1)
        elif line.startswith('### '):doc.add_heading(plain(line[4:]),2)
        elif line.startswith('> '):doc.add_paragraph(plain(line[2:]),style='Intense Quote')
        else:doc.add_paragraph(plain(line))
    doc.save(target)
    assert len(Document(target).tables)==2
    engine.check_originals()
    print(target)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['memos','report']);a=p.parse_args()
    {'memos':engine.memos,'report':report}[a.action]()
