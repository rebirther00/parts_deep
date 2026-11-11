"""Write v6 review artifacts, reusing the verified native HWPX memo writer."""
from pathlib import Path
import argparse
import re
import write_v5_review as engine

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'reports/논문초안_v6_리뷰_20261110'
engine.DEST = DEST
engine.SOURCE = ROOT / 'manuscript/논문초안_v6.hwpx'
engine.ANNOTATED = ROOT / 'manuscript/논문초안_v6_검토메모_20261110.hwpx'
engine.COMMENTS = [
    ('학습 세션에 같은 판정을 적용하면', '교정 C1 — U 반례의 원인을 더 정확히',
     'U와 도어 판정의 동치를 삭제한 핵심 수정은 적절합니다. 다만 OOF 7건·시간 분할 4건을 모두 “영상 대부분이 보류되고 남은 영상이 정답”으로 설명하면 부정확합니다. 3건은 정답/오판/보류가 9/6/5 또는 9/8/3입니다. “보류를 제외한 영상에서 정답 클래스가 최빈값이었으며 일부 사례에는 정답·오판·보류가 함께 있었다”로 고치세요. 총 11건은 실물 8개의 평가 결과입니다. 차별성 문서의 “반례 11개”도 평가 건수로 명시하세요. 결론의 U 최대 50%도 성공 보장처럼 연결하지 말고, 모든 계산 가능한 영상의 잔차가 판정 구간 안이었다는 사실과 나누어 쓰면 명확합니다.'),
    ('공통 미학습 조건. 해당 형상군을 실험실·현장 학습에서', '확인·기록 교정 C2/C4 — 시험 좌표 역할과 분모',
     '새 모델 8개의 학습·선택·실험실 모니터링에서 제외 형상군이 빠졌고, 실제 시험 좌표임을 대조했습니다. 모델마다 시험 8장·48점의 PCK8 100%, 제외 형상군만 합한 8장·48점의 오차 중앙값 2.5258화소·최대 5.1526화소가 재현됩니다. 본문에 합산 분모를 명시하세요. 다만 새 metrics의 test_role_localization.scope는 여전히 “checkpoint-selection VAL images; ... not independent test”라는 예전 문자열입니다. 실제 누출은 발견하지 못했고 설명 메타데이터가 틀린 문제입니다. 기존 산출물을 보존한 정정 기록과 생성 함수의 역할별 scope 처리를 권합니다.'),
    ('공통 학습 조건. 운용 중인 형상군을 목록에서 빼는 경우이다.', '명료화 C3 — 마스킹 수식과 검증 자료',
     '구현의 q_j=p_j/Σ(i≠c)p_i, j≠c를 한 줄로 명시하면 좋겠습니다. 임계값은 정답 형상군이 c가 아닌 검증 영상에 같은 마스킹을 적용해 다시 정합니다. 본문에 이 표본 제외 조건도 넣으세요. 새 가중치 학습 없이 출력과 임계값을 바꾸는 조건이라는 설명은 맞고, S를 고정한 홀 조건도 코드·저장 값과 일치합니다.'),
    ('세 조건을 함께 보면, 이미 학습한 형상군을', '문구 교정 C3 — 정확히 5% 보류는 아님',
     '“검증 영상의 5%를 보류하도록”을 “검증 영상의 최소 약 95%를 수락하도록”으로 통일하세요. 새 마스킹 조건의 실제 검증 수락률은 95.0000~95.0835%였습니다. 이 운용점의 등록 도어 보류를 일반 CNN의 필연적 한계로 해석하지 않고, 공통 미학습 조건에 차별성의 범위를 둔 현재 서술은 적절합니다.'),
    ('Table 8 Pooled leave-one-class-out results', '확인·편집 권고 C4 — 조건별 결과와 캡션',
     '공통 학습 120개 평가와 새 홀 검출기 8개를 확인했으며 Table 7·8의 주요 집계·AUROC는 일치합니다. 이전 R3/R4 비교 조건 문제는 실질적으로 해결됐습니다. 캡션이 길므로 반복 판정·AUROC 평균 설명을 주석 또는 4.4절로 옮기고 조건 이름을 강조하면 읽기 쉽겠습니다. 비대칭 조건은 참고 문장이나 보충자료로 줄여도 되며 필수 변경은 아닙니다. 한글 정본의 최종 배치는 별도로 확인하세요.'),
]


def report():
    from docx import Document
    from docx.shared import Cm, Pt
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    engine.check_originals()
    target = DEST / '검토의견.docx'
    if target.exists():
        raise FileExistsError(target)
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.top_margin = sec.bottom_margin = Cm(1.9)
    sec.left_margin = sec.right_margin = Cm(2)
    for name in ['Normal', 'Title', 'Heading 1', 'Heading 2', 'Intense Quote']:
        style = doc.styles[name]
        style.font.name = 'Noto Sans CJK KR'
        style._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), 'Noto Sans CJK KR')
    doc.styles['Normal'].font.size = Pt(10)
    doc.styles['Normal'].paragraph_format.line_spacing = 1.15
    doc.styles['Normal'].paragraph_format.space_after = Pt(6)
    doc.styles['Title'].font.size = Pt(20)
    sec.header.paragraphs[0].text = '논문초안 v6 검토 | 2026-11-10 | 원고 원본 보존'
    footer = sec.footer.paragraphs[0]
    footer.alignment = 2
    field = OxmlElement('w:fldSimple')
    field.set(qn('w:instr'), 'PAGE')
    footer._p.append(field)

    def plain(text):
        return re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1 (\2)', text).replace('**', '').replace('`', '')

    table = None
    for line in (DEST / '검토의견.md').read_text().splitlines():
        line = line.strip()
        if line.startswith('|'):
            values = [plain(x.strip()) for x in line.strip('|').split('|')]
            if all(re.fullmatch(r'[:\- ]+', x) for x in values):
                continue
            if table is None:
                table = doc.add_table(rows=0, cols=len(values))
                table.style = 'Light Shading Accent 1'
            row = table.add_row()
            if len(table.rows) == 1:
                row._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
            row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
            for cell, value in zip(row.cells, values):
                cell.text = value
            continue
        table = None
        if not line:
            continue
        if line.startswith('# '):
            doc.add_heading(plain(line[2:]), 0)
        elif line.startswith('## '):
            doc.add_heading(plain(line[3:]), 1)
        elif line.startswith('### '):
            doc.add_heading(plain(line[4:]), 2)
        elif line.startswith('> '):
            doc.add_paragraph(plain(line[2:]), style='Intense Quote')
        else:
            doc.add_paragraph(plain(line))
    doc.save(target)
    checked = Document(target)
    assert len(checked.tables) == 3
    assert any('R1~R5는 실질적으로 해결됐다' in p.text for p in checked.paragraphs)
    engine.check_originals()
    print(target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['memos', 'report'])
    args = parser.parse_args()
    {'memos': engine.memos, 'report': report}[args.action]()
