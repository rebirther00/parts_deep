"""Reuse the established review renderer and attach one v10 proofing memo."""
from pathlib import Path
import argparse
import write_v9_review as renderer

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'reports/논문초안_v10_리뷰_20261110'
renderer.DEST = DEST
renderer.engine.DEST = DEST
renderer.engine.SOURCE = ROOT / 'manuscript/논문초안_v10.hwpx'
renderer.engine.ANNOTATED = ROOT / 'manuscript/논문초안_v10_검토메모_20261110.hwpx'
renderer.engine.COMMENTS = [
    ('Table 4 Five-fold out-of-fold identification', '최종 교정 권고 — RGBE 신뢰구간 상한 반올림',
     'v9의 필수 편집 지적은 해결됐고 새 학습이나 재구성은 필요하지 않습니다. 이 표 RGBE CNN 행의 359/361 Wilson 95% 구간을 다시 계산하면 98.0027846–99.8479369%입니다. 현재 98.0–99.9%의 상한을 일반 반올림에 맞춰 98.0–99.8%로, 또는 두 자리 표기인 98.00–99.85%로 정리하는 것을 권합니다. 현재 값은 바깥쪽으로 넓힌 표기이지만 별도의 바깥쪽 반올림 규칙이 없으므로 다른 구간과 통일하기 위한 교정입니다. 정답 수, 정확도, Macro-F1과 비교 결론은 그대로입니다. 이전 검토에서 놓친 표기 차이입니다. 원본은 보존했습니다.')
]


def report():
    from docx import Document
    renderer.report()
    target = DEST / '검토의견.docx'
    doc = Document(target)
    doc.sections[0].header.paragraphs[0].text = '논문초안 v10 전반 검토 | 2026-11-10 | 원고 원본 보존'
    # The first table has three columns; the shared renderer defaults to two.
    from docx.shared import Cm
    from docx.oxml import OxmlElement
    table = doc.tables[0]
    widths = [3.0, 12.2, 1.8]
    for col, width in zip(table.columns, widths):
        col.width = Cm(width)
    import re
    source = (DEST / '검토의견.md').read_text().split('## v9 검토 반영 여부')[1].split('## 최종 교정 의견')[0]
    values = [[c.strip() for c in line.strip().strip('|').split('|')] for line in source.splitlines() if line.strip().startswith('|')]
    values = [row for row in values if not all(re.fullmatch(r'[:\- ]+', c) for c in row)]
    assert len(values) == len(table.rows)
    for row, cells in zip(table.rows, values):
        for cell, text, width in zip(row.cells, cells, widths):
            cell.text = text
            cell.width = Cm(width)
    for row in table.rows[:-1]:
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.keep_with_next = True
    doc.save(target)
    assert Document(target).tables[0].cell(1, 2).text == '해결'
    renderer.engine.check_originals()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['memos', 'report'])
    args = parser.parse_args()
    {'memos': renderer.engine.memos, 'report': report}[args.action]()
