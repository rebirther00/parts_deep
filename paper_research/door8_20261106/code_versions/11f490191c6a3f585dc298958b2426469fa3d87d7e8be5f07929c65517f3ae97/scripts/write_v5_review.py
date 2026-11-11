"""Create annotated v5 HWPX and separate report; never alter author originals."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import json
import re

from write_v4_review import manuscript_structure, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'reports/논문초안_v5_리뷰_20261110'
SOURCE = ROOT / 'manuscript/논문초안_v5.hwpx'
ANNOTATED = ROOT / 'manuscript/논문초안_v5_검토메모_20261110.hwpx'
HP = 'http://www.hancom.co.kr/hwpml/2011/paragraph'

COMMENTS = [
    ('도어 한 개의 상대 잔차', '필수 R1 — U와 도어 실패는 동치가 아님',
     'U는 계산 가능한 영상의 잔차 중앙값이고, 도어 판정은 보류하지 않은 영상의 최빈값입니다. 저장 결과에서 깊이 OOF 7개·시간 분할 4개가 U≥1인데 정답입니다. 예: 20261022/E25_door_RH/s_085619의 폴드 4는 U=1.415, 20장 중 19장 보류·1장 정답이어서 세션 정답입니다. “U<1은 중앙 측정값이 판정 구간 내부임을 뜻하며 도어 성공·실패와 일대일 대응하지 않는다”로 고치세요. U는 정답 CAD를 사용하는 진단 통계입니다. U=1 경계와 측정 불가도 별도 취급해야 합니다. 기존 판정 규칙을 바꾸라는 뜻은 아닙니다.'),
    ('학습 세션에 같은 판정을 적용한 결과는', '필수 R2 — 사전 예측 주장을 후향적 비교로',
     '합산 7.8%와 8.0%는 맞지만, 깊이 폴드 0은 학습 6.54%→시험 13.51%, 폴드 4는 7.80%→4.35%, 시간 분할은 4.15%→18.80%입니다. 같은 기간에서 잘 예측했다는 표현도 합산 결과에 한정해야 합니다. 같은 학습 세션으로 검출기와 S·K를 맞춘 뒤 평가한 값이므로 독립 검증된 예측 성능은 아닙니다. “학습–시험 잔차·실패의 후향적 비교”로 바꾸고 초록·서론·Table 5·결론·차별성 문서를 함께 맞추세요. 1,083건은 361도어의 3회 반복 판정입니다. 폴드별 표와 대체 문장은 별도 검토서에 있습니다.'),
    ('이 비교에서 홀 검출기는', '필수 R3 — 학습 노출 차이는 최종 보류에도 영향',
     '홀 검출기가 제외 형상군의 실험실·현장 영상을 보았음을 공개한 점은 적절합니다. 그러나 “검출 정확도에만 관계된다”는 답변은 비교의 문제를 해소하지 않습니다. 검출 정확도는 거리 측정과 보류 결과에 직접 영향을 줍니다. “동일한 미지 클래스 학습 조건의 일반화 우월성이 아니라 서로 다른 학습 조건의 운용 결과 비교”로 한정하세요. 학습에 없던 E23 여섯 도어는 보완 사례이며 이 비교의 조건 차이를 없애지는 않습니다. 엄격판 재학습을 하지 않는다면 주장 범위를 조정하면 됩니다.'),
    ('품종 지식이 학습 가중치 밖에 있으므로', '필수 R4 — 제외는 축척 재보정, E23 등록은 CAD만 변경',
     '4.4절과 코드에서는 형상군을 제외할 때 그 형상군을 뺀 학습 세션으로 축척 S도 다시 구합니다. 따라서 이 결과 전체를 “CAD 목록만 변경”한 것으로 요약하면 부정확합니다. “제외에서는 검출기 고정·CAD 목록 변경·축척 재보정, 별도 E23 등록에서는 기존 축척을 유지한 CAD 값 추가”로 구분하세요. 마지막의 가까운 품종에 다른 랜드마크가 반드시 필요하다는 문장도 측정 잔차 대비 간격이 부족한 경우로 한정해야 합니다.'),
    ('다품종 혼류 생산에서는', '필수 R4 — CNN의 클래스 제거에 재학습이 필수는 아님',
     '“품종을 추가하거나 빼려면 그 품종 영상으로 다시 학습해야 한다”는 일반화는 성립하지 않습니다. 학습된 클래스 출력은 마스킹 또는 해당 출력에 대한 보류로 제외할 수 있습니다. 이번 7클래스 재학습은 그 클래스를 보지 않은 기준선을 만들기 위한 실험 선택입니다. “본 연구에서 구현한 고정 출력 분류기는 등록하지 않은 새 클래스 이름을 출력할 수 없다”로 범위를 한정하세요. 학습 가중치에 품종 정보가 전혀 없다는 의미의 “CAD에만”도 피하는 것이 정확합니다.'),
    ('제안 방법은 영상에서 학습하는 부분과', '표현 정밀화 — 판정 목록과 학습 가중치의 분리',
     '“공통 랜드마크 검출 가중치와 품종별 CAD 판정 목록을 분리했다”는 명확한 운영상의 주장입니다. 그러나 클래스별 영상·표본 구성·S 보정에 형상군 정보가 쓰였으므로 “품종 지식은 CAD에만”을 학습 표현에 품종 정보가 전혀 없다는 의미로 쓰지 마세요. 분리 구조와 조건부 목록 운용을 실제 현장에서 검증한 기여는 유지할 수 있습니다.'),
    ('본 연구의 데이터는 한 사업장의', '필수 R5 — 30 mm 이내의 미등록 보류와 등록 후 구분',
     '“기존 값과 30 mm 이내의 D를 갖는 품종은 구분하거나 거절할 수 없다”는 과도합니다. 미등록 측정값이 기존값의 30 mm 안이면 수락될 수 있지만, 신규 CAD 값을 등록한 후에는 두 값 간격보다 측정 잔차가 충분히 작으면 구분 가능합니다. D가 같거나 간격이 잔차에 비해 부족할 때 다른 특징이 필요하다고 쓰세요. 참 CAD 차이와 실제 측정값 차이도 구분해야 합니다. 같은 문단의 사전 예측 주장은 R2와 함께 조정하세요.'),
    ('Excavator side door classes look alike', '초록 교정 — 분석 범위·학습 조건·74개 집계',
     '8/8과 24/3 구분은 개선됐습니다. 다만 “24 + 3”을 24 training and three selection images로 풀어 쓰세요. “Training sessions anticipated test reliability”는 후향적 비교로 낮추고, CAD 제외 시 S 재보정과 두 방법의 제외 형상군 학습 노출 차이를 반영해야 합니다. “all 74 ... of that class”는 형상군 하나의 도어가 74개라는 오해를 줍니다. 여덟 실행 합계 74개(형상군별 2~14개)라고 쓰세요. 현재 공백 기준 151단어이며 별도 검토서에 교체 예시가 있습니다.'),
    ('Table 7 Unregistered classes.', '확인·권고 — 수치는 일치, 반복 판정과 평균 정의 명시',
     'CNN 16회와 홀 40회의 저장 예측을 독립 재집계했고 주요 수치는 일치했습니다. 폴드 0으로 직접 비교를 맞춘 것은 적절합니다. 등록 판정 518=74×7, OOF의 2,527=361×7은 반복 판정 건수임을 명시하세요. AUROC 평균은 8개 실행의 동일 가중 평균입니다. 신뢰구간은 관측 도어의 기술 요약이며 새로운 형상군 전체와 재학습 변동을 포함하지 않습니다. 별도 차별성 문서의 질문 3·9·10 답변은 검토서의 수정안을 참고하세요.'),
]


def check_originals():
    values = json.loads((DEST / '원본해시.json').read_text())
    for name, digest in values.items():
        assert sha256(ROOT / name) == digest, f'Original changed: {name}'
    return values


def memos():
    from hwpx.document import HwpxDocument
    from hwpx.oxml.paragraph import HwpxOxmlParagraph
    hashes = check_originals()
    if ANNOTATED.exists():
        raise FileExistsError(ANNOTATED)
    before = manuscript_structure(SOURCE)
    doc = HwpxDocument.open(SOURCE)
    assert not doc.validate().issues
    assert not doc.notes.memos
    section = doc.sections[0]
    candidates = [(p, ''.join(p.itertext())) for p in section.element.iter('{' + HP + '}p')]
    created = datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')
    manifest = []
    for number, (prefix, heading, text) in enumerate(COMMENTS, 1):
        hits = [(p, t) for p, t in candidates if t.startswith(prefix)]
        assert len(hits) == 1, (prefix, len(hits))
        el, original = hits[0]
        memo = doc.notes.add_memo(heading + '\n\n' + text, anchor=HwpxOxmlParagraph(el, section),
                                  author='Codex 원고 검토', created=created, number=number)
        manifest.append(dict(number=number, field_id=memo.field_id, paragraph_id=el.get('id'),
                             heading=heading, text=text, anchor_excerpt=original[:250]))
    doc.save_to_path(ANNOTATED)
    checked = HwpxDocument.open(ANNOTATED)
    assert len(checked.notes.memos) == len(COMMENTS)
    assert not checked.validate().issues
    assert manuscript_structure(ANNOTATED) == before
    check_originals()
    write_json(DEST / '검토메모목록.json', manifest)
    write_json(DEST / '검토완료.json', dict(review_date='2026-11-10', completed_at=created,
        original_sha256=hashes, original_unchanged=True, native_hwpx=True, memos_added=len(COMMENTS),
        annotated=str(ANNOTATED.relative_to(ROOT)), annotated_sha256=sha256(ANNOTATED),
        checks=dict(all_original_paragraphs_preserved=len(before['paragraphs']), tables=before['tables'],
                    figures=before['figures'], image_bytes_unchanged=True, library_validation_issues=0),
        layout_scope='Native Hancom rendering and memo UI not checked; companion PDF is a DOCX approximation.'))
    print(json.dumps(dict(memos=len(COMMENTS), paragraphs=len(before['paragraphs']), original_unchanged=True), ensure_ascii=False))


def report():
    from docx import Document
    from docx.shared import Cm, Pt
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    check_originals()
    target = DEST / '검토의견.docx'
    if target.exists():
        raise FileExistsError(target)
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.top_margin = sec.bottom_margin = Cm(1.9)
    sec.left_margin = sec.right_margin = Cm(2)
    for name in ['Normal', 'Title', 'Heading 1', 'Heading 2', 'Intense Quote']:
        s = doc.styles[name]
        s.font.name = 'Noto Sans CJK KR'
        s._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), 'Noto Sans CJK KR')
    doc.styles['Normal'].font.size = Pt(10)
    doc.styles['Normal'].paragraph_format.line_spacing = 1.15
    doc.styles['Normal'].paragraph_format.space_after = Pt(6)
    doc.styles['Title'].font.size = Pt(20)
    sec.header.paragraphs[0].text = '논문초안 v5·차별성 검토 | 2026-11-10 | 원고 원본 보존'
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
    reopened = Document(target)
    assert len(reopened.tables) == 3
    assert any('U=1.415' in p.text for p in reopened.paragraphs)
    check_originals()
    print(target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['memos', 'report'])
    args = parser.parse_args()
    {'memos': memos, 'report': report}[args.action]()
