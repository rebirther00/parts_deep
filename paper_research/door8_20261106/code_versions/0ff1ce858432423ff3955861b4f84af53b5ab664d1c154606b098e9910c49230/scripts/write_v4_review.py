"""Create a native HWPX memo copy and a separate review; preserve source files."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import hashlib
import json
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'reports/논문초안_v4_리뷰_20261109'
SOURCE = ROOT / 'manuscript/논문초안_v4.hwpx'
ANNOTATED = ROOT / 'manuscript/논문초안_v4_검토메모.hwpx'
HP = 'http://www.hancom.co.kr/hwpml/2011/paragraph'

# Match the original paragraphs, including the abstract inside the front table.
COMMENTS = [
    ('Excavator side doors of different models', '필수 1 — 초록의 감독량 구분',
     '현재 문장은 교차검증과 시간 분할 모두 학습 8장·선택 8장을 쓴 것으로 읽힙니다. 실제 교차검증은 폴드마다 8/8장, 시간 분할은 이전 기간의 학습 24장·선택 3장입니다(본문 4.2절과 Table 6은 맞음). 두 조건을 별도 문장으로 나누세요. 예: “Eight training and eight validation images per fold yielded 361/361 correct identifications. A separate temporal model using 24 training and three validation images from the earlier period identified all 133 later doors.” 전체 초록 수정 예시는 별도 검토서에 있습니다.'),
    ('임계값 운용점의 선택에 따른 차이를 보기 위해', '필수 2 — OOF 합산 곡선과 운용점의 정책 차이',
     '수치는 저장 점수 재계산 결과와 일치합니다. 다만 OOF 곡선은 서로 다른 5개 모델의 원시 점수를 합쳐 하나의 공통 임계값을 바꾸고, 사전 운용점은 폴드마다 따로 정한 검증 임계값의 결과를 합칩니다. 그래서 사각형이 합산 곡선 위에 놓이지 않을 수 있습니다. RGB에서는 0.9999238253인 영상이 수락되는 반면 다른 폴드의 0.9999997616인 영상은 거절됩니다. 본문/캡션에 이 차이를 밝혀 주세요. 합산 곡선은 탐색적 기술 통계이며 단일 배포 모델의 검증된 보류 곡선은 아닙니다. 추가 학습 없이 설명을 보완할 수 있습니다.'),
    ('Fig. 4 Exploratory risk-coverage analysis', '권고 1 — 임계값 동점과 연결선',
     'RGB OOF의 처음 세 실현 가능한 판정 비율은 50.662%, 58.516%, 63.105%입니다. 같은 점수의 영상이 함께 들어오므로 사이의 모든 비율이 실제 임계값으로 실현되지는 않습니다. 관측점 마커 또는 “연결선은 시각적 안내이며 동점 사이 중간 비율은 실제 운용점을 뜻하지 않는다”는 설명을 권합니다. OOF의 공통 임계값 곡선과 폴드별 임계값 사각형의 차이도 캡션에 명시하세요.'),
    ('E23(Fig. 4(c))에서 홀 방법은', '필수 3 — 30 mm 선정 경위',
     'RGB도 사후 임계값으로 E23을 분리할 수 있음을 공개한 점은 적절합니다. 마지막의 “CAD 치수 간격에 따라 정한 고정값 30 mm”는 선정 근거를 점검해 주세요. 기존 docs/실험계획.md에는 과거 개발에서 정한 설정으로 기록되어 있고, 최소 CAD 간격 41 mm의 절반은 20.5 mm입니다. 별도 도출 근거가 없다면 “과거 개발에서 정하고 본 평가 전에 고정한 CAD 거리 허용값 30 mm”로 쓰는 것이 정확합니다. 정답 영역은 인접 CAD 중간값과 보류 경계 중 안쪽 범위입니다. 시험 결과를 보고 임계값을 다시 맞추라는 뜻은 아닙니다.'),
    ('무작위 분할에서 외관 CNN의 정확도는', '권고 2 — 결론에 새 E23 분석 반영',
     '5.6절에서는 RGB도 정상 시험 영상을 모두 수락하면서 E23을 전부 거절할 사후 임계값 구간이 있었음을 보고했습니다. 결론에도 거절 자체의 일반적 우월성을 확인한 것은 아니라는 점을 짧게 연결하면 균형이 좋아집니다. 홀 방식의 의미는 충분히 떨어진 치수와 같은 홀 배치라는 조건에서 CAD 등록만으로 식별 대상을 확장한 사례입니다. 현재 정확도나 6/6 결과가 틀렸다는 지적은 아닙니다.'),
    ('Table 5 Pixel-width versus depth-plane distance', '권고 3 — 잔차 통계의 실제 표본 수',
     '전체 입력은 7,104장이지만 잔차를 계산할 수 있는 영상 수는 픽셀 7,087장, 깊이 7,086장입니다. 깊이의 계산 가능한 CAD 거리 보류값도 잔차 통계에 포함됩니다. 각주에 이 분모를 밝히면 좋겠습니다. 전체 정확도의 분모는 계속 7,104장으로 유지합니다.'),
    ('이 논문은 【부처명 확인】', '투고 전 확인 — 후기 기관명과 지정 문구',
     '현재 공식 기관명은 “한국산업기술기획평가원(KEIT)”입니다. 원고의 “한국산업기술평가관리원”은 예전 명칭입니다. https://www.keit.re.kr/ . 부처명·사업명·과제번호 및 사사 문구는 실제 협약서의 요구 문구로 확인해 주세요. 본 검토에서 이 과제의 지원 관계까지 검증한 것은 아닙니다.'),
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def check_originals():
    evidence = json.loads((DEST / '근거대조.json').read_text())
    for name, digest in evidence['original_sha256'].items():
        assert sha256(ROOT / name) == digest, f'Original changed: {name}'
    return evidence


def manuscript_structure(path):
    from lxml import etree
    ns = {'hp': HP}
    # Native memos have an inline copy and a memogroup copy; exclude both.
    keep = "not(ancestor::hp:fieldBegin[@type='MEMO']) and not(ancestor::*[local-name()='memo'])"
    with zipfile.ZipFile(path) as z:
        root = etree.fromstring(z.read('Contents/section0.xml'))
        paragraphs = [
            (p.get('id'), ''.join(p.xpath(f'.//hp:t[{keep}]/text()', namespaces=ns)))
            for p in root.xpath(f'//hp:p[{keep}]', namespaces=ns)
        ]
        media = {n: hashlib.sha256(z.read(n)).hexdigest()
                 for n in z.namelist() if n.startswith('BinData/')}
        return dict(paragraphs=paragraphs, images=media,
                    tables=len(root.xpath('//hp:tbl', namespaces=ns)),
                    figures=len(root.xpath('//hp:pic', namespaces=ns)))


def create_memos():
    from hwpx.document import HwpxDocument
    from hwpx.oxml.paragraph import HwpxOxmlParagraph
    evidence = check_originals()
    if ANNOTATED.exists():
        raise FileExistsError(ANNOTATED)
    before = manuscript_structure(SOURCE)
    doc = HwpxDocument.open(SOURCE)
    assert not doc.validate().issues
    assert len(doc.notes.memos) == 0
    section = doc.sections[0]
    # Snapshot first, so new memo content cannot match a later anchor.
    candidates = [(p, ''.join(p.itertext())) for p in section.element.iter('{' + HP + '}p')]
    created = datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')
    manifest = []
    for number, (prefix, heading, text) in enumerate(COMMENTS, 1):
        hits = [(p, t) for p, t in candidates if t.startswith(prefix)]
        assert len(hits) == 1, (prefix, len(hits))
        element, original_text = hits[0]
        paragraph = HwpxOxmlParagraph(element, section)
        memo = doc.notes.add_memo(heading + '\n\n' + text, anchor=paragraph,
                                  author='Codex 원고 검토', created=created, number=number)
        manifest.append(dict(number=number, field_id=memo.field_id, paragraph_id=element.get('id'),
                             heading=heading, text=text, anchor_excerpt=original_text[:250]))
    doc.save_to_path(ANNOTATED)
    checked = HwpxDocument.open(ANNOTATED)
    assert len(checked.notes.memos) == len(COMMENTS)
    assert not checked.validate().issues
    assert manuscript_structure(ANNOTATED) == before, 'Manuscript text or embedded images changed'
    check_originals()
    write_json(DEST / '검토메모목록.json', manifest)
    write_json(DEST / '검토완료.json', dict(completed_at=created, native_hwpx=True,
               memos_added=len(COMMENTS), original_sha256=evidence['original_sha256'],
               original_unchanged=True, annotated_sha256=sha256(ANNOTATED),
               annotated=str(ANNOTATED.relative_to(ROOT)),
               checks=['Original HWPX/MD/DOCX/PDF hashes unchanged',
                       'All 432 original paragraph IDs/text preserved, excluding memo content',
                       'Embedded image bytes unchanged; 19 tables and 4 figures preserved',
                       'Native memo copy reopens and has no library validation issues'],
               layout_scope='Native Hancom rendering and memo UI visibility not inspected; supplied PDF is a DOCX approximation.'))
    print(json.dumps(dict(memos=len(COMMENTS), paragraphs=len(before['paragraphs']),
                          original_unchanged=True, output=str(ANNOTATED)), ensure_ascii=False))


def create_report():
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
        style = doc.styles[name]
        style.font.name = 'Noto Sans CJK KR'
        style._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), 'Noto Sans CJK KR')
    doc.styles['Normal'].font.size = Pt(10)
    doc.styles['Normal'].paragraph_format.line_spacing = 1.15
    doc.styles['Normal'].paragraph_format.space_after = Pt(6)
    doc.styles['Title'].font.size = Pt(20)
    sec.header.paragraphs[0].text = '논문초안 v4 HWPX 검토 | 2026-11-09 | 원고 원본 보존'
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
    assert len(reopened.tables) == 2
    assert any('24장·선택 3장' in p.text for p in reopened.paragraphs)
    check_originals()
    print(target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['memos', 'report'])
    args = parser.parse_args()
    {'memos': create_memos, 'report': create_report}[args.action]()
