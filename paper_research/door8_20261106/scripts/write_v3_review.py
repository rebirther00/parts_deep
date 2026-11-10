"""Write review copies and anchored comments without changing the author's v3."""
from pathlib import Path
import re
import sys
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from docx import Document
from docx.shared import Cm, Pt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from lxml import etree
from door8.core import ROOT, read_json, dump, sha256, now

DEST = ROOT / 'reports/논문초안_v3_리뷰_20261109'
SOURCE = ROOT / 'manuscript/논문초안_v3.docx'
ANNOTATED = ROOT / 'manuscript/논문초안_v3_검토메모.docx'

# Paragraph numbers refer to python-docx's document.paragraphs, one-based.
COMMENTS = [
    (65, '필수 1 — 지그 공차의 근거',
     '±2.22%×약 1,475 mm ≈33 mm는 다른 오차가 없는 전면 평행 조건의 일차 민감도입니다. 실제 관측 CAD 잔차 최대 1.14%, 자세·양자화·보정·제조 편차가 이미 여유를 사용하므로 ±33 mm를 보장 공차로 제시할 수 없습니다. 이상식도 오차=−ΔZ/(S+ΔZ)로 비대칭입니다. “민감도 환산이며 실제 공차는 오차 예산과 거리·자세 변화 실험으로 결정”으로 수정하거나, 공차 주장을 유지할 통제 실험을 추가하세요. 17 mm 역시 위치 변화의 실측값이 아닙니다.'),
    (86, '필수 2 — 3.3분과 새 현장 적용 범위',
     '40장 16분 23초는 사용자 확인 시간으로 인정할 수 있습니다. 그러나 3.3분은 학습용 8장만의 비례 환산입니다. 체크포인트 선택용 현장 좌표 8장이 추가로 필요해 동일 평균의 16장 합계는 약 6.6분입니다. 실험실 좌표와 학습 세션 품종 라벨도 사용했습니다. 한 현장·한 시드의 결과로 “새 현장에 필요한 주석은 수 분”을 일반화하지 말고 “동일 현장에서 학습 8장과 검증 8장을 사용해 회복했다”로 한정하세요.'),
    (34, '필수 3 — 미등록 품종은 자동으로 거절되지 않음',
     '이 규칙은 미등록 여부 자체가 아니라 최근접 CAD와의 차이>30 mm를 검사합니다. 미등록 품종의 D가 등록값 부근이거나 같으면 기존 품종으로 수락됩니다. CAD만 추가하는 확장도 동일 홀 정의·기하 검사·충분한 치수 간격이 성립할 때 가능합니다. E23에서 관측한 조건부 기능으로 기술하고 모든 신규 품종으로 확대하지 마세요.'),
    (80, '필수 4 — CNN 비교의 운용점과 주장 범위',
     '검증의 약 95% 수락 임계값은 명시된 기준선이고 결과도 맞습니다. 다만 홀의 30 mm 기준과 판정 비율이 맞춰진 비교는 아닙니다. 정상 20~25개 보류는 선택한 CNN 운용점에서의 관측입니다. 기존 저장 점수로 정확도/위험–판정 비율 곡선을 추가하고 실제 임계값을 공개하는 것이 좋습니다. 새 임계값을 시험 E23 점수로 최적화하면 탐색 분석으로 표시해야 합니다. “CNN은 재학습이 필요”도 일반 CNN이 아니라 여기서 구현한 고정 8출력 분류기에 한정하세요. OpenMax 등 다른 개방 집합 CNN도 존재합니다.'),
    (7, '필수 5 — 초록의 오분류·보류와 대조군 범위',
     '“misjudged 29 doors”는 저장 결과와 집계 의미가 다릅니다. 깊이 OOF는 오분류 16개, 보류 13개입니다. “misclassified 16 doors and abstained on 13”으로 나누세요. 실험실 전용 대조군의 정식 결과는 fold 0 시험 74개 전부 보류로 제시되어 있으므로 “every field door”가 361개 전체를 뜻하지 않게 범위를 한정하세요. 3.3분도 학습 8장만의 환산이며 현장 검증 8장은 별도입니다.'),
    (87, '필수 6 — 프레임 최대와 세션 중앙값 최대가 혼재',
     '픽셀 최대 1.14%는 계산 가능한 OOF 프레임 기준입니다. 깊이의 약 8%는 세션별 중앙값의 최대(+8.24%)이고, 전체 계산 가능한 OOF 프레임 최대는 +10.08%(시간 분할 +10.27%)입니다. 같은 통계 단위로 비교하세요. “오차”는 실물 실측값 대비가 아니라 CAD 기준 잔차입니다. 마지막의 “지그 공차를 ±33 mm로 정할 수 있다”도 근사 민감도 표현으로 낮춰야 합니다. 근거: reports/extension_v3/analysis.json.'),
    (65, '필수 7 — Word 변환에서 범위 기호 손실',
     '문서 생성기의 ~…~ 아래첨자 문법이 숫자의 범위 기호 ~와 충돌했습니다. 이 문단에서 “−1.14+1.11%”, “1,4741,475”, “595%”처럼 기호가 사라지고 본문 102자가 아래첨자로 변했습니다. Markdown 범위를 en dash(–) 또는 “에서/까지”로 바꾸거나 변환 문법을 수정한 뒤 재생성하세요. 검토본은 오류를 보이기 위해 원래 본문을 유지했습니다.'),
    (70, '보강 1 — 깊이 실패의 원인 단정',
     '같은 자동 좌표로 픽셀/깊이를 비교하고 수동점에서도 잔차가 남았다는 것은 깊이 거리 경로의 문제를 지지합니다. 하지만 수동점 검증 8장으로 전체 361개 실패의 원인을 검출과 무관하다고 확정할 수는 없습니다. 평면의 배경 포함, 유효 깊이, 자세, 내부 파라미터 및 보정의 기여는 아직 분리하지 않았습니다. “깊이 측정·평면 추정 경로가 주요 제한 요인임을 시사한다”로 낮추세요.'),
    (51, '보강 2 — 수동 좌표 진단이 증명한 범위',
     '수동 좌표 진단의 픽셀 8/8 정답과 CAD 잔차는 실제 결과와 맞습니다. 다만 이것은 클래스별 한 장인 검증 8장의 진단입니다. “실패 원인은 판정 규칙이 아니라 검출”을 모든 입력과 촬영 조건에 대한 단정으로 확대하지 말고 “해당 검증 표본에서는 정확한 좌표가 주어지면 판정 가능하여 검출을 우선 개선할 근거를 얻었다”로 한정하면 방어하기 쉽습니다.'),
    (52, '보강 3 — 고정 참조의 독립성 의미',
     '기간 이전에 고정한 가중치가 같은 결과를 보인 것은 유용합니다. 그러나 동일 도어 자료로 평가했으므로 새 독립 표본의 재현은 아닙니다. “별도 학습 이력의 고정 검출기에서도 같은 경향”으로 표현하세요. 참조 학습의 107/27 분할에는 학습 현장 13장 외에 같은 세션의 현장 검증 3장도 있으므로 Table 2에 검증 감독까지 표시하는 것이 정확합니다.'),
    (48, '보강 4 — 후향적 설계 선택 과정',
     '학습·시험 역할을 직접 대조했고 중복은 발견하지 않았습니다. 다만 v1 시험 실패를 본 뒤 현장 적응을 설계했고 fold 0 검증 결과로 8장을 선택한 순서를 여기에 함께 적으세요. 같은 5개 그룹을 순환하는 OOF는 모델별 학습 분리를 달성하지만 앞선 개발 선택의 영향까지 제거한 중첩 CV/새 블라인드 시험은 아닙니다. 프로토콜 고정 후 미조정이라는 사실과 이전 관찰 사실을 같이 밝혀야 합니다.'),
    (61, '보강 5 — 동등성 및 감독 정보 해석',
     '점추정은 유사하지만 동등성·비열등성을 검정한 것은 아닙니다. “관측 차이는 작고 우월성을 확인하지 못했다”가 더 정확합니다(저장된 McNemar p=1.0). 또한 품종 라벨은 S 보정 외에 클래스별 현장 좌표 표본 구성에도 사용했습니다. “판정 단계에서는 클래스별 분류 가중치 대신 CAD 최근접값을 사용한다”로 쓰면 감독 정보가 전혀 필요 없었다는 오해를 줄일 수 있습니다.'),
    (71, '보강 6 — 관측과 원인 가설 분리',
     '세션 내 분산이 매우 작고 격자 간격이 약 6.25픽셀인 것은 확인 가능한 관측입니다. 이것만으로 도어 간 잔차의 원인을 거치 상태·양자화로 확정하거나 부화소 보간이 실제 오차를 줄인다고 단정할 수는 없습니다. “양자화가 분포에 영향을 주었을 가능성이 있으며, 부화소 보간 효과는 후속 검증이 필요”로 수정하거나 같은 데이터의 보간 제거 실험을 추가하세요.'),
    (43, '보강 7 — 임계값과 실제 검증 판정 비율',
     '신뢰도 동점 때문에 실제 검증 수락률은 정확히 95%가 아닙니다. RGB fold 0은 임계값 0.9999998808에서 96.01%, 시간 분할은 0.9999995232에서 95.62%입니다. “검증 영상의 최소 약 95%를 수락하도록 정한 임계값”이라고 쓰고 보조표에 임계값/검증 수락률/시험 수락률을 넣으세요. 이 관측은 계산 오류라는 뜻은 아닙니다.'),
    (37, '보강 8 — 평가 프레임의 실제 추출 모집단',
     'v3에서 로컬 미러 설명이 빠졌습니다. 포함 361세션의 원 수집 36,142쌍 전체에서 직접 추출한 것이 아니라, 로컬 보유 7,376쌍에서 세션당 최대 20장을 뽑아 7,104쌍을 고정했습니다. 두 문장으로 복원하면 표본 선택이 분명해집니다. 361개가 각각 다른 실물이라는 사실은 그대로 유지합니다.'),
    (78, '보강 9 — E23 사례의 난이도와 표본 수',
     '458 mm는 기존 최소 724 mm와 266 mm 떨어져 있어 현재 거절 규칙으로 구분하기 좋은 조건입니다. 6개 실물의 사례를 일반적인 개방 집합 인식 성능으로 확대하지 마세요. 6/6 성공의 Wilson 95% 하한은 약 61%입니다. 현재 8클래스 본 실험과 구분한 부가 사례 구성은 타당하며, 가까운 치수의 미등록 품종까지 주장할 경우에만 그에 맞는 추가 평가가 필요합니다.'),
    (84, '보강 10 — 추가 실험의 우선순위',
     '한 현장·한 카메라·한 시드·후향 자료·일치 라벨의 잔여 오류 가능성을 명시한 점은 적절합니다. 지금은 새 학습보다 과도한 결론과 비교 운용점의 수정을 우선할 수 있습니다. 이후 추가 시드, 모델 고정 후 새 작업 도어, 일치 사례의 도면/작업기록 기반 층화 확인을 권합니다. 시드 반복을 독립 도어 표본 증가로 세지 않도록 하세요.'),
    (95, '확인 완료 — 참고문헌 4의 서지',
     '출판사 검색 결과에서 Shengbin Lin, Suhua Xiao, Ruihao Lin, Jiayao Wu, Wenbin Luo; Measurement 258 Part C, 119276, 2026; DOI 10.1016/j.measurement.2025.119276을 확인했습니다. 작성 확인 사항의 서지 확인은 이 근거로 완료 표시할 수 있습니다. 직접 원문 페이지는 403이어서 유료 원문 전체를 새로 읽었다는 뜻은 아닙니다. https://www.sciencedirect.com/science/article/abs/pii/S0263224125026351'),
]


def create_report():
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = sec.bottom_margin = Cm(2)
    sec.left_margin = sec.right_margin = Cm(2.2)
    for name in ['Normal', 'Title', 'Heading 1', 'Heading 2']:
        s = doc.styles[name]
        s.font.name = 'Malgun Gothic'
        s._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), '맑은 고딕')
    doc.styles['Normal'].font.size = Pt(10)
    doc.styles['Normal'].paragraph_format.line_spacing = 1.15
    doc.styles['Title'].font.size = Pt(20)
    sec.header.paragraphs[0].text = '논문초안 v3 검토 의견 | 2026-11-09 | 원고 본문 수정 없음'
    footer = sec.footer.paragraphs[0]
    footer.alignment = 2
    f = OxmlElement('w:fldSimple'); f.set(qn('w:instr'), 'PAGE'); footer._p.append(f)
    def plain(x):
        return re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1 (\2)', x).replace('**', '').replace('`', '')
    table = None
    for line in (DEST / '검토의견.md').read_text().splitlines():
        line = line.strip()
        if line.startswith('|'):
            values = [plain(x.strip()) for x in line.strip('|').split('|')]
            if all(re.fullmatch(r'[:\- ]+', x) for x in values):
                continue
            if table is None:
                table = doc.add_table(rows=0, cols=len(values)); table.style = 'Light Shading Accent 1'
            row = table.add_row()
            if len(table.rows) == 1:
                row._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
            row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
            for c, value in zip(row.cells, values):
                c.text = value
            continue
        table = None
        if not line:
            continue
        if line.startswith('# '):doc.add_heading(plain(line[2:]), 0)
        elif line.startswith('## '):doc.add_heading(plain(line[3:]), 1)
        elif line.startswith('### '):doc.add_heading(plain(line[4:]), 2)
        elif line.startswith('> '):doc.add_paragraph(plain(line[2:]), style='Intense Quote')
        else:doc.add_paragraph(plain(line))
    doc.save(DEST / '검토의견.docx')


def main():
    evidence = read_json(DEST / '근거대조.json')
    if sha256(SOURCE) != evidence['original_sha256']:
        raise RuntimeError('Author manuscript changed since review: re-check before annotating.')
    if ANNOTATED.exists() or (DEST / '검토의견.docx').exists():
        raise FileExistsError('Preserve existing review outputs.')
    original = Document(SOURCE)
    texts = [p.text for p in original.paragraphs]
    tables = [[[c.text for c in row.cells] for row in t.rows] for t in original.tables]
    existing = len(original.comments)
    manifest = []
    for paragraph, heading, body in COMMENTS:
        p = original.paragraphs[paragraph - 1]
        anchors = [r for r in p.runs if r.text]
        comment = original.add_comment(anchors, text=heading + '\n\n' + body, author='Codex 원고 검토', initials='CX')
        manifest.append(dict(comment_id=comment.comment_id, paragraph=paragraph, heading=heading, text=body,
                             anchor_excerpt=p.text[:180]))
    original.save(ANNOTATED)
    checked = Document(ANNOTATED)
    assert [p.text for p in checked.paragraphs] == texts
    assert [[[c.text for c in row.cells] for row in t.rows] for t in checked.tables] == tables
    assert len(checked.comments) == existing + len(COMMENTS)
    ns = {'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    with zipfile.ZipFile(SOURCE) as before, zipfile.ZipFile(ANNOTATED) as after:
        assert all(before.read(n) == after.read(n) for n in before.namelist() if n.startswith('word/media/'))
        root = etree.fromstring(after.read('word/document.xml'))
        starts = set(root.xpath('//w:commentRangeStart/@w:id', namespaces=ns))
        ends = set(root.xpath('//w:commentRangeEnd/@w:id', namespaces=ns))
        refs = set(root.xpath('//w:commentReference/@w:id', namespaces=ns))
        assert starts == ends == refs
    create_report()
    assert sha256(SOURCE) == evidence['original_sha256']
    assert sha256(ROOT/'manuscript/논문초안_v3.md') == evidence['original_md_sha256']
    dump(DEST / '검토메모목록.json', manifest)
    dump(DEST / '검토완료.json', dict(completed_at=now(), comments_added=len(COMMENTS),
         original_unchanged=True, original_sha256=evidence['original_sha256'],
         annotated_manuscript=str(ANNOTATED.relative_to(ROOT)), annotated_sha256=sha256(ANNOTATED),
         report='reports/논문초안_v3_리뷰_20261109/검토의견.docx',
         checks=['same manuscript paragraphs and table text', 'same embedded image bytes',
                 'all comment ranges matched', 'original DOCX and Markdown hashes unchanged']))
    print({'comments':len(COMMENTS), 'annotated':str(ANNOTATED), 'report':str(DEST/'검토의견.docx')})


if __name__ == '__main__':
    main()
