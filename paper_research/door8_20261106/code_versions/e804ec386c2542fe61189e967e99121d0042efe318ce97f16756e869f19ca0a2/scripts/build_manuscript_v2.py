"""Create a separate, evidence-linked development draft; preserve all v1 outputs."""
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from door8.core import ROOT, read_json, outpath, dump, sha256, now


def section(text, start, end):
    return text.split(start, 1)[1].split(end, 1)[0]


def markdown_document(text, path, figures):
    from docx import Document
    from docx.shared import Cm, Pt
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    def plain(s):
        return re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1 (\2)', s).replace('**', '').replace('`', '')
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = sec.bottom_margin = Cm(2)
    sec.left_margin = sec.right_margin = Cm(2.2)
    for name in ['Normal', 'Title', 'Heading 1', 'Heading 2', 'Heading 3']:
        style = doc.styles[name]
        style.font.name = 'Malgun Gothic'
        style._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), '맑은 고딕')
    doc.styles['Normal'].font.size = Pt(10)
    doc.styles['Normal'].paragraph_format.line_spacing = 1.25
    doc.styles['Title'].font.size = Pt(22)
    sec.header.paragraphs[0].text = '연구용 초안 v2 | 현장 적응 개발 검증 | 최종 시험·전향적 검증 미완료'
    footer = sec.footer.paragraphs[0]
    footer.alignment = 2
    field = OxmlElement('w:fldSimple')
    field.set(qn('w:instr'), 'PAGE')
    footer._p.append(field)
    table = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith('|'):
            cells = [plain(x.strip()) for x in line.strip('|').split('|')]
            if all(re.fullmatch(r'[:\- ]+', c) for c in cells):
                continue
            if table is None:
                table = doc.add_table(rows=0, cols=len(cells))
                table.style = 'Light Shading Accent 1'
            row = table.add_row()
            if len(table.rows) == 1:
                repeat_header = OxmlElement('w:tblHeader')
                row._tr.get_or_add_trPr().append(repeat_header)
            keep_row = OxmlElement('w:cantSplit')
            row._tr.get_or_add_trPr().append(keep_row)
            for cell, value in zip(row.cells, cells):
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
    doc.add_page_break()
    doc.add_heading('개발 검증 도표', 1)
    for source, caption in figures:
        doc.add_picture(str(source), width=Cm(16))
        doc.add_paragraph(caption)
    doc.save(path)


def build():
    output = outpath('manuscript/논문초안_v2_개발검증.md')
    word = output.with_suffix('.docx')
    if output.exists() or word.exists():
        raise FileExistsError('Preserve existing manuscripts; use a new version for further revisions.')
    old = (ROOT / 'manuscript/논문초안_v1.md').read_text()
    report = ROOT / 'reports/field_v2_development_comparison'
    combined = read_json(report / 'comparison.json')
    assert combined['test_frames_evaluated'] == 0
    assert combined['field_sampling_confirmation']['distinct_physical_doors'] == 361
    rows = combined['table']
    full = next(r for r in rows if r['budget'] == 24 and r['mode'] == 'pixel')
    table = '\n'.join(
        f'| {r["budget"]} | {r["mode"]} | {r["frame_correct"]}/{r["frame_n"]} ({100*r["frame_accuracy"]:.2f}%) | '
        f'{100*r["frame_coverage"]:.2f}% | {r["session_correct"]}/{r["session_n"]} | '
        f'{100*r["session_coverage"]:.2f}% | {r["session_macro_f1"]:.4f} |' for r in rows)
    loc_rows = []
    for budget, loc in combined['localization_by_budget'].items():
        ch = loc['per_channel']
        loc_rows.append(f'| {budget} | {100*ch["bolt"]["pck8"]:.2f}% | {100*ch["hinge"]["pck8"]:.2f}% | '
                        f'{100*ch["latch"]["pck8"]:.2f}% | {100*loc["all"]["pck8"]:.2f}% |')
    introduction = section(old, '## 1. 서론', '## 2. 관련 연구')
    introduction = introduction.split('본 연구가 검증하려는 기여는')[0] + (
        '본 연구의 중심 질문은 실험실에서 학습한 국소 홀 검출기가 현장으로 전이되지 않을 때, '
        '제한된 현장 수동 좌표로 검출을 회복할 수 있는지, 그리고 검출 이후 픽셀·깊이 거리 판정에서 '
        '어떤 차이가 남는지이다. 이를 위해 실험실 전용 대조군을 보존하고 현장 좌표 학습량을 '
        '8/16/24장으로 나누며, 사람의 좌표를 입력하는 진단과 자동 검출 결과를 구분한다. '
        '새로운 신경망 구조의 최초성이나 외관 분류 대비 우월성을 전제하지 않는다. '
        '현재 초안은 이 질문에 대한 한 폴드의 개발 검증까지 기술하며, 최종 일반화 주장은 후속 평가에 남겨둔다.\n\n')
    related = section(old, '## 2. 관련 연구', '## 3. 홀 랜드마크 기반 품종 식별 방법')
    method = section(old, '## 3. 홀 랜드마크 기반 품종 식별 방법', '## 4. 실험 설계')
    method = method.replace(
        '수동으로 주석한 실험실 영상 118장을 사용하며 그중 94장은 학습, 24장은 체크포인트 선택에 사용한다.',
        '수동으로 주석한 실험실 영상 118장 중 94장은 학습에 사용한다. 실험실 전용 0장 대조군은 나머지 24장의 손실로 체크포인트를 선택했다. '
        '현장 적응 8/16/24장 모델은 같은 실험실 학습 94장에 해당 폴드의 현장 학습 좌표만 추가하며, '
        '현장 검증 좌표 8장의 손실로 체크포인트를 선택한다. 이때 실험실 검증 24장은 모니터링에만 사용한다.')
    method = method.replace(
        '보조 제거 실험에서는 검출과 학습 보정을 고정한 채 기하 거절 규칙의 일부를 제거하거나, 평면 깊이 1400~1500 mm 및 기울기 15° 이하의 자세 조건을 추가한다. 보조 실험의 실패·보류도 평가 분모에 포함한다. 시험 성능을 본 뒤 유리한 거절 조건만 대표 결과로 선택하지 않는다.',
        '이번 적응 개발 비교에서는 기존 검출 임계값, 기하 검사와 CAD 보류 규칙을 고정했다. 깊이 또는 자세 조건을 사후 조정하여 이 결과를 개선하지 않았다. '
        '후속 기하 제거 실험과 자세 조건 분석은 별도 프로토콜로 수행하며 실패·보류도 평가 분모에 포함한다.')
    refs = section(old, '## 참고문헌', '## 그림 구성 메모')
    text = f'''# 소량 현장 수동 좌표를 이용한 굴착기 박판 도어 홀 랜드마크 적응과 CAD 기반 식별의 개발 검증

Development Validation of Hole-Landmark Adaptation with Limited Manual Field Annotations for CAD-Based Identification of Excavator Sheet-Metal Doors

저자·소속: [입력 필요]

> 연구용 초안 v2, 2026-11-09. 40장 수동 주석과 Fold 0/seed 42의 현장 적응 개발 검증을 반영했다. 361세션이 서로 다른 실물 도어 361개라는 현장 수집 담당자의 확인을 반영했다. 이 버전은 투고 완성본이 아니다. 새로운 모델의 최종 시험, 전체 교차검증 및 전향적 현장 평가는 미완료이며 기존 v1 원고·결과는 보존한다.

## 초록

굴착기 박판 도어의 현장 영상에서는 전체 전경과 슬롯을 안정적으로 분할하기 어려운 사례가 있어, 내부 가공 홀의 위치와 CAD 치수를 이용하는 식별 방법을 검토하였다. 실제 작업에서 서로 다른 도어 361개를 촬영한 361세션의 7,104 RGB–깊이 영상 쌍을 고정하고 실물에 대응하는 세션 단위로 분할하였다. 실험실 수동 좌표 118장만으로 개발한 검출기는 현장 기준 홀 검출에 실패하여 초기 시험에서 전체 입력을 보류하였다. 이에 현장 수동 좌표 40장을 추가로 작성하고 개발 폴드에서 학습 후보 24장, 검증 8장, 시험용 8장으로 구분하였다. 실험실 학습 94장에 현장 8/16/24장을 추가한 모델을 각각 ImageNet 초기화로 새로 학습하고, 같은 검증 73세션·1,455프레임에서 비교하였다. 24장 추가 모델의 픽셀 거리 판정은 {full['frame_correct']}/1,455프레임({100*full['frame_accuracy']:.2f}%)과 {full['session_correct']}/73세션을 맞혔다. 동일 검증 자료에서 RGB 분류기는 모든 프레임과 세션을 맞혔으므로 홀 방법의 우월성을 입증한 결과는 아니다. 수동 좌표 진단과 자동 검출 비교는 현장 홀 검출의 적응 가능성을 보여주지만, 깊이 기반 거리 오차와 새 촬영 조건으로의 일반화는 추가 검증이 필요하다. 본 결과는 모델 선택에 사용된 개발 검증이며 최종 시험 성능과 구분한다.

핵심어: 굴착기 도어, 홀 랜드마크, 현장 적응, 수동 주석, CAD 치수, 판정 보류

## Abstract

Internal hole landmarks and CAD dimensions are investigated for identifying excavator sheet-metal doors where complete foreground and slot segmentation can be unreliable. The retrospective dataset comprises 7,104 RGB–depth pairs from 361 production sessions, each depicting a distinct physical door, with session-disjoint partitions. A detector developed using 118 manually annotated laboratory images abstained on all inputs in the initial field test because reference holes were not reliably detected. Forty field images were then manually annotated and allocated, in the development fold, to 24 candidate training images, eight validation images, and eight reserved test images. Models were trained separately from ImageNet initialization using 94 laboratory training images plus 8, 16, or 24 field training images. On the same validation set of 73 sessions and 1,455 frames, the 24-image model with pixel-based distance classification correctly classified {full['frame_correct']} frames ({100*full['frame_accuracy']:.2f}%) and {full['session_correct']} sessions. The RGB classifier correctly classified all validation frames and sessions; therefore, these results do not establish superiority of the landmark method. Manual-coordinate diagnostics and automatic predictions support the feasibility of adapting field landmark detection, while depth-based measurement errors and generalization to new capture conditions require further study. These are checkpoint-selection validation results, not independent final test results.

## 1. 서론
{introduction}
## 2. 관련 연구
{related}
## 3. 홀 랜드마크 기반 품종 식별 방법
{method}
## 4. 데이터와 개발 검증 설계

### 4.1 현장 자료와 실물 단위 분리

수집 기간은 2026-08-27~10-29이다. 원 DB 379세션 중 기존 무효 7세션과 작업자 클래스 라벨의 오류가 육안 확인되어 정정된 11세션을 통째로 제외하였다. 정정 세션 중 6세션은 연구 대상 밖인 E23이다. 사용자는 작업자와 홀 분류가 불일치하는 사례 및 레이더 옵션을 전수 육안 확인했다. 이 연구는 라벨 정정 효과를 기여로 사용하지 않으며 레이더 옵션과 VLM은 품종 분류에 사용하지 않는다.

포함된 361세션에는 원 수집 36,142개의 RGB–깊이 쌍이 기록되어 있다. 로컬 보유 7,376쌍에서 세션당 최대 20프레임을 균등 추출하여 7,104쌍을 고정하였다. 직접 평가하는 자료는 이 로컬 미러 표본이며 원 수집 전 프레임은 아니다. 카메라는 ZED X Mini이고 메타데이터 해상도는 1920×1200, 기본 저장 간격은 2초이다. 카메라 시리얼은 한 종류이다.

2026-11-09 현장 수집 담당자는 “완전히 각각 다른 실물 도어이다. (현장 실제 작업)”라고 확인했다. 따라서 361세션은 서로 다른 도어 361개에 대응하며, 세션 분리는 실물 도어 분리이기도 하다. 동일 세션의 여러 프레임은 한 실물의 반복 관측이다. 이전 초안의 ‘동일 실물 재촬영 여부를 알 수 없다’는 제한은 이 확인으로 해소하였다. 생산 로트 식별정보와 다른 현장·카메라에 대한 평가가 없다는 제한은 별개로 남는다.

수집 기간의 자료는 과거 개발에서 관찰되었다. 실물 분리가 성립하더라도 새로운 분할만으로 전체 자료가 미관찰 블라인드 시험이 되는 것은 아니다. 특히 v1 시험 실패를 확인한 뒤 이번 적응 실험을 설계했다는 개발 순서를 공개한다.

### 4.2 고정 분할과 40장 수동 좌표

클래스별로 세션을 층화하여 고정된 다섯 그룹을 구성하였다. k번째 폴드는 그룹 k를 시험, 그룹 (k+1) mod 5를 검증, 나머지 세 그룹을 학습에 사용한다. 같은 세션의 프레임은 분할을 넘지 않는다. 세션 간 정확히 같은 RGB 해시는 발견되지 않았다. 첫 개발 폴드(Fold 0)는 학습 214세션·4,171프레임, 검증 73세션·1,455프레임, 시험 74세션·1,478프레임으로 구성된다.

현장 좌표 후보는 모델 점수를 이용하지 않고 클래스와 CV 그룹마다 한 세션, 그 세션의 가운데 추출 프레임을 선택하여 총 40장을 구성했다. 사용자가 원본을 보고 여섯 점을 직접 표시했다. 40장·240점 모두 가시 좌표가 기록되었고 범위, 원본 해시, 주석 출처 및 분할 역할 검사를 통과했다. 볼트 네 점은 순서 없는 한 채널이므로 버튼의 좌상·우상 순서는 볼트 네 점 간 대응의 정답 순서를 강제하지 않는다.

현재 폴드에서 학습 좌표 후보는 24장, 검증 좌표는 8장, 시험 좌표는 8장이다. 고정된 선택 규칙으로 클래스마다 1/2/3장을 선택해 8 ⊂ 16 ⊂ 24장의 중첩 학습 집합을 구성했다. 이번 신규 모델은 시험 1,478프레임의 특징을 추출하지 않았으며 시험 좌표도 평가하지 않았다. 다섯 폴드로 확장할 때도 40장 전체로 학습한 공통 검출기를 사용해서는 안 된다.

### 4.3 모델 학습, 보정 및 비교 조건

현장 적응 모델은 각각 실험실 94장과 현장 8/16/24장으로 새로 학습하였다. 80 epoch, batch size 4, AdamW 학습률 0.0001과 동일한 증강·히트맵 구성을 사용하였다. 현장 검증 좌표 8장의 히트맵 손실이 최소인 체크포인트를 선택했다. 실험실 검증 24장은 모니터링용이다. 현장 0장 대조군은 기존 실험실 전용 모델을 보존했으며 그 학습과 체크포인트 선택에는 현장 좌표를 사용하지 않았다. 따라서 0장과 추가 주석 모델의 차이에는 현장 학습 감독뿐 아니라 현장 검증에 의한 모델 선택도 포함된다.

| 현장 학습 좌표 | 실험실 학습 | 현장 검증 좌표 | 실험실 검증 역할 | 최적화 스텝 |
|---:|---:|---:|---|---:|
| 8장 | 94장 | 8장 | 모니터링 24장 | 8160 |
| 16장 | 94장 | 8장 | 모니터링 24장 | 8800 |
| 24장 | 94장 | 8장 | 모니터링 24장 | 9440 |

80 epoch는 같지만 표본 수에 따라 최적화 스텝이 달라진다. 8/16/24장이라는 수는 현장 학습 좌표 영상 수이며 모든 감독 정보의 총량은 아니다. 거리 스케일 보정에는 각 모델의 예측 좌표와 전체 학습 세션의 기존 품종 라벨을 사용했다. 수동 좌표 진단의 보정값은 자동 모델에 주입하지 않았다. 기존 운영 가중치, 과거 현장 좌표 16장, 자동 생성 좌표, VLM은 사용하지 않았다.

외관 비교는 기존 Fold 0/seed 42의 RGB 및 RGBE 모델이 저장한 동일 검증 프레임 예측을 사용한다. 두 분류기는 현장 학습 세션의 품종 라벨로 학습했으므로 현장 영상 사용량과 감독 형태가 홀 검출기와 다르다. 홀 방식은 CAD와 점 좌표를 추가로 사용한다. 현재 비교는 시스템 수준 비교이며 동일 라벨 비용 또는 동일 입력 해상도의 표현 비교라고 주장하지 않는다.

### 4.4 지표와 해석 범위

전체 정확도는 보류와 실패를 포함한 전체 입력을 분모로 삼는다. 판정 비율(coverage)은 품종을 출력한 입력의 비율이며, 수락된 입력에서의 정확도와 구분한다. 세션 예측은 수락된 프레임 예측의 최빈값으로 정하고 동률 또는 모두 보류이면 세션도 보류한다. 이 집계는 최대 20프레임의 오프라인 결과이며 실시간 단일 프레임 판정 성능이 아니다.

점 국소화는 현장 검증 8장의 가시점 48개에 대해 원본 해상도 8픽셀 이내의 예측 비율(PCK8)로 평가한다. 미검출도 실패로 세고 볼트 네 점은 최소 거리 대응을 사용한다. 현장 적응 모델은 이 검증 좌표로 체크포인트를 선택했으므로 PCK8도 독립 시험 수치가 아니다. CAD 잔차는 추정 거리와 클래스의 설계 거리 차이이며 실물 직접 측정에 대한 계측 오차가 아니다.

한 폴드·한 시드의 개발 결과에 통계적 우월성을 부여하지 않는다. 향후 OOF 집계에서는 시드별로 각 세션의 시험 예측을 한 번만 사용하고, 다른 시드의 동일 세션을 독립 표본으로 합치지 않는다. 실물 도어는 세션마다 다르지만 생산 로트·수집일의 공통 조건과 교차검증 학습 집합의 중첩은 불확실성 해석에 고려해야 한다.

## 5. 개발 검증 결과와 논의

### 5.1 사람의 좌표를 입력한 기하 진단

수동 좌표를 이용하여 학습 24장에서만 스케일을 추정하고 검증 8장을 판정했다. 픽셀 방식은 8/8 정답, 깊이 방식은 7/8 정답과 1장 보류였다. 픽셀 CAD 절대 잔차는 중앙값 1.84 mm, 최대 4.52 mm였고, 깊이 방식은 중앙값 4.09 mm, 최대 31.32 mm였다. 학습·검증 32장 모두 기존 기하 검사를 통과했다. 시험 좌표는 이 진단에 사용하지 않았다.

이 결과는 정확한 점이 주어졌을 때 해당 검증 표본을 픽셀 거리로 구분할 수 있음을 보여준다. 자동 검출 성능이나 정밀 계측 정확도를 뜻하지 않는다. 깊이 방식의 큰 잔차는 사람이 좌표를 지정해도 일부 남았으므로 검출기 학습만으로 모든 거리 오차가 제거된다고 기대해서는 안 된다.

### 5.2 현장 학습 좌표 수에 따른 자동 판정

아래 표는 모든 방법에 동일한 개발 검증 73세션·1,455프레임을 적용한 결과이다. 새로운 시험 결과가 아니다.

| 현장 학습 좌표(장) | 거리 방식 | 프레임 정답 | 프레임 판정 비율 | 세션 정답 | 세션 판정 비율 | 세션 macro-F1 |
|---:|---|---|---:|---|---:|---:|
{table}

동일 검증 자료의 RGB와 RGBE는 각각 1,455/1,455프레임과 73/73세션 정답이었다. 현장 적응으로 홀 방식의 판정이 회복됐다는 관측과 RGB보다 우수하다는 주장은 별개이다. 정확도가 같은 세션 결과만으로 두 방법의 동등성이나 비열등성까지 입증한 것도 아니다.

8장 모델도 픽셀 방식에서 73/73세션을 맞혔다. 16장·24장으로 늘려도 세션 성능은 같고, 24장 모델은 프레임 정답이 한 장 더 많았다. 따라서 이 개발 결과에서는 주석량 증가의 일관된 이점을 관찰하지 못했으며 24장을 최적 학습량으로 선택할 근거도 없다. 8장의 충분성을 다른 폴드와 촬영 조건에 일반화하려면 별도 검증이 필요하다.

### 5.3 홀 위치 복원과 실패 단계의 분리

| 현장 학습 좌표(장) | 볼트 PCK8 | 힌지 PCK8 | 래치 PCK8 | 전체 PCK8 |
|---:|---:|---:|---:|---:|
{chr(10).join(loc_rows)}

실험실 전용 대조군에서는 현장 검증 영상의 볼트 홀은 복원됐지만 힌지·래치 기준 홀은 충분히 검출되지 않았다. 24장 추가 모델은 검증 점 48개 모두를 원본 8픽셀 이내에서 검출했다. 24장 모델의 전체 검증 프레임에서 기하 검사 통과는 1,441건이며, 기준 홀 누락 12건과 볼트 배열 부적합 2건은 보류했다. 이 14건을 제외한 픽셀 판정에서 오분류는 없었다. 이는 한정된 현장 감독으로 해당 검출 문제가 회복될 수 있다는 개발 근거이다.

동일한 예측 좌표를 사용하더라도 깊이 방식은 추가 오분류와 보류를 보였다. 따라서 검출, 깊이 평면 추정, 거리 보정 및 CAD 경계 판정을 단계별로 분석해야 한다. 현장 적응 결과만으로 촬영 거리·자세 변화나 다른 조명에서도 픽셀 스케일 방식이 더 유리하다고 일반화할 수 없다. 그런 조건 변화는 후속 시험 항목이다.

### 5.4 기존 시험과 현재 개발 결과의 구분

v1에서 이미 완료한 최초 시험은 74세션·1,478프레임이다. RGB의 세션 정확도는 100%, RGBE는 98.65%, 실험실 전용 홀 방식은 모두 보류, 고정 프롬프트 MobileSAM 외형 치수 기준선은 35.14%였다. 이 실패를 보고 현재 후속 실험을 설계했으며 기존 시험 결과를 지우거나 현재 검증 결과와 한 표의 동등한 시험 결과로 섞지 않았다.

현재 새로운 적응 모델의 시험 성능, 5폴드×3시드 OOF, 시간 분할 및 모델 고정 이후 새 현장 평가는 미완료이다. 과거 단일 폴드의 약한 MobileSAM 기준선만으로 분할 방법 일반에 대한 우월성을 주장하지 않는다. 관련 비교가 논문의 주요 주장이라면 수동 마스크나 적절한 프롬프트를 포함한 더 충실한 분할 비교가 필요하다.

### 5.5 주석 비용, 처리 시간 및 한계

이번 40장 주석의 실제 작업 시간을 측정하지 않았다. 저장 시각 차이는 휴식·다른 작업을 포함할 수 있어 주석 시간으로 대체하지 않는다. 따라서 ‘소수의 좌표 영상으로 개발 검증 성능을 회복했다’는 관측은 가능하지만 ‘RGB보다 라벨 비용이 낮다’는 주장은 근거가 없다. 그 주장을 검증하려면 동일한 현장 영상 예산의 RGB 학습과 점·클래스 라벨의 작업 시간 및 비용을 함께 측정해야 한다.

학습 및 특징 추출은 RTX 5090을 다른 작업과 공유하며 실행했다. 이번 실행 시간은 처리 지연 비교표로 사용하지 않는다. 실시간성을 주장하려면 모델·기하·클래스 판정까지 동일한 경계로 동기화하여 측정하고, 실제 카메라 취득부터 최종 판정까지의 지연, 오판 및 보류를 함께 평가해야 한다.

본 결과는 현장 검증 8장으로 체크포인트를 선택한 한 폴드·한 시드의 후향적 개발 결과이다. 전체 검증에는 73개의 다른 실물이 포함되지만 조건의 범위는 단일 현장·카메라에 한정된다. 특히 E25 LH FRT는 전체 데이터에서도 9세션뿐이다. 생산 로트별 분리와 외부 현장 일반화는 확인되지 않았다. 0장 대조군과 추가 주석 모델의 체크포인트 선택 자료가 다르고, 8/16/24장 간 최적화 스텝도 달라 학습량 효과를 완전히 분리하지 못했다.

홀 검출 자체는 RGB 학습 모델이므로 외관 변화와 무관한 방법이 아니다. 단일 CAD 거리로 품종을 구분하는 현재 규칙은 동일 거리를 갖는 신규 품종에 바로 확장되지 않는다. 제조 공차, 실물 치수·두께 편차, 센서 설정 이력 및 조명 조건의 정량 기록도 보완해야 한다. 이러한 미확인 사항은 361세션의 실물 분리가 확인됐다는 사실과 구분한다.

## 6. 결론

서로 다른 실물 도어 361개의 세션을 구분한 연구 자료와 40장의 수동 현장 좌표를 이용하여 홀 기반 도어 식별의 현장 적응을 개발 검증하였다. 실험실 전용 모델의 기준 홀 검출 실패와, 현장 좌표를 추가한 모델의 검출·판정 회복을 같은 개발 검증 입력에서 확인했다. 사람의 좌표 진단은 검출 실패와 거리 추정 오차를 분리하는 데 사용하였다.

24장 추가 모델의 픽셀 판정은 개발 검증에서 {full['frame_correct']}/1,455프레임과 {full['session_correct']}/73세션 정답이었다. 기존 RGB 비교군도 높은 성능을 보였으므로 현재 결론은 홀 방법의 현장 적용 가능성과 남아 있는 오차 구조에 한정한다. 전체 교차검증, 조건 변화 평가 및 모델 고정 이후 전향적 검증을 완료하기 전에는 방법의 우월성, 낮은 라벨 비용 또는 실시간 현장 신뢰성이 입증되었다고 주장하지 않는다.

## 참고문헌
{refs}
## 작성자 메모 — 투고 원고에서 삭제

이 버전에서 주장할 수 있는 내용은 ‘실험실 전용 홀 검출의 현장 실패가 수동 현장 좌표 추가와 현장 검증 선택으로 회복되었고, 검출 이후 픽셀·깊이 방식의 오차가 달랐다’이다. 아직 ‘RGB보다 강건하다’, ‘더 적은 라벨 비용’, ‘깊이를 쓰면 더 정확하다’, ‘새 현장에서 검증 완료’는 주장할 수 없다.

다음 단계는 개발 결과를 근거로 주 가설과 비교 조건을 고정하고, 폴드마다 재학습·보정하는 전체 반복 평가를 수행하는 것이다. 특히 조명·배경·자세·촬영 거리 변화에서 RGB와 픽셀/깊이 방식의 실패가 달라지는지 확인해야 한다. 외관 변화 강건성이나 비용 절감이 실제로 나오지 않으면 그 주장을 제목과 기여에서 제외한다. 새 자료는 지금 수집하되 개발 중 확인한 자료와 최종 모델 고정 후의 시험 자료를 구분한다.

결과 원본: reports/field_v2_development_comparison/comparison.json, 각 reports/holes_field*_f0_s42_v2_validation/, reports/field40_manual_f0_v1/. 실물 분리 확인: reports/field_sampling_confirmation_20261109.json. 수동 라벨: data/annotation_snapshots/field40_20261109_v1/. 개발 프로토콜: config/field_adaptation_v2_protocol.json.
'''
    output.write_text(text, encoding='utf8')
    markdown_document(text, word, [(report / 'field_budget_comparison.png',
        '그림 1. 현장 학습 좌표 수에 따른 개발 검증 프레임 정확도와 판정 비율. 시험 또는 OOF 성능이 아니다.')])
    dump('reports/manuscript_v2_build.json', {
        'created_at': now(), 'source_comparison_sha256': sha256(report / 'comparison.json'),
        'sampling_confirmation_sha256': sha256(ROOT / 'reports/field_sampling_confirmation_20261109.json'),
        'manuscript_markdown': str(output.relative_to(ROOT)), 'markdown_sha256': sha256(output),
        'manuscript_docx': str(word.relative_to(ROOT)), 'docx_sha256': sha256(word),
        'v1_preserved': True, 'scope': 'Development validation only; final test and prospective evaluation pending.'})
    print(str(word), flush=True)


if __name__ == '__main__':
    build()
