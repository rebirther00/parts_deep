const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, ImageRun, Table, TableRow, TableCell,
  WidthType, AlignmentType, BorderStyle, ShadingType, PageBreak, TableOfContents, LevelFormat,
  Footer, Header, PageNumber, VerticalAlign,
} = require('docx');

const FIG = path.join(__dirname, 'fig2');
const OUT = process.argv[2] || path.join(__dirname, 'hole_report.docx');
const FONT = '맑은 고딕';
const BLUE = '2F6DB5';
const PAGE_W = 11906 - 1701 * 2;

const run = (t, o = {}) => new TextRun({ text: t, font: FONT, size: o.size || 22, bold: o.bold, color: o.color, italics: o.italics });
const p = (t, o = {}) => new Paragraph({ alignment: o.align, spacing: { after: o.after ?? 120, line: 320 }, children: Array.isArray(t) ? t : [run(t, o)] });
const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 360, after: 160 }, children: [new TextRun({ text: t, font: FONT, size: 32, bold: true, color: BLUE })] });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 280, after: 120 }, children: [new TextRun({ text: t, font: FONT, size: 26, bold: true, color: '1F4E79' })] });
const h3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, spacing: { before: 200, after: 80 }, children: [new TextRun({ text: t, font: FONT, size: 23, bold: true })] });
const bullet = (t) => new Paragraph({ numbering: { reference: 'bul', level: 0 }, spacing: { after: 60, line: 300 }, children: Array.isArray(t) ? t : [run(t)] });
const num = (t) => new Paragraph({ numbering: { reference: 'num', level: 0 }, spacing: { after: 60, line: 300 }, children: Array.isArray(t) ? t : [run(t)] });
const note = (t) => new Paragraph({ spacing: { before: 80, after: 160 }, indent: { left: 200 }, border: { left: { style: BorderStyle.SINGLE, size: 18, color: BLUE, space: 8 } }, shading: { type: ShadingType.CLEAR, fill: 'EAF2FB', color: 'auto' }, children: [run(t, { size: 20 })] });
const code = (lines) => new Table({ width: { size: PAGE_W, type: WidthType.DXA }, columnWidths: [PAGE_W],
  borders: { top: { style: BorderStyle.NONE }, bottom: { style: BorderStyle.NONE }, left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE }, insideH: { style: BorderStyle.NONE }, insideV: { style: BorderStyle.NONE } },
  rows: [new TableRow({ children: [new TableCell({ width: { size: PAGE_W, type: WidthType.DXA }, shading: { type: ShadingType.CLEAR, fill: 'F3F3F3', color: 'auto' }, margins: { top: 100, bottom: 100, left: 160, right: 160 },
    children: lines.map((l) => new Paragraph({ spacing: { after: 0, line: 260 }, children: [new TextRun({ text: l, font: 'Consolas', size: 18 })] })) })] })] });
const img = (file, widthPx, caption) => {
  const data = fs.readFileSync(path.join(FIG, file));
  const w = data.readUInt32BE(16), h = data.readUInt32BE(20);
  const out = [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, children: [new ImageRun({ type: 'png', data, transformation: { width: widthPx, height: Math.round(widthPx * h / w) } })] })];
  if (caption) out.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [run(caption, { size: 18, color: '555555' })] }));
  return out;
};
const table = (headers, rows, widths, opts = {}) => {
  const total = widths.reduce((a, b) => a + b, 0); const cw = widths.map((w) => Math.round(PAGE_W * w / total));
  const cell = (t, i, hdr) => new TableCell({ width: { size: cw[i], type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
    shading: hdr ? { type: ShadingType.CLEAR, fill: 'D9E2F3', color: 'auto' } : undefined, margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ spacing: { after: 0, line: 280 }, alignment: opts.center && i > 0 && !hdr ? AlignmentType.CENTER : undefined,
      children: [new TextRun({ text: String(t), font: opts.mono && i === 0 && !hdr ? 'Consolas' : FONT, size: 19, bold: hdr })] })] });
  return new Table({ width: { size: PAGE_W, type: WidthType.DXA }, columnWidths: cw,
    rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, i, true)) }), ...rows.map((r) => new TableRow({ children: r.map((c, i) => cell(c, i, false)) }))] });
};
const gap = (n = 120) => new Paragraph({ spacing: { after: n }, children: [] });
const pb = () => new Paragraph({ children: [new PageBreak()] });

const cover = [
  gap(2400),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [new TextRun({ text: '홀 랜드마크 기반 도어 판별기', font: FONT, size: 40, bold: true, color: BLUE })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [new TextRun({ text: '학습 기법 · 평가 결과 · 사용법 · 향후 계획', font: FONT, size: 30, bold: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 }, children: [new TextRun({ text: 'CNN 랜드마크 검출 + CAD 치수 기반 기하 판정', font: FONT, size: 26, color: '555555' })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: BLUE, space: 4 } }, spacing: { after: 400 }, children: [] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [run('부품 인식 AI — 굴착기 도어 분류 과제', { size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [run('작성일: 2026년 8월 26일', { size: 24 })] }),
  pb(),
  new Paragraph({ children: [new TextRun({ text: '목 차', font: FONT, size: 30, bold: true, color: BLUE })], spacing: { after: 200 } }),
  new TableOfContents('목차', { hyperlink: true, headingStyleRange: '1-2' }),
  pb(),
];

const s1 = [
  h1('1. 요약'),
  p('굴착기 도어 8종은 형상이 거의 같고 치수(폭 830~1458mm)만 다르다. 형상을 학습하는 CNN 분류기는 사무실 데이터에서 100%였으나 현장에서 14%로 무너졌다. 본 보고서는 이를 대체하는 새 방식 — 도어 상단 모서리 프레임 홀 2개의 거리 D를 재서 CAD 치수와 대조하는 기하 판정 — 의 학습 기법과 평가 결과를 정리한다. AI(CNN)는 형상 분류가 아니라 "홀 위치 검출"에만 쓰고, 판정은 설명 가능한 치수 비교로 한다.'),
  table(['항목', '결과'], [
    ['홀 검출 정확도 (홀드아웃 27장)', '6점 모두 중앙값 1.0~1.4px, 4px 이내 100%'],
    ['공식 test 분할 (189장, 라벨 학습분 제외)', '판정 134장 중 134 정답 — 100%'],
    ['사무실 전체 (1,162장, 라벨 학습분 제외)', '판정 814장 중 806 정답 — 99.0% (그룹 99.5%)'],
    ['현장 세션 (경주, 16장)', '16/16 정답, D 885.9mm (CAD 886)'],
    ['라벨링 비용', '134장 × 6점, 약 1시간 (전용 브라우저 도구)'],
    ['추론 속도', 'RTX 5090에서 홀 단계 41ms/프레임'],
  ], [5, 8]),
  gap(),
  note('보류(판정 안 함) 비율은 사무실 데이터에서 약 30%다. 힌지측 홀이 프레임 밖이거나 이미지 경계에 붙은 경우로, 도어를 화면에 꽉 채워 찍은 사무실 촬영 방식 때문이다. 카메라가 지그에 고정된 현장에서는 보류 0%였다.'),
];

const s2 = [
  h1('2. 판별 원리'),
  h2('2.1 왜 홀 거리인가'),
  bullet('8종 모두 좌우 세로 프레임 상단에 Ø6mm 홀이 있고, 위치가 가장자리에서 52~54mm·상단에서 105mm로 동일하다 (CAD 8종 분석). 따라서 두 홀의 거리 D = 도어 폭 − 106mm.'),
  bullet('8종의 폭이 모두 달라 D만으로 분류된다. 최소 간격은 FRT 3종의 41~47mm, 나머지는 50~200mm.'),
  bullet('홀 중심은 클램프가 외곽을 훼손해도 살아남는 랜드마크이고, 조명·배경·녹과 무관하다.'),
  bullet('중앙 보강대의 패드 홀은 옵션/리비전에 따라 패턴(2+2 vs 1+3)이 바뀌어 사용하지 않는다. E25_RH STL의 패드는 실물과 다름이 확인됐다.'),
  ...img('corner_marked_E25_door_RH.png', 300, '그림 1. CAD 내판 렌더 — 판별에 쓰는 상단 모서리 홀 2개 (E25_RH, D=886mm)'),
  table(['클래스', '폭(mm)', 'CAD D(mm)', '사람 라벨 실측 D 중앙값', '차이'], [
    ['E25_door_LH_FRT', '830', '724', '721', '−3'], ['E30_door_LH_FRT', '871', '765', '761', '−4'], ['E38_door_LH_FRT', '918', '812', '807', '−5'],
    ['E25_door_RH', '991', '886', '886', '0'], ['E25_door_LH_RR', '1143', '1037', '1032', '−5'], ['E30_E38_door_RH', '1193', '1087', '1080', '−7'],
    ['E30_door_LH_RR', '1263', '1158', '1154', '−4'], ['E38_door_LH_RR', '1458', '1352', '1355', '+3'],
  ], [4, 2, 2.5, 4, 1.5], { center: true, mono: true }),
  gap(),
  p('표 1. CAD와 사람 라벨(GT) 기준 실측 D — 8종 모두 ±7mm 이내. 홀만 정확히 찾으면 GT 기준 최근접 분류 87/87 = 100%.', { size: 20, color: '555555' }),
  h2('2.2 판정 파이프라인'),
  num('CNN이 원본 프레임에서 6점을 검출: 래치 볼트홀 4개(각공 주변) + 상단 모서리 홀 2개(힌지측·래치측).'),
  num('depth로 도어 평면을 피팅하고, 두 모서리 홀을 평면에 투영해 mm 거리 D를 구한다. 근사 intrinsics의 일정 편향은 카메라 모드별 상수 K_DEPTH(1080p 0.8235, 1200p 0.8505)로 보정.'),
  num('볼트홀 4개로 만든 국소 좌표계(157×96mm 사각형)로 기하 일관성을 검사한다: 래치측 홀이 볼트 사각형 기준 (±160, ±190)mm 위치인지, 두 홀이 같은 선상인지, 서로 반대편인지, D가 600~1500mm인지. 하나라도 어긋나면 판정을 보류한다.'),
  num('통과 시 CAD D 표에서 최근접 클래스. 실시간에서는 10프레임 윈도의 판정 프레임 D 중앙값으로 집계하고, 판정 프레임이 3개 미만이면 기존 속성 파이프라인(통풍구 U-Net)으로 폴백, 두 방식의 그룹이 어긋나면 "보류"를 출력한다.'),
  ...img('measure_corner03.png', 520, '그림 2. 현장 프레임(20260810 E25_RH)의 홀 쌍과 D — 실측 891mm vs CAD 886mm'),
];

const s3 = [
  h1('3. 학습 기법'),
  h2('3.1 데이터와 라벨링'),
  bullet('라벨 134장: 사무실 datasets 8종 × 15장(클래스 내 10장 간격 균등 추출) 118장 + 현장 E25_RH 세션 16장. 이미지당 6점, 원본 픽셀 좌표, 가려진 점은 "안 보임" 표시.'),
  bullet('전용 브라우저 도구(tools/label_holes.py)로 클릭 라벨링: 6배 확대경, 점 순서 자동 진행, 자동 저장. 134장 × 6점을 약 1시간에 완료.'),
  bullet('분할: 클래스별 층화 20% 홀드아웃 (27장), 나머지 107장 학습. 홀드아웃은 학습 중 5 epoch마다 GT로 평가.'),
  ...img('label_ui.png', 560, '그림 3. 라벨링 도구 화면 — 왼쪽 점 목록/진행률, 오른쪽 위 확대경'),
  h2('3.2 모델과 학습 목표'),
  table(['항목', '설정', '이유'], [
    ['입력', '1280×768 letterbox (원본 1920×1080/1200)', 'Ø6mm 홀(원본 4~5px)이 3px 이상 남는 최소 해상도'],
    ['백본', 'ResNet18 (ImageNet 사전학습)', '107장의 소량 데이터에서 사전학습이 필수. Jetson 배포 고려'],
    ['디코더', 'FPN식 측면 연결 → stride 4 히트맵', '작은 홀의 위치 정밀도(1px)와 도어 전체 문맥을 함께 확보'],
    ['출력 채널', 'bolt(4 피크) / corner_hinge / corner_latch', '볼트를 좌상·우상으로 구분하지 않아 회전 불변 — 사각형 기하는 검출 후에 맞춤'],
    ['타깃', 'σ=2 가우시안 히트맵 (스트라이드 4 좌표)', '위치 회귀보다 다중 피크·안정 수렴에 유리'],
    ['손실', '가중 MSE (양성 픽셀 ×21)', '히트맵이 대부분 0이라 단순 MSE는 모두 0으로 붕괴'],
    ['피크 추출', '5px NMS + 포물선 서브픽셀 보정', '히트맵 격자(4px) 이하 정밀도'],
  ], [2.5, 5, 6]),
  gap(),
  h2('3.3 증강'),
  p('도어는 바닥/지그에 임의 각도로 놓이므로 회전 불변이 핵심이다. 학습 샘플마다 임의 아핀: 회전 ±180°, 스케일 0.75~1.15, 이동 ±60px, 50% 좌우 플립, 밝기·대비(0.7~1.3, ±30), 30% 확률 블러. 에폭당 원본 4회 재샘플.'),
  h2('3.4 최적화와 발산 사례'),
  bullet('1차 시도: AdamW, OneCycle max_lr 3e-4, FP16 autocast, 120 epoch → 20 epoch까지 좋았으나(홀드아웃 ≤8px 88%) 이후 손실이 0.0018→0.0047로 튀며 발산, 회복 못 함. 원인은 OneCycle 정점 LR과 FP16의 결합으로 판단.'),
  bullet('2차(채택): AdamW lr 1e-4, 3 epoch 워밍업 + 코사인 감쇠, gradient clip 1.0, FP32, 80 epoch. 30 epoch부터 홀드아웃 ≤8px 100%, 발산 없음.'),
  ...img('training_curve.png', 520, '그림 4. 2차 학습 곡선 — 손실(로그)과 홀드아웃 8px 이내 비율 (DB training_metrics에서 생성)'),
  h2('3.5 왜 규칙 기반 검출은 폐기했나'),
  p('학습 전에 블랙햇 모폴로지·NCC 템플릿으로 홀을 찾는 규칙 기반 검출을 3차례 시도했다. 자기 보고 지표로는 97.7%까지 나왔으나, 사람 라벨(GT)로 재보니 방향(뒤집힘) 판정 48%, 볼트홀 오검출 53%, 모서리 홀 오검출 60%였다. 녹·접힘선·그림자를 홀로 잡는 문제는 임계값 조정으로 해결되지 않았다. 학습 검출기는 같은 GT에서 오검출 0%다. 교훈: 검출 정확도는 반드시 GT로 재야 하며, 라벨은 학습 데이터이자 평가 기준이다.'),
];

const s4 = [
  h1('4. 평가 결과'),
  h2('4.1 홀 검출 정확도 (GT 대비)'),
  table(['채널', '홀드아웃 27장 중앙값', '≤4px', '현장 16장 중앙값', '≤4px', '오검출(>20px)'], [
    ['볼트홀 (×4)', '1.0 px', '100%', '0.7 px', '98%', '0%'], ['모서리 홀 힌지측', '1.4 px', '100%', '1.1 px', '100%', '0%'], ['모서리 홀 래치측', '1.2 px', '100%', '1.4 px', '100%', '0%'],
  ], [3.5, 3, 1.5, 3, 1.5, 2.5], { center: true }),
  gap(),
  p('원본 해상도 0.75~0.83 px/mm에서 1px ≈ 1.3mm. D 오차로 환산하면 ±2~3mm로, 클래스 간 최소 간격 41mm의 1/10 이하다.', { size: 20 }),
  h2('4.2 판정 정확도'),
  table(['평가 세트', '전체', '판정', '정답', '판정 정확도', '그룹 정확도', '보류 사유(상위)'], [
    ['test 분할 (seed42)', '189', '134 (71%)', '134', '100%', '100%', 'no_corner 22, latch_offset 10, same_side 10'],
    ['datasets 전체', '1,162', '814 (70%)', '806', '99.0%', '99.5%', 'no_corner 185, near_border 56, same_side 46'],
    ['현장 E25_RH', '16', '16 (100%)', '16', '100%', '100%', '—'],
  ], [3, 1.5, 2, 1.5, 2, 2, 5], { center: true }),
  gap(),
  p('표 3. 라벨에 쓴 107장은 모든 세트에서 제외. 보류는 판정을 내리지 않은 것으로, 오판과 구분된다.', { size: 20, color: '555555' }),
  table(['클래스', 'test 판정/정답', 'datasets 판정/정답', 'D 중앙값', 'CAD', '오판 내역'], [
    ['E25_door_LH_FRT', '18/18', '116/116', '724', '724', ''], ['E30_door_LH_FRT', '17/17', '97/95', '767', '765', '→E25_LH_FRT 2'], ['E38_door_LH_FRT', '15/15', '94/92', '810', '812', '→E30_LH_FRT 2'],
    ['E25_door_RH', '23/23', '129/129', '889', '886', ''], ['E25_door_LH_RR', '16/16', '97/97', '1035', '1037', ''], ['E30_E38_door_RH', '14/14', '101/97', '1091', '1087', '→E25_LH_RR 3, →E30_LH_RR 1'],
    ['E30_door_LH_RR', '18/18', '101/101', '1162', '1158', ''], ['E38_door_LH_RR', '13/13', '79/79', '1356', '1352', ''],
  ], [4, 2.5, 3, 2, 1.5, 4], { center: true, mono: true }),
  gap(),
  h2('4.3 성공 샘플'),
  ...img('montage_success.png', 620, '그림 5. 클래스별 성공 예 — 파란 원 볼트홀 4, 빨강/주황 원 모서리 홀, 빨간 선 D'),
  ...img('sample_detections_cnn.png', 620, '그림 6. GT(초록 링)와 검출(색 원) 비교 — 좌하는 하단 홀을 잡아 보류된 예, 우하는 현장'),
  h2('4.4 실패 샘플과 원인'),
  ...img('montage_failure.png', 620, '그림 7. 오판 8건 전부 — 빨간 선이 도어 세로 방향으로 그어짐'),
  p('오판 8건(0.7%)은 모두 한 유형이다. 힌지측 상단 홀이 프레임 밖(또는 경계)일 때 검출기가 같은 프레임의 하단 모서리 홀을 힌지측 홀로 잡고, 래치측 상단 홀과의 거리(도어 높이 방향, ≈930~1140mm)를 D로 계산해 RR/RH 계열로 오판한다. 상·하단 홀은 국소 형상이 동일해 검출기 단독으로는 구분할 수 없고, 기하 게이트(같은 선상 검사)가 대부분 걸러내지만 볼트 프레임이 부정확한 경우 통과한다.'),
  h2('4.5 보류 샘플'),
  ...img('montage_abstain.png', 560, '그림 8. 보류 사유별 예 — no_corner(홀 미검출), near_border, same_side(하단 홀 검출을 게이트가 차단), not_collinear, latch_offset, no_frame'),
];

const s5 = [
  h1('5. 사용법'),
  p('모든 스크립트는 door_pipeline 디렉터리에서 실행하며, scripts/ 래퍼가 venv 활성화와 경로를 처리한다.'),
  table(['단계', '명령', '입력 → 출력'], [
    ['라벨링', 'scripts/hole_label.sh [장수] [추가폴더]', '브라우저 :8090 → labels/holes/*.json'],
    ['학습', 'scripts/hole_train.sh [epochs=80]', 'labels/holes → attribute_models/hole_landmarks/model.pth, DB training_sessions'],
    ['평가', 'scripts/hole_evaluate.sh [폴더]', 'test 분할·datasets·현장 → eval_classifier.json, report/hole_analysis/samples/*, DB evaluation_results'],
    ['단일 이미지', 'python scripts/hole_classify_image.py rgb.png [--out r.jpg]', '판정 JSON + 표시 이미지'],
    ['실시간', 'scripts/hole_inference.sh [리플레이폴더]', 'ZED 또는 폴더 재생 → 웹 UI :5003, /api/inference_result'],
  ], [2, 5.5, 5.5]),
  gap(),
  code([
    '# 예: 현장 세션 리플레이로 확인',
    'scripts/hole_inference.sh datasets_field/E25_door_RH_s_091317',
    'curl localhost:5003/api/inference_result',
    '# → {"class":"E25_door_RH","source":"hole","hole":{"D_mm":885.3,"n_judged":10,"gate":"ok",...}}',
    '',
    '# 새 클래스/리비전 추가: CAD에서 D 하나 뽑아 hole_classifier.CAD_D 에 추가 (학습 불필요)',
    '# 새 카메라 모드: 라벨 10장으로 K_DEPTH 캘리브레이션 (16_evaluate 결과의 D 중앙값 / CAD)',
  ]),
  gap(),
  h2('5.1 실시간 API 필드'),
  table(['필드', '의미'], [
    ['class / group / confidence', '최종 판정 (source가 hole이면 판정 프레임 비율, attr이면 속성 파이프라인 확률)'],
    ['source', 'hole(홀 판별기) / attr(속성 파이프라인 폴백) / conflict(두 방식 그룹 불일치 → 보류)'],
    ['hole.pred / D_mm / n_judged', '윈도 집계 결과: 클래스, D 중앙값, 판정 프레임 수'],
    ['hole.gate / frame_D_mm / points', '최신 프레임의 게이트 결과, D, 검출점(원본 픽셀, 점수)'],
  ], [4, 9], { mono: true }),
];

const s6 = [
  h1('6. 향후 계획'),
  table(['순위', '항목', '내용', '기대 효과'], [
    ['1', '하단 모서리 홀 채널 추가', '라벨 도구에 7·8번 점(하단 홀) 추가 완료. 134장 × 2점 클릭(~30분) 후 5채널로 재학습. 게이트에 상·하 구분 추가', '오판 8건 유형 제거, 보류율 감소'],
    ['2', '현장 다종 검증', '현장 세션이 쌓이는 대로 scripts/hole_evaluate.sh datasets_field 실행. E25_RH 외 7종의 K_DEPTH·D 확인', '현장 정확도 근거 확보'],
    ['3', 'Zed Box(Jetson) 배포', 'ResNet18-FPN 1280×768 TensorRT/FP16 변환, 속도 측정 (5090 41ms 기준). 06_factory_capture와 연동해 취득 세션마다 자동 판정 → DB 기록', '현장 무인 판정'],
    ['4', '촬영 프로토콜', '도어 전체 + 여백 확보, 힌지측 프레임 상단이 반드시 화면 안에 들어오게. 보류율을 현장 0% 수준으로 유지', '보류 최소화'],
    ['5', '스케일 이중화', 'depth 평면 + 볼트 피치(현장 해상도에서 0.4% 오차) 교차검증, 불일치 시 보류', 'depth 이상 시 강건성'],
    ['6', '속성 파이프라인 정리', '통풍구 U-Net은 그룹 폴백 전용으로 축소, CNN(RGBE) 분류기는 이 경로에서 제외', '유지보수 단순화'],
    ['7', '데이터 관리', '라벨·평가 결과·모델을 door_pipeline.db에 계속 기록(현재 학습/평가 자동 기록 중). 라벨 수 증가 시 재학습 자동화', '이력 추적'],
  ], [1, 3, 6.5, 3]),
  gap(),
  h2('6.1 알려진 제약'),
  bullet('힌지측 상단 홀이 프레임 밖이면 판정 불가 — 촬영 프로토콜로 해결하는 것이 정공법이며, 그 경우 속성 파이프라인이 그룹만 제공.'),
  bullet('K_DEPTH는 카메라 모드(세로 해상도)별 상수. 새 카메라·모드에는 캘리브레이션 필요.'),
  bullet('현장 검증은 E25_RH 1종 16장뿐. 다른 7종의 현장 성능은 사무실 결과로부터의 추정이다.'),
  h1('부록. 파일 구성'),
  table(['경로', '내용'], [
    ['hole_classifier.py', '추론 모듈 (detect / classify / aggregate, K_DEPTH, CAD_D, 기하 게이트)'],
    ['15_train_hole_landmarks.py', '학습 + 홀드아웃 평가 (DB 기록)'],
    ['16_evaluate_hole_classifier.py', '판정 평가 (test 분할·datasets·현장·임의 폴더)'],
    ['14_realtime_inference_attribute.py', '실시간 서버 — 홀 판별기 1순위, 속성 파이프라인 폴백'],
    ['tools/label_holes.py · tools/make_hole_samples.py', '라벨링 도구 · 샘플/학습곡선 생성'],
    ['scripts/hole_*.sh · scripts/hole_classify_image.py', '실행 래퍼'],
    ['labels/holes/', '라벨 134장 (원본 픽셀 좌표 JSON)'],
    ['attribute_models/hole_landmarks/', 'model.pth · split.json · eval.json · eval_classifier.json'],
    ['report/hole_analysis/', 'CAD 홀 분석, GT 검증, 샘플 이미지, 본 보고서'],
  ], [5, 8], { mono: true }),
];

const doc = new Document({
  creator: 'door_pipeline', title: '홀 랜드마크 기반 도어 판별기 보고서',
  styles: { default: { document: { run: { font: FONT, size: 22 } } }, paragraphStyles: [
    { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { font: FONT, size: 32, bold: true, color: BLUE }, paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0 } },
    { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { font: FONT, size: 26, bold: true, color: '1F4E79' }, paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 1 } },
    { id: 'Heading3', name: 'Heading 3', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { font: FONT, size: 23, bold: true }, paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 2 } } ] },
  numbering: { config: [
    { reference: 'bul', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 480, hanging: 240 } } } }] },
    { reference: 'num', levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 480, hanging: 300 } } } }] } ] },
  features: { updateFields: true },
  sections: [{ properties: { page: { margin: { top: 1560, bottom: 1440, left: 1701, right: 1701 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: 'AAAAAA', space: 2 } }, children: [run('홀 랜드마크 기반 도어 판별기 · 2026-08-26', { size: 16, color: '777777' })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18, color: '777777' })] })] }) },
    children: [...cover, ...s1, ...s2, ...s3, ...s4, ...s5, ...s6] }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log('wrote', OUT, buf.length); });
