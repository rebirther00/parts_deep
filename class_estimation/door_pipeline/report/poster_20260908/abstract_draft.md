# 학회 초록 초안 (2026-09-08, v1)

대상: 한국자동차공학회 어스무빙머시너리 부문(1순위) / 유공압건설기계학회(2순위). 학회별 분량·형식 확정 후 조정.
저자 정보는 `paper/KCI_논문_초고_RGBE_강건성_v6.md` 기준.

---

## 국문

**굴착기 도어 용접 공정 자동화를 위한 현장 데이터 수집·검증 체계 구축 사례**

이민수<sup>1#</sup>
<sup>1</sup> 한국건설기계부품연구원

**초록**

굴착기 도어 용접 공정은 현재 작업자가 부품 기종을 판단하여 지그에 장착하는 수동 공정이며, 비전 기반 자동 판별과 로봇 연계로 전환하려면
현장 조건에서 확보한 데이터와 신뢰할 수 있는 라벨, 그리고 현장 성능을 과장하지 않는 평가 기준이 선행되어야 한다. 본 연구는 협력사 용접
현장에 RGB-D 카메라(ZED X Mini)와 엣지 캡처 장치를 설치하고, 작업자가 기종을 선택하면 2초 간격으로 세션 단위 영상과 메타데이터를 자동
저장하여 NAS 정본과 학습 서버 데이터베이스로 연동하는 수집 체계를 구축하였다. 2주간 도어 9종 84세션 약 8,000쌍을 확보하였으며, 프레임
좌표에서 모서리 프레임 홀 간 거리를 측정하는 기하 기반 판별기로 세션 단위 라벨을 자동 검증하여 작업자 오선택 3세션과 기종 목록에 없던
신규 부품(E23) 4세션을 발견·정정하였다. 연속 프레임의 중복을 고려하여 세션 단위 분할과 세션 그룹 교차검증을 평가 기준으로 정의하였다.
기하 기반 판별기는 실험실 영상 위주 134장의 6점 홀 라벨만으로 학습하고 현장 추가 라벨링 없이 현장 1,699장에서 판정 정확도 99.9%를
보였으며, 신규 부품은 CAD 치수 등록만으로 즉시 판별되었다. 외관 기반 CNN(ResNet18, RGB+Edge)은 현장 데이터 세션 교차검증에서
97.4%를 보여 확보한 데이터가 학습 데이터로도 충분함을 확인하였다. 오판은 촬영 프로토콜 정착 전 세션에 집중되어 프로토콜 개선점으로
반영하였다. 구축한 체계는 엣지 자동 판정, 6자유도 자세 추정 연계, 로봇 자동 공정으로 확장할 기반이 된다.

**주요어**: 굴착기 도어, 용접 공정 자동화, 현장 데이터 수집, 라벨 검증, 세션 단위 평가, 부품 판별

---

## English

**A Case Study on Building an On-site Data Collection and Validation System for Automating the Excavator Door Welding Process**

Minsu Lee<sup>1#</sup>
<sup>1</sup> Korea Construction Equipment Technology Institute (KOCETI)

**Abstract**

The excavator door welding process is currently manual: an operator identifies the part model and mounts it on the jig. Moving to vision-based
part identification and robotic handling requires data captured under real shop-floor conditions, trustworthy labels, and an evaluation protocol
that does not overstate field performance. We installed an RGB-D camera (ZED X Mini) with an edge capture device at a supplier's welding line;
once the operator selects the model, images and metadata are recorded every two seconds as sessions and synchronized to a NAS master copy and a
training-server database. In two weeks we collected about 8,000 RGB-D pairs over 84 sessions of nine door models. A geometry-based classifier
that measures the distance between two corner frame holes verified labels at the session level, revealing three operator mislabels and four
sessions of a model absent from the list (E23), which were corrected. Because consecutive frames are near-duplicates, we defined session-level
splitting and session-grouped cross-validation as the evaluation protocol. Trained on only 134 images with six hole landmarks each, mostly from
laboratory captures, the geometry-based classifier reached 99.9% decision accuracy on 1,699 field images without additional field labeling, and the
new model was supported by registering one CAD dimension. An appearance-based CNN (ResNet18, RGB+Edge) achieved 97.4% under session-grouped
cross-validation, confirming that the collected data is sufficient for training. Errors concentrated in sessions captured before the imaging
protocol was settled and were fed back into the protocol. The system provides the basis for edge-side automatic identification, 6-DoF pose
estimation, and robotic automation.

**Keywords**: excavator door, welding automation, on-site data collection, label validation, session-level evaluation, part identification
