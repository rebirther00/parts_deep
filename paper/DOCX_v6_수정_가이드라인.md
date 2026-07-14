# JKSPE_Template_KOR_v2.docx → v6 반영 수정 가이드라인

> **기준 문서**: `JKSPE_Template_KOR_v2.docx` (v5 기반)
> **반영 대상**: `KCI_논문_초고_RGBE_강건성_v6.md`
> **핵심 변경**: Texture Aug RGBD 제거 → Baseline RGB 추가, 논조 완화

---

## 전체 개요: 변경이 필요한 모든 섹션

| 섹션 | 변경 수준 | 요약 |
|------|:--------:|------|
| 제목/초록/Abstract | **전면 교체** | 4종 모델 변경 + 논조 완화 |
| 1. 서론 | **전면 교체** | 연구 목적 3개 항 모두 변경 |
| 2.1 | 유지 | 변경 없음 |
| 2.2 | **제목 변경 + 내용 수정** | "텍스처 불변 학습" → "텍스처 편향과 형상 기반 인식" |
| 2.3 | 유지 | 변경 없음 |
| 2.4 | 유지 | 변경 없음 |
| 3.1 | **수정** | Fig.1 구조도에 Baseline RGB 추가 |
| 3.2 | **수정** | RGB/Edge-only 3채널 설명 추가 |
| 3.3 (Table 1) | **전면 교체** | 4종 모델 표 완전 변경 |
| 3.3.1 | **전면 교체** | "Baseline RGBD" → "Baseline RGB" |
| 3.3.2 | **삭제** | "Texture 불변 증강 RGBD" 제거 |
| 3.3.3 | **번호 변경** | 3.3.2로 번호 변경 (내용 유지) |
| 3.3.4 | **번호 변경** | 3.3.3으로 번호 변경 (내용 유지) |
| 새 3.3.2 | **번호만** | Edge-only → 3.3.2 |
| 새 3.3.3 | **번호만** | RGBE Hybrid → 3.3.3 |
| 4장 | 유지 | 변경 없음 |
| 5.1 | **전면 교체** | 모든 숫자/모델명 변경 |
| 5.2 → v6의 5.3 | **전면 교체** | 클래스별 F1 데이터 완전 변경 |
| 5.3 → v6의 5.4 | **전면 교체** | 강건성 Δ 데이터 변경 |
| 5.4 → v6의 5.5 | **전면 교체** | 통계 검정 데이터 변경 |
| 5.5 → v6의 5.6 | **전면 교체** | 학습 수렴 데이터 변경 |
| 5.6 → v6의 5.7 | **전면 교체** | 해상도 비교 데이터 변경 |
| 5.7 → v6의 5.8 | **전면 교체** | 224 통계 데이터 변경 |
| 5.8 → v6의 5.9 | **수정** | Grad-CAM 모델명 변경 |
| 6장 | **전면 교체** | 고찰 전체 재작성 |
| 7장 | **전면 교체** | 결론 전체 재작성 |
| ACKNOWLEDGEMENT | **작성 필요** | 현재 템플릿 기본값 상태 |
| REFERENCES | 유지 | 참고문헌 동일 (양식 수정 필요 — 아래 참조) |
| 저자소개 | **작성 필요** | 현재 템플릿 기본값 상태 |

---

## 섹션별 상세 수정 가이드

### 제목

**현재 (DOCX v2)**: 제목이 내부에 있음 (확인 필요 — DOCX 상단 메타데이터)

**v6으로 변경**:
- 국문: 굴착기 부품 분류를 위한 RGB-Edge 하이브리드 입력 표현의 텍스처 변형 강건성 비교 연구
- 영문: A Comparative Study on Texture Variation Robustness of RGB-Edge Hybrid Input Representation for Excavator Part Classification

### 초록 / Abstract

v6.md의 초록/Abstract 전문으로 **전면 교체**.

핵심 변경 포인트:
- "Baseline RGBD, Texture 불변 증강 RGBD" → "Baseline RGB, RGBD"
- "우수성을 입증" → "비교 분석", "경향을 보일 수 있음"
- 통계적 유의성 부재 명시 (p > 0.05)
- 224 해상도: "Texture Aug RGBD 1위" → "RGBD 1위"

---

### 1. 서론

**변경 범위**: 본문 단락 12~13 (연구 목적 부분)

**현재 (DOCX)**:
> 첫째, ... 기존 RGBD 대비 텍스처 변형 강건성의 **우수성을 실험적으로 입증**한다.
> 둘째, Baseline RGBD, **Texture 불변 증강 RGBD**, Edge-only, RGBE Hybrid의 4종 ...

**v6으로 교체**:
> 첫째, ... 순수 RGB 및 RGBD와의 텍스처 변형 강건성을 **비교**한다.
> 둘째, **Baseline RGB**, RGBD, Edge-only, RGBE Hybrid의 4종 ...
> 셋째, ... **배포 환경에 따른 실용적 지침**을 제시한다.

**추가**: 서론 2~3번째 단락 사이에 "RGB 카메라 기반 딥러닝 분류 모델은..." 및 "그러나 텍스처 변형에 대한 강건성 관점에서..." 단락 추가 (v6.md 참조)

---

### 2.2 절 제목 변경

**현재**: `2.2 텍스처 불변 학습 및 형상 기반 인식`
**v6으로**: `2.2 텍스처 편향과 형상 기반 인식`

본문 내용도 v6.md에 맞춰 수정. Texture Aug 관련 직접 언급("본 연구의 Texture Aug RGBD 모델의 이론적 근거") 제거.

---

### 3.1 시스템 개요

**Fig. 1** 구조도에 **Baseline RGB [R,G,B]** 경로 추가 필요.
현재: RGBD, Edge-only, RGBE 3종만 표시
v6: RGB, RGBD, Edge-only, RGBE 4종 모두 표시

---

### 3.2 공통 아키텍처

**추가 문장**:
> "Baseline RGB와 Edge-only 모델은 표준 3채널 입력을 사용하므로, ImageNet 사전학습 가중치를 그대로 활용한다."

---

### 3.3 입력 표현별 모델 구성

#### Table 1 — 전면 교체

**현재 (DOCX)**:

| Model | Channels | ... |
|-------|----------|-----|
| Baseline RGBD | 4 | R,G,B,D |
| Texture Aug RGBD | 4 | R,G,B,D |
| Edge-only | 3 | E,E,E |
| RGBE Hybrid | 4 | R,G,B,E |

**v6으로 교체**:

| Model | Channels | ... |
|-------|----------|-----|
| **Baseline RGB** | **3** | **R,G,B** |
| RGBD | 4 | R,G,B,D |
| Edge-only | 3 | E,E,E |
| RGBE Hybrid | 4 | R,G,B,E |

#### 3.3.1 → Baseline RGB (전면 교체)

**현재**: Baseline RGBD 설명
**v6으로**: Baseline RGB 설명 (v6.md 3.3.1 참조)

#### 3.3.2 Texture 불변 증강 RGBD → **삭제**

이 항을 완전히 삭제.

#### 번호 재배정

- 현재 3.3.3 Edge-only → **새 3.3.2**
- 현재 3.3.4 RGBE Hybrid → **새 3.3.3**

---

### 5장 실험 결과 — 전면 교체 필요

#### 5.1 (448 정확도) — Table 5

**현재**: Baseline RGBD / Texture Aug RGBD / Edge-only / RGBE Hybrid
**v6으로**: Baseline RGB / RGBD / Edge-only / RGBE Hybrid

새 데이터:

| Model | Test Original | Aug (FG) | Aug (Full) |
|-------|:---:|:---:|:---:|
| Baseline RGB | 99.89 ± 0.24 | 90.69 ± 3.08 | 92.17 ± 3.50 |
| RGBD | 100.00 ± 0.00 | 90.79 ± 1.57 | 92.91 ± 1.89 |
| Edge-only | 98.83 ± 0.79 | 69.31 ± 5.36 | 73.76 ± 3.17 |
| RGBE Hybrid | 100.00 ± 0.00 | 91.64 ± 1.65 | 93.86 ± 1.70 |

본문 설명도 "Baseline RGBD" → "Baseline RGB", "Baseline RGB도 99.89%로 근접" 등으로 변경.

**참고**: v6에서는 Macro F1을 별도 Table 7로 분리했으나, DOCX에서는 Accuracy만 별도 Table이고 F1은 본문 언급. v6 구조를 따라 Macro F1 Table도 추가하거나, 현재처럼 본문 언급만 해도 됨.

#### 5.2 (클래스별 F1) — Table 6

**현재 모델명**: Baseline RGBD / Texture Aug / Edge-only / RGBE
**v6 모델명**: Baseline RGB / RGBD / Edge-only / RGBE Hybrid

**모든 숫자** v6.md Table 8 데이터로 교체.

**본문 해석**도 교체:
- "6클래스 최고" → "5클래스 최고"
- "Baseline RGBD와 Texture Aug RGBD가 각 1클래스" → "Baseline RGB가 1클래스, RGBD가 2클래스"
- "전방 좌측 도어 혼동 건수" → v6 내용으로 교체 (혼동 건수 분석 삭제, RGB의 일부 우위 관찰 추가)

#### 5.3 (강건성 Δ) — Table 7

**현재**: 4종 (RGBD, Texture Aug, Edge, RGBE)
**v6으로**: 4종 (RGBE, RGBD, RGB, Edge)

새 데이터:

| Model | Δ(Aug FG) | Δ(Aug Full) | Avg Δ | Rank |
|-------|:---------:|:----------:|:-----:|:----:|
| RGBE Hybrid | −8.36 ± 1.65 | −6.14 ± 1.70 | −7.25 | 1 |
| RGBD | −9.21 ± 1.57 | −7.09 ± 1.89 | −8.15 | 2 |
| Baseline RGB | −9.21 ± 3.10 | −7.73 ± 3.50 | −8.47 | 3 |
| Edge-only | −29.52 ± 5.49 | −25.08 ± 2.96 | −27.30 | 4 |

본문: "Texture Aug RGBD(−8.68%p)" → "Baseline RGB(−8.47%p)"

#### 5.4 (통계 검정) — Table 8

**전면 교체**. 모든 모델 쌍을 v6.md Table 10으로 교체.

본문: "Baseline RGBD · Texture Aug RGBD · RGBE Hybrid ... p > 0.08" → "Baseline RGB · RGBD · RGBE Hybrid ... p > 0.22"

#### 5.5 (학습 수렴)

**현재**: Baseline RGBD 25.6 에포크, Texture Aug 25.4 에포크
**v6으로**: Baseline RGB 23.4 에포크 추가, Texture Aug 제거

새 데이터:

| Model | Best Val Accuracy | Early Stop Epoch |
|-------|:---:|:---:|
| Baseline RGB | 100.00 ± 0.00% | 23.4 ± 1.0 |
| RGBD | 100.00 ± 0.00% | 25.6 ± 4.2 |
| Edge-only | 99.15 ± 0.26% | 34.4 ± 4.3 |
| RGBE Hybrid | 100.00 ± 0.00% | 26.8 ± 3.7 |

**Fig. 4** 학습 곡선 이미지도 교체 필요 → `artifacts/summary_noaux/learning_curves_448.png`

#### 5.6 (해상도 비교) — Table 9

**전면 교체**. 4종 모델 모두 v6.md Table 12, 13 데이터로 교체.

**핵심 변경**: 224에서 "Texture Aug RGBD 1위(−6.03%p)" → "RGBD 1위(−8.30%p)"

**Fig. 5** 이미지도 교체 → `artifacts/summary_noaux/resolution_comparison.png`

#### 5.7 (224 통계) — **Table 14 신규 추가**

**현재**: 문장만 존재, 근거 표 없음
**v6으로**: Table 14 (Pairwise paired t-test results, Macro F1, 224×224) 추가 + 본문 보강

Table 14 데이터 (12행):

| Model A | Model B | Dataset | t-stat | p-value | Sig. (α=0.05) |
|---------|---------|---------|:------:|:------:|:--------------:|
| Baseline RGB | RGBD | Aug (FG) | −0.960 | 0.3915 | No |
| Baseline RGB | Edge-only | Aug (FG) | 3.789 | 0.0193 | **Yes** |
| Baseline RGB | RGBE Hybrid | Aug (FG) | 0.611 | 0.5743 | No |
| RGBD | Edge-only | Aug (FG) | 13.755 | 0.0002 | **Yes** |
| RGBD | RGBE Hybrid | Aug (FG) | 3.192 | 0.0332 | **Yes** |
| Edge-only | RGBE Hybrid | Aug (FG) | −7.889 | 0.0014 | **Yes** |
| Baseline RGB | RGBD | Aug (Full) | −0.845 | 0.4458 | No |
| Baseline RGB | Edge-only | Aug (Full) | 1.745 | 0.1559 | No |
| Baseline RGB | RGBE Hybrid | Aug (Full) | −0.118 | 0.9114 | No |
| RGBD | Edge-only | Aug (Full) | 10.093 | 0.0005 | **Yes** |
| RGBD | RGBE Hybrid | Aug (Full) | 2.562 | 0.0625 | No |
| Edge-only | RGBE Hybrid | Aug (Full) | −6.603 | 0.0027 | **Yes** |

**448 대비 핵심 차이**:
- RGBD vs RGBE Hybrid, Aug (FG): **유의** (p=0.0332) ← 448에서는 비유의 (p=0.3266)
- Baseline RGB vs Edge-only, Aug (Full): **비유의** (p=0.1559) ← 448에서는 유의 (p=0.0027)

#### 5.8 Grad-CAM

**5.8.1 제목**: "4종 모델 attention 비교" → "모델 간 attention 비교"

**본문**:
- "Baseline RGBD와 Texture Aug RGBD는 도어 표면 전체에..." → "RGBD는 도어 표면 전체에..."
- Texture Aug 관련 언급 모두 제거, RGB 관련 관찰 반영

**Fig. 6** 이미지 교체 → `artifacts/summary_noaux/gradcam_fig6.png` (RGB baseline 포함 버전으로 업데이트됨)

**5.8.2**: "RGBE Hybrid의 강건성이 Edge 채널에 의한 구조적 attention 안정화에서 기인함을 뒷받침한다" → "관련될 수 있음을 시사한다. 다만 이는 정성적 관찰이며..."

---

### 6장 고찰 — 전면 교체

v6.md의 6장 전체로 교체. 주요 변경점:

| 소절 | 현재 (DOCX) | v6 |
|------|------------|-----|
| 6.1 | "가장 강건한 모델" | "강건성 경향과 통계적 한계" |
| 6.2 | RGBD vs RGBE만 비교 | RGB도 포함한 3종 비교, RGB≈RGBD 관찰 |
| 6.3 | Edge-only 한계 (동일) | 유사하나 p값 업데이트 (0.002→0.003) |
| 6.4 | "Texture Aug RGBD 224에서 1위" | "RGBD 224에서 1위" |
| 6.5 (한계) | 없음 | **신규 추가**: 연구 한계 4항목 |

---

### 7장 결론 — 전면 교체

v6.md의 7장 전체로 교체.

**핵심 변경**:
- (1)항: "Texture Aug(−8.68%p)" → "Baseline RGB(−8.47%p)", p > 0.08 → p > 0.22
- (3)항: "Texture Aug RGBD 224에서 1위" → "RGBD 224에서 1위"
- (5)항: 새 관찰 — "Baseline RGB가 RGBD와 유사한 강건성"
- 향후 연구에 "보다 큰 규모의 데이터셋을 통한 검정력 확보" 추가

---

### ACKNOWLEDGEMENT

**현재**: 템플릿 기본값 ("여기에 후기를 입력하시오")
**작성 필요**:
> 본 연구는 산업통상자원부의 "굴착기 혼류 생산을 위한 로봇용접 및 AI 기반 영상 PAUT 복합 검사 시스템 개발" 과제의 지원을 받아 수행되었습니다.

---

### 저자소개

**현재**: 템플릿 기본값 ("Gil Dong Hong", "abc@dfg.ac.kr")
**작성 필요**: 실제 저자 정보로 교체

---

## 참고문헌 양식 점검

### 발견된 양식 불일치

| 번호 | 문제 | 수정 방법 |
|:----:|------|----------|
| **[7]** | 저자명 형식이 다름: `Soltan, S., Oleinikov, A., ...` (성, 이니셜 순서) | 다른 참고문헌과 동일하게 `S. Soltan, A. Oleinikov, M. F. Demirci, and A. Shintemirov` 로 수정 |
| **[9]** | 앞에 불필요한 공백 존재: `⎵Y. Li, M. Paluri, ...` | 선행 공백 제거 |

### 양식 일관성 확인 결과

나머지 [1]~[6], [8], [10]~[18] 참고문헌은 아래 양식을 일관되게 따르고 있음:

- **저자**: `이니셜. 성, 이니셜. 성, and 이니셜. 성`
- **학회 논문**: `"제목," in Proc. 학회명, pp. 페이지, 연도.`
- **저널 논문**: `"제목," 저널명, vol. X, no. Y, pp. 페이지, 연도.`
- **번호**: `[1]` ~ `[18]` 순서 정확

v6.md에서 참고문헌 목록 자체는 변경 없음 (동일한 18개 참고문헌).

---

## 교체 필요한 이미지 파일

| Fig. | DOCX 현재 상태 | 교체 파일 경로 |
|------|--------------|--------------|
| Fig. 1 | v5 구조도 (3종) | Mermaid 재렌더링 또는 재작성 (RGB 추가) |
| Fig. 2 | 8종 도어 이미지 | 변경 없음 |
| Fig. 3 | 분할 프로토콜 | 변경 없음 |
| Fig. 4 | 학습 곡선 (4종 v5) | `artifacts/summary_noaux/learning_curves_448.png` |
| Fig. 5 | 해상도 비교 (v5) | `artifacts/summary_noaux/resolution_comparison.png` |
| Fig. 6 | Grad-CAM 4모델 (v5) | `artifacts/summary_noaux/gradcam_fig6.png` |
| Fig. 7 | Grad-CAM RGBD vs RGBE | `artifacts/summary_noaux/gradcam_fig7.png` |

---

## 작업 체크리스트

- [ ] 제목 교체
- [ ] 초록/Abstract 교체
- [ ] 1장 서론 수정 (연구 목적 3항)
- [ ] 2.2 제목/내용 수정
- [ ] 3.1 Fig.1 구조도 업데이트
- [ ] 3.2 3채널 모델 설명 추가
- [ ] 3.3 Table 1 교체
- [ ] 3.3.1 Baseline RGB로 교체
- [ ] 3.3.2 Texture Aug RGBD 삭제
- [ ] 3.3.3~3.3.4 번호 재배정 (3.3.2~3.3.3)
- [ ] 5.1 Table 5 데이터 교체 + 본문
- [ ] 5.2 Table 6 (클래스별 F1) 교체 + 본문
- [ ] 5.3 Table 7 (강건성 Δ) 교체 + 본문
- [ ] 5.4 Table 8 (통계 검정) 교체 + 본문
- [ ] 5.5 학습 수렴 데이터 교체 + Fig.4 이미지 교체
- [ ] 5.6 Table 9 (해상도 비교) 교체 + Fig.5 이미지 교체
- [ ] 5.7 224 통계: **Table 14 신규 추가** + 본문 보강
- [ ] 5.8 Grad-CAM 본문 수정 + Fig.6, Fig.7 이미지 교체
- [ ] 6장 고찰 전면 교체
- [ ] 7장 결론 전면 교체
- [ ] ACKNOWLEDGEMENT 작성
- [ ] 참고문헌 [7] 저자명 양식 수정
- [ ] 참고문헌 [9] 선행 공백 제거
- [ ] 저자소개 작성
