# 도어 검사 인스트럭션 셋 v0 (DOC_instruction_set.md)

작성: 2026-11-03. AI 팩토리 데이터 체크리스트 1-7(옵션) "인스트럭션 셋 데이터" 산출물. 생성기 claude-opus-5-5, 생성 스크립트 `vlm/build_instruction_set.py`.

## 무엇인가

- 현장 세션(사용자 확정 라벨·품번) 1건당 대표 프레임 1장 + 확정 사실표(클래스·품번·레이더·CAD D·측정 D·마진·자세·레이더 검사 점수)를 주고, **사실표 안의 값만으로** 한국어 질문/응답 4쌍(identify/reason/radar/inspect)을 쓰게 했다.
- 형식: JSONL, 1행 = 1세션. `image`(door_pipeline 기준 상대경로) · `facts`(근거 사실표) · `conversations`(LLaVA 식 human/gpt 교대, human 에 `type`) · `generator` · `created_at`.
- 용도: 시각-언어 검사 보조 모델의 지도 미세조정(SFT)·평가, 작업자 질의응답 데모. 라벨 정답은 사실표가 정본이며 응답 문장은 생성물이다.

## 규모

- 세션 339 / 대상 339 · 질문/응답 쌍 1356
- 유형별: identify 339, inspect 339, radar 339, reason 339
- 클래스별 세션: E23_door_LH_FRT 6, E25_door_LH_FRT 7, E25_door_LH_RR 64, E25_door_RH 62, E30_E38_door_RH 68, E30_door_LH_FRT 39, E30_door_LH_RR 46, E38_door_LH_FRT 20, E38_door_LH_RR 27
- 비용: 339건 $5.856 input_tokens=519186 cache_read_tokens=330902 cache_write_tokens=979 output_tokens=478249 (배치 50% 적용)

## 검증 결과 (2026-11-03, 생성 직후 자동 검사)

- 339세션 전부 성공(오류·거부 0), 질문/응답 1,356쌍(identify/reason/radar/inspect 각 339).
- 응답에 등장한 품번(110982-xxxxx)이 사실표 밖인 경우 **0건**, identify 응답에 정답 품번이 빠진 경우 **0건**.
- 응답 길이 중앙값 238자(최대 365자). 클래스별 세션: E30_E38_door_RH 68 · E25_door_LH_RR 64 · E25_door_RH 62 · E30_door_LH_RR 46 · E30_door_LH_FRT 39 · E38_door_LH_RR 27 · E38_door_LH_FRT 20 · E25_door_LH_FRT 7 · E23_door_LH_FRT 6.
- 한계: inspect 응답은 이미지 묘사라 자동 검증 불가(표본 육안 검수 필요). 사실표에 없는 내용은 "확인 필요"로 답하도록 했고 실제로 그렇게 답한 문장이 있다(예: 브래킷 위치). 클래스 불균형(E23·E25 FRT 소수)은 현장 생산 비율 그대로다.

## 검수 방법(권장)

- 응답의 숫자·품번이 `facts` 와 다르면 폐기(스크립트 `--verify` 로 숫자 포함 검사 가능 — 향후).
- inspect 유형은 이미지 묘사라 사실표로 검증 불가 → 표본 육안 검수.

## 파일

- `vlm/artifacts/instruction_set_v0.jsonl` (git 미추적, NAS 백업 권장)
- 배치 ID·상태: `vlm/artifacts/instruction_batch.json`
