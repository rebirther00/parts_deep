# vlm/ — Claude API(vision) 활용 보조 작업 (2026-11-03, API 크레딧 11/17 만료분)

학습 PC 에서 실행. 키는 `ANTHROPIC_API_KEY` 또는 `~/.config/anthropic/api_key`(600, dx300_rl 과 공용). 모델 Claude Opus 5.5. 현장 이미지가 Anthropic API 로 전송된다(사용자 승인 11/3).

| 파일 | 역할 |
|---|---|
| `vlm_common.py` | 클라이언트·960px JPEG 인코딩·대표 이미지 15장 프리픽스(캐시 브레이크포인트)·도메인 설명·판정 JSON 스키마·비용 집계 |
| `audit_sessions.py` | ①② 독립 2차 감사: `--source collect`(확정 세션 vs GT) / `--source infer`(07 판정 vs VLM + 미확정 원인 설명). 결과 `artifacts/audit_*.jsonl`, 요약 `../report/vlm_audit/*.md`, DB `vlm_audits` |
| `cron_audit_infer.py` | **cron(30분, ingest_infer 2분 뒤)**: 새 추론 세션만 2프레임 감사 → DB. **무료 크레딧까지만** — 2026-11-17 부터 자동 중지(`VLM_AUDIT_UNTIL`), 누적 $100 상한(`VLM_AUDIT_CAP_USD`), 잠금 파일, 로그 `logs/vlm_audit.log`, 상태 `artifacts/cron_state.json`. webapp /inference 'VLM 2차 의견' 열·`?vlm=diff` 필터·상세 표·대시보드 카드가 이 결과를 보여 준다 |
| `build_instruction_set.py` | ③ 인스트럭션 셋: 세션 사실표+프레임 → 질문/응답 4쌍. `--pilot N`(동기) → `--submit`(Batch) → `--collect`. 산출 `artifacts/instruction_set_v0.jsonl`, `../DOC_instruction_set.md` |

```bash
source ~/parts_deep/venv/parts_deep/bin/activate && cd ~/parts_deep/class_estimation/door_pipeline
python vlm/audit_sessions.py --source collect --dry-run        # 대상·비용만
python vlm/audit_sessions.py --source collect --limit 5        # 파일럿
python vlm/audit_sessions.py --source collect                  # 전체(339세션 1프레임, ≈$12, ~30분)
python vlm/audit_sessions.py --source collect --frames 3 --sessions vlm/artifacts/disagree_collect_<run>.txt   # 불일치만 3프레임 재감사
python vlm/audit_sessions.py --source infer --frames 2         # 07_factory_infer 세션(NAS 에서 rgb_0000/0005 자동 pull)
python vlm/build_instruction_set.py --pilot 3 && python vlm/build_instruction_set.py --submit && python vlm/build_instruction_set.py --collect --wait 60
```

- VLM 은 형상군·레이더·이상 설명까지가 신뢰 범위. E30↔E38 FRT(폭 47mm 차)는 육안 판별 불가 전제 — 클래스 일치율은 참고치.
- `_pts` 오버레이 이미지는 참조에서 제외(판정기 결과 누출 방지). 추론 세션도 thumb.jpg 대신 원본 rgb 를 쓴다.
