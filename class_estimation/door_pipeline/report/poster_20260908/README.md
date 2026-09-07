# 포스터 자료 폴더 (2026-09-08)

방향: "굴착기 도어 용접 공정 자동화를 위한 현장 데이터 수집·검증 체계 구축 사례" — 근거·패널 구성은
`../CNN_현장평가_정리_포스터검토_20260908.md` 6절. 기존 포스터 양식 파일은 저장소·PC에 없어 학회 양식 확보 시 재배치.

| 파일 | 패널 | 내용 |
|---|---|---|
| abstract_draft.md | — | 국문·영문 초록 초안 v1 (학회 분량 확정 후 조정) |
| fig1_process_flow.(dot/png/svg) | 1 배경 | 현재 수동 공정 → 선행 요건(본 연구) → 목표 자동 공정 |
| fig2_system.(dot/png/svg) | 2 수집 체계 | 카메라·엣지 캡처 UI·NAS·DB·라벨 검증 루프·웹 |
| fig4_data_scale.(png/svg) | 2 수집 체계 | 클래스별 세션 수·RGB-D 쌍 수 (DB 기준, 수집 진행 중이라 갱신 필요) |
| fig6_label_qa_E23.jpg, fig7_label_qa_mislabel.jpg | 3 품질 관리 | 미등록 부품 E23 발견, 클래스 오선택 정정 대조 시트 |
| fig9_hole_detection.jpg | 3·5 | 홀 랜드마크 검출 오버레이(볼트홀 4·모서리 홀 2) |
| fig3_session_split.(png/svg) | 4 평가 프로토콜 | 세션 단위 분할 개념도(실제 split) |
| fig8_failure_case.jpg | 5 활용 검증 | CNN 실패 세션(촬영 조건 변형) vs 학습 세션 |
| fig5_roadmap.(png/svg) | 6 로드맵 | 자동 공정 로드맵(계획 부분은 가안) |

결과 표(패널 5)는 포스터에서 직접 작성: 홀 판별기 9종 1,699장 판정 98%·판정 정확도 99.9% / CNN 세션 CV 97.4%(77세션 1,663장) /
자세 정합 잔차 med 3.25~4.9mm. 수치 정본은 `../hole_analysis/field_20260907/README.md`.

재생성: `dot -Tpng -Gdpi=200 figX.dot -o figX.png`; matplotlib 그림은 이 세션의 스크립트(폰트 NotoSansCJK-Bold.ttc를 addfont로 등록) —
데이터가 늘면 DB 쿼리 기반이라 다시 실행하면 갱신됨.
