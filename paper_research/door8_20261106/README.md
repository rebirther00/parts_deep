# 도어 8클래스 논문 재개발

기존 운영 프로젝트를 수정하지 않는 독립 연구 폴더이다. 데이터 원본과 수동 라벨은 읽기 전용 입력이며, 모델·분할·중간 특징·평가·문서는 모두 이 폴더 안에 저장한다. 운영 모델을 논문 학습의 초기 가중치로 사용하지 않는다.

**현재 상태: 데이터 고정, 실험 코드 구현, 시드 42의 새 모델 3개 학습 및 첫 폴드 비교, 논문 초안 작성. 정식 5폴드 × 3시드 비교 실험과 향후 전향적 현장 검증은 아직 완료되지 않았다. `pilot_*` 결과는 논문 성능표에 사용할 수 없다.**

**2026-11-09 완료:** 현장 수동 좌표 40장(240점)을 검증·고정하고, 수동 좌표 진단 및 현장 학습 좌표 0/8/16/24장 개발 비교를 완료했다. 픽셀 방식의 검증 73세션·1,455프레임에서 0장 대조군은 모두 보류, 8/16/24장 모델은 각각 1,440/1,440/1,441프레임과 모두 73세션을 맞혔다. RGB/RGBE는 같은 검증 프레임을 전부 맞혀 홀 방식의 우월성을 입증하지 않는다. 새 적응 모델은 Fold 0 시험 영상·좌표를 평가하지 않았다. 전체 반복 평가와 전향적 시험은 미완료이다. 결과와 갱신된 원고는 아래 v2 링크를 사용한다.

**실물 독립성 확인:** 2026-11-09 사용자가 361세션 모두 실제 작업에서 각각 다른 도어를 촬영했다고 확인했다. 따라서 세션 분할은 실물 도어 분할에 해당한다. 한 세션 내 프레임을 독립 개체로 세지 않으며, 기존 관찰 기간의 후향적 자료라는 점은 유지한다. 근거: `reports/field_sampling_confirmation_20261109.json`.

**첫 결과와 후속 단계:** 실험실 전용 홀 검출기는 최초 시험 74세션을 전부 보류했다. 이 대조군을 보존하고 현장 수동 좌표 적응의 개발 검증을 완료했다. 다음은 프로토콜을 고정한 반복 평가와 새 촬영 조건에서의 검증이다. `suite --execute`는 실패한 조건도 포함한 v1 반복 실행이며, 현장 적응 모델을 자동으로 만드는 명령이 아니다.

## 먼저 읽을 문서

- [최신 논문 초안 v2](manuscript/논문초안_v2_개발검증.docx): 현장 적응 개발 검증과 실물 분리 확인 반영. 투고 완성본은 아님.
- [현장 적응 비교 보고서](reports/field_v2_development_comparison/결과.docx): 0/8/16/24장 실제 결과와 한계.
- [이전 논문 초안 v1](manuscript/논문초안_v1.md): 최초 시험 결과까지의 역사적 버전. 이후 확인 사항은 v2에 반영.
- [실험 계획](docs/실험계획.md): 연구 질문, 비교 조건, 누수 방지, 추가 수집 항목.
- [재투고 판단과 심사 대응](docs/재투고_판단.md): 무엇이 새 기여이고 무엇이 아직 부족한지.
- [첫 폴드 실제 결과](reports/첫폴드_결과.md): 동일 시험 74세션 비교. 전체 반복 결과와 구분.
- [수동 현장 좌표 진단](reports/field40_manual_f0_v1/진단결과.md): 학습 24장으로 보정하고 검증 8장 평가. 자동 모델 성능·독립 시험과 구분.
- [v2 개발 검증 상태](reports/field_v2_validation_status.json): 모델·특징·수치·원본 해시 검증 기록.
- [v1 검증 상태](reports/validation_status.json): 최초 실행 기록.
- [고정 데이터 요약](data/snapshots/v1/summary.json), [제외 세션](data/snapshots/v1/excluded_sessions.csv).

## 데이터와 결정

| 항목 | 고정 내용 |
|---|---|
| 연구 범위 | E23을 제외한 8클래스; VLM·레이더 옵션·자동 홀 좌표 라벨은 이번 품종 분류 연구에 사용하지 않음 |
| 현장 자료 | 2026-08-27~10-29, 361세션, 원 수집 36,142 RGB/depth 쌍 |
| 현재 실험 입력 | 로컬 보유 7,376쌍에서 세션당 최대 20장 균등 추출 → 7,104쌍 |
| 제외 | 기존 무효 7세션 + 육안 확인 후 클래스가 정정된 11세션(그중 E23 6세션); 총 18세션 |
| 홀 수동 라벨 | 실험실 118장 + 현장 40장. 실험실 학습 94장; v1은 실험실 검증 24장으로 선택, v2는 현장 검증 8장으로 선택. 과거 현장 16장 제외 |
| 실물 도어 수 | 361개: 세션마다 다른 실물임을 수집 담당자가 확인 |
| 현장 분할 | 세션 단위 층화 5폴드; 시험 k, 검증 k+1, 학습 나머지 3폴드 |
| 보조 시간 분할 | 10월 3일 이전 학습 193/검증 35세션, 이후 시험 133세션 |
| 데이터의 지위 | 기존 개발에서 관찰한 기간을 재구성한 후향적 실험; 새 블라인드 시험이 아님 |

작업자와 홀 분류가 불일치했던 세션 및 레이더 옵션은 사용자가 전수 육안 확인했다. 이번 연구는 오라벨을 정정해 사용한 효과를 주장하지 않으며 해당 클래스 정정 세션을 통째로 제외한다. 레이더 옵션이 있는 영상 자체를 제외하는 것은 아니다. 옵션 라벨을 사용하지 않고 같은 도어 형상군으로 묶으며, 브래킷 좌표를 출력하는 기존 4채널 검출기는 사용하지 않는다. 미래 수집 자료는 모델/프로토콜 고정 시점 이후 별도로 구축한다.

## 실행

이 폴더에서 아래 Python을 사용한다. 기존 가상환경을 읽어 사용하며 패키지 설치·업데이트를 하지 않는다. 모든 캐시는 이 폴더의 `cache/`로 향한다.

```bash
cd /home/koceti/parts_deep/paper_research/door8_20261106
../../venv/parts_deep/bin/python run.py audit --full-hash
../../venv/parts_deep/bin/python -m unittest discover -s tests -v
```

고정된 `v1`을 덮어쓰지 않는다. 원본 데이터나 포함 기준을 바꾸는 경우 새 이름으로 `--snapshot v2 freeze`하고 그 스냅샷에서 모델 전체를 다시 생성한다. 실험 도중 파일 누락·수정이 감지되면 중단한다.

정식 단일 폴드 실행 예:

```bash
../../venv/parts_deep/bin/python run.py train-cnn --kind rgb --fold 0 --seed 42 --run rgb_f0_s42
../../venv/parts_deep/bin/python run.py train-cnn --kind rgbe --fold 0 --seed 42 --run rgbe_f0_s42
../../venv/parts_deep/bin/python run.py train-holes --seed 42 --run holes_lab_s42
../../venv/parts_deep/bin/python run.py extract --kind holes --run holes_lab_s42 --features holes_lab_s42_full
../../venv/parts_deep/bin/python run.py extract --kind mobile_sam --features mobile_sam_full
../../venv/parts_deep/bin/python run.py eval-cnn --run rgb_f0_s42 --report rgb_f0_s42_test
../../venv/parts_deep/bin/python run.py eval-cnn --run rgbe_f0_s42 --report rgbe_f0_s42_test
../../venv/parts_deep/bin/python run.py eval-geometry --features holes_lab_s42_full --fold 0 --mode pixel --report holes_pixel_f0_s42
../../venv/parts_deep/bin/python run.py eval-geometry --features holes_lab_s42_full --fold 0 --mode depth --report holes_depth_f0_s42
../../venv/parts_deep/bin/python run.py eval-geometry --features mobile_sam_full --fold 0 --mode mask --report mask_f0
../../venv/parts_deep/bin/python run.py compare --reports rgb_f0_s42_test rgbe_f0_s42_test holes_pixel_f0_s42 holes_depth_f0_s42 mask_f0 --output comparison_f0_s42
```

전체 5폴드 × 3시드 작업 목록을 생성하거나 실행한다. 완료 산출물은 건너뛰고, 미완료 폴더는 임의로 지우지 않고 중단한다. 전체 학습은 수 시간 이상 걸릴 수 있으며 아래 명령은 배포·운영 서비스 변경을 하지 않는다.

```bash
../../venv/parts_deep/bin/python run.py suite
../../venv/parts_deep/bin/python run.py suite --execute
# 시간 분할까지 포함할 경우
../../venv/parts_deep/bin/python run.py suite --execute --temporal
```

5개 폴드가 완료되면 **시드별로** 중복 없는 OOF 결과를 만든다. 서로 다른 시드의 동일 시험 영상을 독립 표본으로 합치지 않는다.

```bash
../../venv/parts_deep/bin/python run.py collect-oof --reports rgb_f0_s42_test rgb_f1_s42_test rgb_f2_s42_test rgb_f3_s42_test rgb_f4_s42_test --output rgb_oof_s42
```

소규모 검증용 명령:

```bash
../../venv/parts_deep/bin/python run.py train-cnn --kind rgb --fold 0 --run my_pilot_rgb --epochs 1 --pilot-per-class 8
../../venv/parts_deep/bin/python run.py train-holes --run my_pilot_holes --epochs 1 --pilot
../../venv/parts_deep/bin/python run.py eval-cnn --run my_pilot_rgb --report my_pilot_validation --pilot
../../venv/parts_deep/bin/python run.py extract --kind holes --run my_pilot_holes --features my_pilot_features --limit 8
```

`--epochs`를 정식 설정과 다르게 지정한 모델도 자동으로 pilot으로 기록한다. pilot 모델/특징으로 정식 시험 성능을 계산하려 하면 거부한다. 파일 이름만 바꾸어 승격할 수 없다.

## 구조와 해석

현장 수동 좌표 후보 40장을 `data/field40_v2_queue.jsonl`에 준비했다. 각 클래스와 CV 그룹에서 1장씩, 모델 점수를 보지 않고 선택했다. 2026-11-09 사용자가 40장 모두 수동 주석했고 별도 라벨 스냅샷으로 고정했다. 매 폴드에서 학습 24장 / 검증 8장 / 점 위치 시험 8장으로 구분한다. **40장 전체를 학습한 공통 검출기로 5폴드를 평가하면 안 된다.** VLM·교사 모델·규칙으로 좌표를 생성하지 않았다.

원본 대시보드와 별개의 로컬 수동 주석 도구를 제공한다. SSH 또는 VS Code 원격 터미널에서는 `--web`으로 브라우저 모드를 사용한다. 일반 실행도 GUI 화면(`DISPLAY`)에 연결할 수 없으면 브라우저 모드로 전환한다. 패키지 추가 설치나 가상환경 활성화는 필요 없으며 아래 Python 경로가 이미 프로젝트 가상환경을 가리킨다. 좌클릭으로 점을 지정하고 휠로 확대하며 우클릭 드래그로 이동한다. 가린 점은 V, 저장 후 다음은 S이다. 저장은 `data/annotations/field40_v2/`에만 이루어진다. 이전 라벨을 고칠 경우 이 폴더의 history에 이전 내용도 보존한다.

```bash
../../venv/parts_deep/bin/python scripts/annotate_field.py --list
../../venv/parts_deep/bin/python scripts/annotate_field.py --web
```

터미널을 켜 둔 채 `http://127.0.0.1:8765/`를 브라우저에서 연다. 접속 정보는 페이지가 자동으로 받으므로 주소 뒤에 토큰을 붙이지 않아도 된다. 원격 접속이면 VS Code **포트(Ports)** 탭에서 **8765**를 전달한다. 브라우저는 서버 PC가 아닌 사용 중인 PC에서 열어도 된다. SSH만 사용하는 경우 로컬 PC에서 `ssh -L 8765:127.0.0.1:8765 koceti@<서버주소>`로 연결한 뒤 같은 주소를 연다. 포트가 사용 중이면 `--web --port 8766`으로 바꾸고 그 포트를 전달한다. 서버는 `127.0.0.1`에만 바인딩하며 Ctrl+C로 종료한다. 실행 중 서버 코드를 수정한 경우 Ctrl+C 후 같은 명령으로 다시 실행해야 적용된다. 초기 연결 오류는 화면 상단에 표시하며, 브라우저 저장소가 제한되어도 주석은 가능하다.

1~4번은 래치 사각 개구 주변 볼트 홀 네 점, 5~6번은 기존 수동 라벨과 같은 의미의 힌지 측·래치 측 기준 홀이다. 볼트 네 점은 학습에서 순서 없는 한 채널로 사용한다. 배경의 체결구나 슬롯 끝점을 찍지 않는다. 출력은 사람이 실제 클릭한 좌표만 저장한다. 현장 좌표를 포함하는 v2는 `scripts/run_field_adaptation.py`로 분리 구현했다. 기존 `run.py train-holes`는 계속 v1 실험실 전용 학습이다.

v2 개발 실험 예시(같은 이름의 실행을 중복 시작하지 않는다):

```bash
../../venv/parts_deep/bin/python scripts/run_field_adaptation.py train --run holes_field24_f0_s42_v2 --budget 24
../../venv/parts_deep/bin/python scripts/run_field_adaptation.py extract --run holes_field24_f0_s42_v2 --features holes_field24_f0_s42_v2_development
../../venv/parts_deep/bin/python scripts/run_field_adaptation.py evaluate --features holes_field24_f0_s42_v2_development --report holes_field24_f0_s42_v2_validation
```

94장의 실험실 학습 좌표와 해당 폴드의 현장 학습 좌표만 학습에 사용한다. 현장 검증 8장으로 체크포인트를 선택하고 실험실 검증 24장은 별도 모니터링한다. 이 단계의 평가 대상은 검증 73세션·1,455프레임으로 독립 시험 성능이 아니다. 별도 실행에서 이미 생성한 LAB-only 특징은 학습·검증 ID로 필터링해 0장 대조군으로 사용한다. 신규 현장 적응 모델은 시험 영상의 특징을 추출하지 않는다.

- `door8/`: 원본 운영 모듈을 import하지 않는 새 코드. 학습, 측정, 보정, 평가를 분리.
- `config/protocol.json`: 다음 스냅샷을 만들 때 사용할 설정. 이미 고정된 실험은 스냅샷 내부 설정을 사용.
- `data/snapshots/v1/`: 분할, 해시, 라벨 출처, 제외 목록, SQLite 복사본. 대용량 원본 이미지는 중복 복사하지 않고 참조.
- `runs/`, `features/`: 모델·특징. 원본과 섞이지 않으며 Git에서는 제외.
- `reports/`: 감사 및 평가 결과. 실패/보류도 전체 분모에 포함.
- `manuscript/`, `docs/`: 초안과 실험 설계.

MobileSAM 비교는 고정 박스 프롬프트와 마스크 치수 특징을 쓰는 기하 기준선이다. 기존 슬롯 U-Net을 재현한 실험도, 최신 분할 방법 전체의 대표 성능도 아니다. 홀 방법은 수동 좌표와 CAD를 추가로 사용하므로 CNN과 같은 감독 정보량의 비교라고 주장하지 않는다. 현장 수집 과정에서 세션과 실물 도어가 일대일임을 확인했다. 생산 로트별 분리와 다른 현장·카메라의 일반화는 아직 확인하지 않았다.
