# 도어 8클래스 논문 재개발

기존 운영 프로젝트를 수정하지 않는 독립 연구 폴더이다. 데이터 원본과 수동 라벨은 읽기 전용 입력이며, 모델·분할·중간 특징·평가·문서는 모두 이 폴더 안에 저장한다. 운영 모델을 논문 학습의 초기 가중치로 사용하지 않는다.

**현재 상태: 데이터 고정, 실험 코드 구현, 시드 42의 새 모델 3개 학습 및 첫 폴드 비교, 논문 초안 작성. 정식 5폴드 × 3시드 비교 실험과 향후 전향적 현장 검증은 아직 완료되지 않았다. `pilot_*` 결과는 논문 성능표에 사용할 수 없다.**

## 먼저 읽을 문서

- [논문 초안](manuscript/논문초안_v1.md): 실제 데이터 수와 방법을 반영한 본문. 미실행 결과는 미측정으로 표시.
- [실험 계획](docs/실험계획.md): 연구 질문, 비교 조건, 누수 방지, 추가 수집 항목.
- [재투고 판단과 심사 대응](docs/재투고_판단.md): 무엇이 새 기여이고 무엇이 아직 부족한지.
- [첫 폴드 실제 결과](reports/첫폴드_결과.md): 동일 시험 74세션 비교. 전체 반복 결과와 구분.
- [현재 검증 상태](reports/validation_status.json): 실제 실행 기록.
- [고정 데이터 요약](data/snapshots/v1/summary.json), [제외 세션](data/snapshots/v1/excluded_sessions.csv).

## 데이터와 결정

| 항목 | 고정 내용 |
|---|---|
| 연구 범위 | E23을 제외한 8클래스; VLM·레이더 옵션·자동 홀 좌표 라벨은 이번 품종 분류 연구에 사용하지 않음 |
| 현장 자료 | 2026-08-27~10-29, 361세션, 원 수집 36,142 RGB/depth 쌍 |
| 현재 실험 입력 | 로컬 보유 7,376쌍에서 세션당 최대 20장 균등 추출 → 7,104쌍 |
| 제외 | 기존 무효 7세션 + 육안 확인 후 클래스가 정정된 11세션(그중 E23 6세션); 총 18세션 |
| 홀 수동 라벨 | 실험실 118장: 보조 학습 94장, 체크포인트 선택 24장. 과거 현장 16장은 제외 |
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

- `door8/`: 원본 운영 모듈을 import하지 않는 새 코드. 학습, 측정, 보정, 평가를 분리.
- `config/protocol.json`: 다음 스냅샷을 만들 때 사용할 설정. 이미 고정된 실험은 스냅샷 내부 설정을 사용.
- `data/snapshots/v1/`: 분할, 해시, 라벨 출처, 제외 목록, SQLite 복사본. 대용량 원본 이미지는 중복 복사하지 않고 참조.
- `runs/`, `features/`: 모델·특징. 원본과 섞이지 않으며 Git에서는 제외.
- `reports/`: 감사 및 평가 결과. 실패/보류도 전체 분모에 포함.
- `manuscript/`, `docs/`: 초안과 실험 설계.

MobileSAM 비교는 고정 박스 프롬프트와 마스크 치수 특징을 쓰는 기하 기준선이다. 기존 슬롯 U-Net을 재현한 실험도, 최신 분할 방법 전체의 대표 성능도 아니다. 홀 방법은 수동 좌표와 CAD를 추가로 사용하므로 CNN과 같은 감독 정보량의 비교라고 주장하지 않는다. 물리적 개체·생산 로트 식별자는 현재 없으며 세션 중복 방지가 개체 독립성까지 증명하지 않는다.
