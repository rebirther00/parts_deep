# class_estimation (분류 스몰 프로젝트) 실행 가이드

## 1) 데이터 생성 (Isaac Sim 환경 필요)
```bash
cd /home/rebirther/isaac_data_output
python class_estimation/01_generate_multi_class_dataset.py
```

- 출력: `class_estimation/datasets/`

---

## 2) 분류 학습
```bash
cd /home/rebirther/isaac_data_output
python3 class_estimation/02_parts_classification.py
```

- 기본 학습 데이터: `class_estimation/datasets/`
- 산출물(모델/인덱스/클래스명): `class_estimation/artifacts/`

---

## 3) 평가
```bash
cd /home/rebirther/isaac_data_output
python3 class_estimation/03_parts_class_evaluation.py
```

- 산출물(결과 이미지/JSON): `class_estimation/artifacts/`


