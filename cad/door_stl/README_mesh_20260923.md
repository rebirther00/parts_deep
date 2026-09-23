# STEP → STL 변환 기록 (2026-09-23)

- `E23_door_LH_FRT_assy.stl` = 어셈블리 STEP `110982-02444B`(보강재·힌지 포함) 메시. **170MB 라 git 미추적**(`.gitignore` `*_assy.stl`) — 아래 .geo 로 2분 만에 재생성.
  추적 파일 `E23_door_LH_FRT.stl` 은 종전 외판 단일 메시(02723) 그대로. `pos_pipeline/01_extract_cad_holes.py` 는 `<class>_assy.stl` 이 있으면 우선 쓴다.
  자세 정식 등록(`pos_pipeline/cad_holes.json` E23, D 457.6·정합 3.1mm·마진 30.5)은 이 어셈블리 메시로 했고 잠정 등록(PROVISIONAL)은 해제했다.
- `E30_door_LH_RR.stl` 은 종전 메시(구 리비전, RADAR 사양) 유지. 신규 `110982-02360K` STEP 은 `mesh_E30_door_LH_RR_02360K.geo` 로 변환은 되지만
  곡률 12 메시에서는 홀 벽면 샘플이 부족해 `01_extract_cad_holes.py` 볼트 직사각 탐지 실패 → 곡률 24·최소 0.8mm(약 420MB)로 재변환해야 등록 가능(미완).
- `gmsh <stp> -2 -format stl -clcurv 36` 은 이 어셈블리들에서 곡선 메싱이 멈춤(20분+) → OCC healing 을 켠 .geo 사용:

```bash
cd cad/door_stl && gmsh mesh_E23_door_LH_FRT_02444B.geo -2 -format stl -o E23_door_LH_FRT_assy.stl      # 약 1~2분
```

- 홀 추출 요건: 홀 벽면 점이 1.2mm 이내로 이어져야 함(`01_extract_cad_holes.find_holes_3d`, `--cluster-mm`). `Mesh.MinimumCirclePoints` 만으로는(STEP 홀이 BSpline 호) 부족하고 `Mesh.MeshSizeFromCurvature` 가 필요.
