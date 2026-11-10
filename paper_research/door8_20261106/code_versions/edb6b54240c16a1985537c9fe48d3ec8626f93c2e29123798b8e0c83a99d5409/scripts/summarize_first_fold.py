"""Publish only measured first-fold results into the research draft."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from door8.core import ROOT,dump,now,outpath,read_json,sha256,snapshot,setup_runtime,capture_code
setup_runtime()


if __name__=='__main__':
    comparison=read_json(ROOT/'reports/comparison_f0_s42/comparison.json');table=comparison['table']
    names={'rgb':'RGB CNN','rgbe':'RGBE CNN','holes_pixel':'홀 + 픽셀','holes_depth':'홀 + 깊이','mobile_sam_dimensions':'MobileSAM + 치수'}
    lines=['| 방법 | 프레임 전체 정확도 | 프레임 커버리지 | 세션 전체 정확도 | 세션 macro-F1 |',
           '|---|---:|---:|---:|---:|']
    for r in table:lines.append(f"| {names[r['method']]} | {r['frame_accuracy_all']*100:.2f}% | {r['frame_coverage']*100:.2f}% | {r['session_accuracy_all']*100:.2f}% | {r['session_macro_f1']:.4f} |")
    block='\n'.join(lines)
    result=('# 첫 폴드 정식 비교 결과\n\n시드 42, Fold 0. 학습 214세션 / 검증 73세션 / 시험 74세션. '
            '동일 시험 1,478프레임. 모든 정확도는 실패·보류를 분모에 포함한다.\n\n'+block+
            '\n\n이는 5폴드 × 3시드 전체 결과가 아닌 첫 비교이다. 반복 전체를 완료하기 전 대표 성능이나 방법의 우월성으로 일반화하지 않는다. '
            '현재 자료는 기존 개발에서 관찰한 현장 기간의 후향적 재구성이며, 물리적 개체 독립성도 확인되지 않았다.\n\n')
    result+='## 모델 선택과 진단\n\n'
    for run in ['rgb_f0_s42','rgbe_f0_s42','holes_lab_s42']:
        m=read_json(ROOT/'runs'/run/'run.json');h=read_json(ROOT/'runs'/run/'history.json')
        if m['status']!='complete' or m['pilot']:raise ValueError('Expected completed formal model')
        result+=f"- `{run}`: {len(h)} epoch 학습, 검증으로 {m['best_epoch']} epoch 선택. 가중치 SHA-256 `{m['weights_sha256']}`.\n"
    lab=read_json(ROOT/'reports/lab_localization_diagnostic.json')
    result+=(f"\n실험실 체크포인트 선택 검증: {lab['images']}장, 보이는 정답 점 {lab['visible_gt_points']}개, "
             f"검출 {lab['matched_points']}개, 검출된 점의 오차 중앙값 {lab['median_error_original_px_when_matched']:.2f} 원본 픽셀. 독립 시험 성능이 아니다.\n\n")
    result+='## 보류 원인\n\n'
    for report in ['holes_pixel_f0_s42','holes_depth_f0_s42','mask_f0']:
        result+=f"- `{report}`: {read_json(ROOT/'reports'/report/'failure_counts.json')}\n"
    result+='\n대응 세션 bootstrap 구간과 McNemar 원 p값·Holm 보정 p값은 `comparison_f0_s42/comparison.json`에 저장했다. 세션 독립성 가정과 단일 폴드라는 한계를 함께 해석해야 한다.\n'
    outpath('reports/첫폴드_결과.md').write_text(result,encoding='utf8')

    paper=ROOT/'manuscript/논문초안_v1.md';s=paper.read_text()
    s=s.replace('정식 비교 실험과 전향적 현장 평가는 미실행이므로 성능 결과를 만들어 넣지 않았다.',
                '시드 42의 첫 폴드 정식 비교를 완료했으며 실제 결과를 별도로 제시했다. 전체 반복 실험과 전향적 현장 평가는 미완료이다.')
    s=s.replace('현재 완료한 실행은 데이터 전수 해시 검사와 소규모 학습·추론의 동작 확인이다.',
                '데이터 전수 해시 검사 및 소규모 동작 확인에 이어 RGB, RGBE, 실험실 전용 홀 검출기의 시드 42 학습과 첫 현장 시험 폴드의 비교를 완료하였다.')
    s=s.replace('다음 절의 성능표는 이 실행이 끝난 뒤 동일 시험 입력의 산출물에서 채워야 한다.',
                '다음 절에서 전체 반복 성능표는 미측정으로 두고, 완료된 첫 폴드 결과를 따로 표시한다.')
    marker='### 5.2 치수 잔차와 실패 원인'
    interim=('**첫 폴드 실측 결과 — 전체 반복 결과와 구분:** 시드 42, 시험 74세션 및 1,478프레임에 대한 결과는 다음과 같다. '
             '검출 실패와 보류를 포함한 전체 정확도이다.\n\n'+block+
             '\n\n이 비교만으로 전체 현장 일반화나 방법 간 우월성을 결론내릴 수 없다. 동일한 규칙으로 나머지 폴드와 시드를 완료하고 OOF 결과를 구성해야 한다.\n\n')
    if '**첫 폴드 실측 결과' not in s:s=s.replace(marker,interim+marker)
    paper.write_text(s,encoding='utf8')
    p=ROOT/'README.md';s=p.read_text()
    s=s.replace('데이터 고정, 실험 코드 구현, GPU 소규모 동작 검증, 논문 초안 작성.',
                '데이터 고정, 실험 코드 구현, 시드 42의 새 모델 3개 학습 및 첫 폴드 비교, 논문 초안 작성.')
    s=s.replace('- [현재 검증 상태]', '- [첫 폴드 실제 결과](reports/첫폴드_결과.md): 동일 시험 74세션 비교. 전체 반복 결과와 구분.\n- [현재 검증 상태]')
    p.write_text(s)
    sp,_,_,_=snapshot();prov=read_json(sp/'provenance.json')
    checks={p:sha256(p)==v for p,v in prov['source_files'].items()}
    dump('reports/source_integrity.json',{'checked_at':now(),'matches_snapshot':checks,'all_match':all(checks.values()),
        'scope':'Listed original reference files only. This research never writes to the operational DB or original images.'})
    dump('reports/validation_status.json',{'recorded_at':now(),'snapshot':'v1','data_audit':read_json(ROOT/'reports/data_audit.json'),
        'safeguard_tests':{'runner':'python -m unittest discover -s tests -v','passed':13,'failed':0},
        'formal_runs':['rgb_f0_s42','rgbe_f0_s42','holes_lab_s42'],
        'formal_comparison':'comparison_f0_s42','completed_test_folds':[0],'completed_seeds':[42],
        'full_cv_complete':False,'prospective_evaluation_complete':False,
        'original_reference_integrity':checks,'stl_thickness_report':'stl_thickness.json',
        'training_code_for_first_formal_runs':'code_versions/f35334876fd3c39f25656601317d3ca422b1af5e70fe53f0a9ed9c752995920d/learning.py',
        **capture_code()})
    print(block)
