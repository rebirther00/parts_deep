"""Finish the first formal fold after its three training processes complete.

This is a foreground job, not a system service. It uses only isolated research
artifacts; no original inference service is invoked or replaced.
"""
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from door8.core import ROOT, read_json


if __name__=='__main__':
    while True:
        statuses={n:read_json(ROOT/'runs'/n/'run.json')['status'] for n in ['rgb_f0_s42','rgbe_f0_s42','holes_lab_s42']}
        if all(v=='complete' for v in statuses.values()):break
        if any(v not in ['running','complete'] for v in statuses.values()):raise RuntimeError(statuses)
        time.sleep(20)
    # Extract only after all training completes: latency is not measured during training.
    jobs=[['extract','--kind','holes','--run','holes_lab_s42','--features','holes_lab_s42_full'],
          ['extract','--kind','mobile_sam','--features','mobile_sam_full'],
          ['eval-cnn','--run','rgb_f0_s42','--report','rgb_f0_s42_test'],
          ['eval-cnn','--run','rgbe_f0_s42','--report','rgbe_f0_s42_test'],
          ['eval-geometry','--features','holes_lab_s42_full','--mode','pixel','--fold','0','--report','holes_pixel_f0_s42'],
          ['eval-geometry','--features','holes_lab_s42_full','--mode','depth','--fold','0','--report','holes_depth_f0_s42'],
          ['eval-geometry','--features','mobile_sam_full','--mode','mask','--fold','0','--report','mask_f0'],
          ['compare','--reports','rgb_f0_s42_test','rgbe_f0_s42_test','holes_pixel_f0_s42','holes_depth_f0_s42','mask_f0','--output','comparison_f0_s42']]
    for args in jobs:
        print('Running',args,flush=True);subprocess.run([sys.executable,str(ROOT/'run.py'),*args],cwd=ROOT,check=True)
