"""Run the frozen field-coordinate development protocol without opening TEST."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from door8.core import setup_runtime
setup_runtime()
from door8.field_experiments import train_field,extract_development,evaluate_development


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train');t.add_argument('--run',required=True);t.add_argument('--budget',type=int,required=True)
    t.add_argument('--seed',type=int,default=42);t.add_argument('--fold',default='0');t.add_argument('--workers',type=int,default=2)
    t.add_argument('--pilot-epochs',type=int,default=0);t.add_argument('--memory-fraction',type=float,default=.22)
    e=sub.add_parser('extract');e.add_argument('--run',required=True);e.add_argument('--features',required=True);e.add_argument('--fold',default='0')
    v=sub.add_parser('evaluate');v.add_argument('--features',required=True);v.add_argument('--report',required=True);v.add_argument('--fold',default='0');v.add_argument('--baseline',action='store_true')
    a=p.parse_args()
    if a.command=='train':train_field(a.run,a.budget,a.fold,a.seed,a.workers,a.pilot_epochs,a.memory_fraction)
    elif a.command=='extract':extract_development(a.run,a.features,a.fold)
    else:evaluate_development(a.features,a.report,a.fold,baseline=a.baseline)
