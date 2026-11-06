#!/usr/bin/env python3
"""All outputs, including caches, are confined to this isolated paper folder."""
import argparse
from door8.core import setup_runtime


def main():
    setup_runtime();p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot',default='v1')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('freeze')
    a=sub.add_parser('audit');a.add_argument('--full-hash',action='store_true')
    a=sub.add_parser('train-cnn');a.add_argument('--kind',choices=['rgb','rgbe'],required=True);a.add_argument('--fold',default='0')
    a.add_argument('--seed',type=int,default=42);a.add_argument('--run',required=True);a.add_argument('--epochs',type=int)
    a.add_argument('--workers',type=int,default=4);a.add_argument('--pilot-per-class',type=int,default=0)
    a=sub.add_parser('train-holes');a.add_argument('--seed',type=int,default=42);a.add_argument('--run',required=True)
    a.add_argument('--epochs',type=int);a.add_argument('--workers',type=int,default=2);a.add_argument('--pilot',action='store_true')
    a=sub.add_parser('extract');a.add_argument('--kind',choices=['holes','mobile_sam'],required=True);a.add_argument('--features',required=True)
    a.add_argument('--run');a.add_argument('--limit',type=int,default=0);a.add_argument('--fold',default='0')
    a=sub.add_parser('eval-cnn');a.add_argument('--run',required=True);a.add_argument('--report',required=True)
    a.add_argument('--workers',type=int,default=4);a.add_argument('--pilot',action='store_true')
    a=sub.add_parser('eval-geometry');a.add_argument('--features',required=True);a.add_argument('--fold',default='0');a.add_argument('--report',required=True)
    a.add_argument('--mode',choices=['pixel','depth','mask'],default='pixel');a.add_argument('--no-gates',action='store_true');a.add_argument('--pose',action='store_true')
    a=sub.add_parser('compare');a.add_argument('--reports',nargs='+',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('suite');a.add_argument('--execute',action='store_true');a.add_argument('--temporal',action='store_true')
    a=sub.add_parser('collect-oof');a.add_argument('--reports',nargs=5,required=True);a.add_argument('--output',required=True)
    args=p.parse_args();s=args.snapshot
    if args.command=='freeze':
        from door8.data import freeze
        freeze(s)
    elif args.command=='audit':
        from door8.data import audit
        audit(s,args.full_hash)
    elif args.command=='train-cnn':
        from door8.learning import train_cnn
        train_cnn(s,args.kind,args.fold,args.seed,args.run,args.epochs,args.workers,args.pilot_per_class)
    elif args.command=='train-holes':
        from door8.learning import train_holes
        train_holes(s,args.seed,args.run,args.epochs,args.workers,args.pilot)
    elif args.command=='extract':
        from door8.features import extract
        extract(s,args.kind,args.features,args.run,args.limit,args.fold)
    elif args.command=='eval-cnn':
        from door8.evaluate import evaluate_cnn
        evaluate_cnn(s,args.run,args.report,args.workers,args.pilot)
    elif args.command=='eval-geometry':
        from door8.evaluate import evaluate_geometry
        evaluate_geometry(s,args.features,args.fold,args.report,args.mode,not args.no_gates,args.pose)
    elif args.command=='compare':
        from door8.evaluate import compare
        compare(s,args.reports,args.output)
    elif args.command=='suite':
        from door8.suite import suite
        suite(s,args.execute,args.temporal)
    elif args.command=='collect-oof':
        from door8.suite import collect_oof
        collect_oof(s,args.reports,args.output)


if __name__=='__main__':main()
