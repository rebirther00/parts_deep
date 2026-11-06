"""Abstention-aware metrics. All denominators include failed/abstained inputs."""
import collections
import math


def scores(rows, classes):
    n=len(rows); judged=[r for r in rows if r.get('pred') in classes]
    correct=sum(r.get('pred')==r['class_name'] for r in rows)
    per={}; matrix={c:{p:0 for p in classes+['ABSTAIN']} for c in classes}
    for r in rows:matrix[r['class_name']][r['pred'] if r.get('pred') in classes else 'ABSTAIN']+=1
    for c in classes:
        tp=matrix[c][c]; support=sum(matrix[c].values()); predicted=sum(matrix[t][c] for t in classes)
        precision=tp/predicted if predicted else 0.;recall=tp/support if support else 0.
        per[c]={'n':support,'correct':tp,'precision':precision,'recall':recall,
                'f1':2*precision*recall/(precision+recall) if precision+recall else 0.}
    return {'n':n,'judged':len(judged),'correct':correct,'abstained':n-len(judged),
            'coverage':len(judged)/n if n else None,'accuracy_all':correct/n if n else None,
            'accuracy_judged':correct/len(judged) if judged else None,
            'macro_f1':sum(v['f1'] for v in per.values())/len(classes),
            'balanced_accuracy':sum(v['recall'] for v in per.values())/len(classes),
            'per_class':per,'confusion':matrix}


def aggregate_sessions(rows):
    by=collections.defaultdict(list)
    for r in rows:by[r['session_dir']].append(r)
    out=[]
    for sid,rr in sorted(by.items()):
        labels={r['class_name'] for r in rr}
        if len(labels)!=1:raise ValueError('Mixed GT labels inside session')
        votes=collections.Counter(r['pred'] for r in rr if r.get('pred'))
        ranked=votes.most_common(); pred=None
        if ranked and (len(ranked)==1 or ranked[0][1]>ranked[1][1]):pred=ranked[0][0]
        out.append({'session_dir':sid,'group_id':rr[0]['group_id'],'class_name':rr[0]['class_name'],'pred':pred,
                    'n_frames':len(rr),'n_judged':sum(votes.values()),'votes':dict(votes)})
    return out


def wilson(success,n,z=1.959963984540054):
    if not n:return None
    p=success/n;den=1+z*z/n;center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,center-half),min(1.,center+half)]


def summarize(rows, classes):
    sessions=aggregate_sessions(rows);sm=scores(sessions,classes)
    sm['wilson95_accuracy_all']=wilson(sm['correct'],sm['n'])
    sm['ci_assumption']='Approximate independent sessions; repeated physical parts/lots may violate this assumption.'
    return {'frame':scores(rows,classes),'session':sm,'session_predictions':sessions}


def paired_session_comparison(a,b,classes,seed=20261106,repeats=2000):
    import numpy as np
    aa={r['frame_id']:r for r in a};bb={r['frame_id']:r for r in b}
    if set(aa)!=set(bb):raise ValueError('Comparison requires exactly identical frame sets')
    if any(aa[k]['class_name']!=bb[k]['class_name'] for k in aa):raise ValueError('GT mismatch')
    sa={r['session_dir']:r for r in aggregate_sessions(a)};sb={r['session_dir']:r for r in aggregate_sessions(b)}
    groups=collections.defaultdict(list);discordant=[0,0]
    for k,ra in sa.items():
        ca=int(ra['pred']==ra['class_name']);cb=int(sb[k]['pred']==sb[k]['class_name'])
        groups[ra['group_id']].append(ca-cb)
        if ca and not cb:discordant[0]+=1
        if cb and not ca:discordant[1]+=1
    vals=list(groups.values());rng=np.random.default_rng(seed);boot=[]
    for _ in range(repeats):
        draw=rng.integers(0,len(vals),len(vals));flat=[x for i in draw for x in vals[i]];boot.append(float(np.mean(flat)))
    n=sum(discordant);k=min(discordant)
    p=min(1.,2*sum(math.comb(n,j) for j in range(k+1))/(2**n)) if n else 1.
    return {'session_accuracy_difference':float(np.mean([x for v in vals for x in v])),
            'group_bootstrap95_difference':np.quantile(boot,[.025,.975]).tolist(),
            'discordant_a_only_b_only':discordant,'exact_mcnemar_p_session':p,'groups':len(groups),
            'note':'Bootstrap resamples groups. McNemar assumes independent sessions. No frame-wise significance claim.'}
