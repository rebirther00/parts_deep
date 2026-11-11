"""Label-free measurements and explicit train-only calibration.

No operational module, historical calibration constant, or trained field model
is imported. Camera intrinsics are supplied separately for every observation.
"""
from collections import defaultdict
import cv2
import numpy as np


def bolt_frame(points):
    if len(points)!=4:return None
    p=np.asarray(points,float)[:,:2];c=p.mean(0)
    pairs=sorted((float(np.linalg.norm(p[i]-p[j])),i,j) for i in range(4) for j in range(i+1,4))
    _,i,j=pairs[2];ex=p[j]-p[i];norm=np.linalg.norm(ex)
    if norm<1:return None
    ex/=norm;ey=np.array([-ex[1],ex[0]]);q=p-c
    s1=np.ptp(q@ex)/157.;s2=np.ptp(q@ey)/96.
    if min(s1,s2)<=0 or abs(s1-s2)/max(s1,s2)>.18:return None
    return {'c':c,'ex':ex,'ey':ey,'scale':(s1+s2)/2}


def geometry_gate(frame,hinge,latch,shape):
    if hinge is None or latch is None:return 'missing_corner'
    if frame is None:return 'invalid_bolt_rectangle'
    h,w=shape[:2]
    if any(p[0]<20 or p[1]<20 or p[0]>w-20 or p[1]>h-20 for p in (hinge,latch)):return 'near_border'
    qh=np.asarray(hinge[:2])-frame['c'];ql=np.asarray(latch[:2])-frame['c'];s=frame['scale']
    uh,wh=qh@frame['ex']/s,qh@frame['ey']/s;ul,wl=ql@frame['ex']/s,ql@frame['ey']/s
    if not (100<=abs(ul)<=230 and 120<=abs(wl)<=270):return 'latch_offset'
    if abs(wh-wl)>70:return 'not_collinear'
    if np.sign(uh)==np.sign(ul):return 'same_side'
    if not 400<=abs(uh-ul)<=1500:return 'distance_range'
    return 'ok'


def depth_measure(depth,points,hinge,latch,intr):
    if depth is None or len(points)<3:return None
    mask=np.zeros(depth.shape,np.uint8)
    xy=np.ascontiguousarray(np.asarray(points,np.float32)[:,:2])
    cv2.fillConvexPoly(mask,cv2.convexHull(xy).astype(np.int32),255)
    mask=cv2.erode(mask,np.ones((15,15),np.uint8));ys,xs=np.where((mask>0)&(depth>0))
    if len(xs)<500:return None
    if len(xs)>40000:
        idx=np.random.default_rng(0).choice(len(xs),40000,replace=False);xs,ys=xs[idx],ys[idx]
    z=depth[ys,xs].astype(float)
    q=np.column_stack(((xs-intr['cx'])*z/intr['fx'],(ys-intr['cy'])*z/intr['fy'],z))
    c=q.mean(0);_,_,v=np.linalg.svd(q-c,full_matrices=False);n=v[-1]
    keep=np.abs((q-c)@n)<10
    if keep.sum()<200:return None
    q=q[keep];c=q.mean(0);_,_,v=np.linalg.svd(q-c,full_matrices=False);n=v[-1]
    def xyz(p):
        r=np.array([(p[0]-intr['cx'])/intr['fx'],(p[1]-intr['cy'])/intr['fy'],1.])
        den=r@n
        return None if abs(den)<1e-8 else r*((c@n)/den)
    a,b=xyz(hinge),xyz(latch)
    if a is None or b is None or min(a[2],b[2])<=0:return None
    d=float(np.linalg.norm(a-b))
    if not np.isfinite(d):return None
    return {'depth_distance_raw_mm':d,'plane_z_mm':float((a[2]+b[2])/2),
            'plane_tilt_deg':float(np.degrees(np.arccos(np.clip(abs(n[2]),0,1)))),
            'plane_inlier_count':int(len(q))}


def fit_scale(train_rows,features,cad,mode):
    """Caller must pass the TRAIN partition, never an unfiltered manifest."""
    values=defaultdict(list);by_camera=defaultdict(list)
    for r in train_rows:
        f=features[r['frame_id']]
        if f.get('gate')!='ok':continue
        raw=f.get('span_normalized') if mode=='pixel' else f.get('depth_distance_raw_mm')
        if raw is not None and raw>0 and np.isfinite(raw):
            values[(str(r['intrinsics'].get('serial','unknown')),r['session_dir'])].append(cad[r['class_name']]/raw)
    for (cam,_),v in values.items():by_camera[cam].append(float(np.median(v)))
    if not values:raise ValueError('No valid TRAIN measurements for calibration; do not use validation/test to repair this.')
    return {'mode':mode,'camera_scales':{k:float(np.median(v)) for k,v in by_camera.items()},
            'train_frame_ids':[r['frame_id'] for r in train_rows],
            'train_group_ids':sorted({r['group_id'] for r in train_rows}),
            'valid_train_sessions':len(values),'aggregation':'median of session medians; per camera'}


def classify_measurement(row,feature,calibration,cfg,gates=True,pose=False):
    reason=feature.get('gate','missing_measurement');g=cfg['geometry']
    if gates and reason!='ok':return None,None,reason
    if pose:
        z,t=feature.get('plane_z_mm'),feature.get('plane_tilt_deg')
        if z is None or t is None:return None,None,'missing_pose'
        if not g['pose_z_mm'][0]<=z<=g['pose_z_mm'][1] or t>g['pose_tilt_deg']:return None,None,'pose_gate'
    key='span_normalized' if calibration['mode']=='pixel' else 'depth_distance_raw_mm'
    raw=feature.get(key);cam=str(row['intrinsics'].get('serial','unknown'))
    if raw is None:return None,None,'missing_measurement'
    if cam not in calibration['camera_scales']:return None,None,'uncalibrated_camera'
    d=raw*calibration['camera_scales'][cam]
    if not np.isfinite(d) or not g['valid_distance_mm'][0]<=d<=g['valid_distance_mm'][1]:return None,None,'distance_range'
    ds=np.abs(np.asarray(cfg['cad_hole_distance_mm'])-d);i=int(ds.argmin())
    if ds[i]>g['unknown_distance_mm']:return None,float(d),'cad_distance_reject'
    return cfg['classes'][i],float(d),'ok'


def mask_descriptor(mask,intr):
    ys,xs=np.where(mask)
    if len(xs)<20:return None
    # Normalized camera coordinates avoid mixing focal lengths; all prompts fixed.
    p=np.column_stack(((xs-intr['cx'])/intr['fx'],(ys-intr['cy'])/intr['fy'])).astype(np.float32)
    if len(p)>20000:p=p[np.linspace(0,len(p)-1,20000,dtype=int)]
    _,size,_=cv2.minAreaRect(p);short,long=sorted(size)
    if short<=1e-6:return None
    fill=float(len(xs)/(intr['fx']*intr['fy']*short*long))
    return [float(np.log(long)),float(np.log(short)),float(np.log(long/short)),fill]


def fit_mask(train_rows,features,classes):
    by=defaultdict(list)
    for r in train_rows:
        v=features[r['frame_id']].get('descriptor')
        if v is not None:by[(r['class_name'],r['session_dir'])].append(v)
    sessions=[(c,np.median(v,axis=0)) for (c,_),v in by.items()]
    if any(not any(c==k for c,_ in sessions) for k in classes):raise ValueError('Missing class in TRAIN mask features')
    values=np.asarray([v for _,v in sessions]);scale=np.std(values,axis=0);scale=np.maximum(scale,1e-3)
    centers={c:np.median([v for k,v in sessions if c==k],axis=0).tolist() for c in classes}
    return {'centers':centers,'scale':scale.tolist(),'train_frame_ids':[r['frame_id'] for r in train_rows],
            'train_group_ids':sorted({r['group_id'] for r in train_rows})}


def predict_mask(feature,calibration):
    if feature.get('descriptor') is None:return None,None,'mask_failure'
    classes=list(calibration['centers']);x=np.asarray(feature['descriptor']);scale=np.asarray(calibration['scale'])
    dist=[float(np.linalg.norm((x-calibration['centers'][c])/scale)) for c in classes];i=int(np.argmin(dist))
    return classes[i],dist[i],'ok'
