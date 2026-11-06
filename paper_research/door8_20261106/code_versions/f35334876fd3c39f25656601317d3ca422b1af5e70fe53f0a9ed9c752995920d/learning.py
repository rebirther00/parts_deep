from __future__ import annotations

import collections
import json
import random
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from .core import ROOT, SOURCE, check_source, digest, dump, jsonl, now, outpath, partition, read_json, read_jsonl, require_fresh_dir, sha256, snapshot
from .metrics import summarize

cv2.setNumThreads(0)
torch.set_num_threads(4)
MEAN=torch.tensor([.485,.456,.406])[:,None,None]
STD=torch.tensor([.229,.224,.225])[:,None,None]
POINT_KEYS=['bolt_tl','bolt_tr','bolt_bl','bolt_br','corner_hinge','corner_latch']


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


def worker_init(_):
    cv2.setNumThreads(0);torch.set_num_threads(1)
    s=torch.initial_seed()%2**32;random.seed(s);np.random.seed(s)


def pretrained_resnet():
    dst=outpath('pretrained/resnet18-f37072fd.pth')
    if not dst.exists():
        src=Path.home()/'.cache/torch/hub/checkpoints/resnet18-f37072fd.pth'
        if not src.is_file():raise FileNotFoundError('Place official ImageNet ResNet18 weights in '+str(dst))
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    return dst


def backbone(pretrained=True):
    r=models.resnet18(weights=None)
    if pretrained:r.load_state_dict(torch.load(pretrained_resnet(),map_location='cpu',weights_only=True))
    return r


class Classifier(nn.Module):
    def __init__(self,kind,nclasses=8,pretrained=True):
        super().__init__();r=backbone(pretrained)
        if kind=='rgbe':
            old=r.conv1;new=nn.Conv2d(4,64,7,stride=2,padding=3,bias=False)
            with torch.no_grad():new.weight[:,:3]=old.weight;new.weight[:,3:]=old.weight.mean(1,keepdim=True)
            r.conv1=new
        r.fc=nn.Identity();self.backbone=r
        self.head=nn.Sequential(nn.Dropout(.3),nn.Linear(512,256),nn.ReLU(),nn.Dropout(.2),nn.Linear(256,nclasses))
    def forward(self,x):return self.head(self.backbone(x))


class HoleNet(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__();r=backbone(pretrained)
        self.stem=nn.Sequential(r.conv1,r.bn1,r.relu);self.l1=nn.Sequential(r.maxpool,r.layer1)
        self.l2,self.l3,self.l4=r.layer2,r.layer3,r.layer4
        self.lat4=nn.Conv2d(512,128,1);self.lat3=nn.Conv2d(256,128,1)
        self.lat2=nn.Conv2d(128,128,1);self.lat1=nn.Conv2d(64,128,1)
        self.head=nn.Sequential(nn.Conv2d(128,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),
                               nn.Conv2d(128,64,3,padding=1),nn.ReLU(),nn.Conv2d(64,3,1))
    def forward(self,x):
        c1=self.l1(self.stem(x));c2=self.l2(c1);c3=self.l3(c2);c4=self.l4(c3)
        p=self.lat4(c4)
        for c,layer in [(c3,self.lat3),(c2,self.lat2),(c1,self.lat1)]:
            p=F.interpolate(p,size=c.shape[2:],mode='bilinear',align_corners=False)+layer(c)
        return self.head(p)


def letterbox(im,w,h):
    ih,iw=im.shape[:2];s=min(w/iw,h/ih)
    M=np.array([[s,0,(w-iw*s)/2],[0,s,(h-ih*s)/2]],np.float32)
    return cv2.warpAffine(im,M,(w,h),borderValue=(114,114,114)),M


def tensor_image(im):
    return (torch.from_numpy(im[:,:,::-1].copy()).permute(2,0,1).float()/255-MEAN)/STD


class CNNData(Dataset):
    def __init__(self,rows,kind,size,train):self.rows,self.kind,self.size,self.train=rows,kind,size,train
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i];check_source(r);im=cv2.imread(r['rgb'])
        if im is None:raise RuntimeError('Unreadable RGB: '+r['rgb'])
        im,_=letterbox(im,self.size,self.size)
        if self.train:
            M=cv2.getRotationMatrix2D((self.size/2,self.size/2),random.uniform(-8,8),random.uniform(.9,1.1))
            im=cv2.warpAffine(im,M,(self.size,self.size),borderValue=(114,114,114))
            # LH/RH labels can change under reflection: do not flip class images.
            im=np.clip(im.astype(np.float32)*random.uniform(.8,1.2)+random.uniform(-20,20),0,255).astype(np.uint8)
            if random.random()<.2:im=cv2.GaussianBlur(im,(3,3),0)
        x=tensor_image(im)
        if self.kind=='rgbe':
            e=cv2.Canny(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),80,200)
            x=torch.cat([x,torch.from_numpy(e.copy()).float()[None]/127.5-1],0)
        return x,r['class_index'],i


def make_loader(ds,batch,shuffle,workers,seed):
    g=torch.Generator().manual_seed(seed)
    return DataLoader(ds,batch_size=batch,shuffle=shuffle,num_workers=workers,pin_memory=torch.cuda.is_available(),
                      worker_init_fn=worker_init,generator=g,persistent_workers=workers>0)


def stratified_limit(rows,n,seed):
    if not n:return rows
    by=collections.defaultdict(list);rng=random.Random(seed)
    for r in rows:by[r['class_name']].append(r)
    out=[]
    for rr in by.values():rng.shuffle(rr);out+=rr[:n]
    return out


@torch.no_grad()
def cnn_predict(net,loader,rows,classes,dev,threshold=0.):
    net.eval();out=[];loss=0.;n=0
    for x,y,idx in loader:
        x=x.to(dev);y=y.to(dev)
        with torch.autocast(device_type=dev.type,enabled=dev.type=='cuda'):logits=net(x)
        loss+=float(F.cross_entropy(logits.float(),y,reduction='sum'));n+=len(y)
        probs=logits.float().softmax(1).cpu().numpy()
        for j,p in zip(idx.tolist(),probs):
            r=rows[j];k=int(p.argmax());confidence=float(p[k])
            out.append({key:r[key] for key in ['frame_id','session_dir','group_id','class_name']}|
                       {'pred':classes[k] if confidence>=threshold else None,'confidence':confidence,'probabilities':p.tolist()})
    return out,loss/max(n,1)


def train_cnn(name,kind,fold,seed,run,epochs=None,workers=4,pilot_per_class=0):
    sp,summary,rows,splits=snapshot(name);cfg=read_json(sp/'protocol.json');parts=partition(rows,splits,fold)
    runpath=require_fresh_dir('runs/'+run);seed_all(seed)
    tr=stratified_limit(parts['train'],pilot_per_class,seed);va=stratified_limit(parts['val'],pilot_per_class,seed)
    hp=cfg['cnn'];dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net=Classifier(kind,len(cfg['classes'])).to(dev)
    dl=make_loader(CNNData(tr,kind,hp['image_size'],True),hp['batch_size'],True,workers,seed)
    vl=make_loader(CNNData(va,kind,hp['image_size'],False),hp['batch_size'],False,workers,seed)
    count=collections.Counter(r['class_index'] for r in tr)
    weights=torch.tensor([(len(tr)/max(count[i],1))**.5 for i in range(len(cfg['classes']))],device=dev)
    weights/=weights.mean();opt=torch.optim.Adam(net.parameters(),lr=hp['learning_rate'])
    scaler=torch.amp.GradScaler('cuda',enabled=dev.type=='cuda')
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=3)
    meta={'model_kind':kind,'snapshot':name,'snapshot_sha256':sha256(sp/'COMPLETE.json'),'protocol_sha256':summary['protocol_sha256'],
          'fold':str(fold),'seed':seed,'classes':cfg['classes'],'image_size':hp['image_size'],
          'training_regime':'field_class_labels','initialization':'ImageNet1K_V1','initialization_sha256':sha256(pretrained_resnet()),
          'train_frame_ids':[r['frame_id'] for r in tr],'val_frame_ids':[r['frame_id'] for r in va],
          'train_group_ids':sorted({r['group_id'] for r in tr}),'val_group_ids':sorted({r['group_id'] for r in va}),
          'pilot':bool(pilot_per_class or (epochs is not None and epochs!=hp['epochs'])),'requested_epochs':epochs or hp['epochs'],
          'started_at':now(),'status':'running','hyperparameters':hp,'device':str(dev),
          'code_sha256':sha256(__file__)}
    dump(runpath/'run.json',meta);hist=[];best=(-1.,float('-inf'));wait=0;t0=time.perf_counter()
    for ep in range(epochs or hp['epochs']):
        net.train();total=0.;seen=0;et=time.perf_counter()
        for x,y,_ in dl:
            x=x.to(dev,non_blocking=True);y=y.to(dev,non_blocking=True);opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type,enabled=dev.type=='cuda'):loss=F.cross_entropy(net(x),y,weight=weights)
            scaler.scale(loss).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_(net.parameters(),1.)
            scaler.step(opt);scaler.update();total+=float(loss.detach())*len(y);seen+=len(y)
        pred,vloss=cnn_predict(net,vl,va,cfg['classes'],dev)
        met=summarize(pred,cfg['classes']);score=(met['session']['balanced_accuracy'],-vloss)
        scheduler.step(vloss)
        record={'epoch':ep+1,'train_loss':total/seen,'val_loss':vloss,'val_session_balanced_accuracy':score[0],
                'seconds':time.perf_counter()-et};hist.append(record);dump(runpath/'history.json',hist)
        if score>best:
            best=score;wait=0;torch.save(net.state_dict(),outpath(runpath/'model.pth'));jsonl(runpath/'val_predictions.jsonl',pred)
            meta.update(best_epoch=ep+1,best_val_session_balanced_accuracy=score[0]);dump(runpath/'run.json',meta)
        else:wait+=1
        print(json.dumps(record),flush=True)
        if wait>=hp['patience']:break
    meta.update(status='complete',finished_at=now(),seconds=time.perf_counter()-t0,weights_sha256=sha256(runpath/'model.pth'))
    dump(runpath/'run.json',meta);return meta


class HoleData(Dataset):
    def __init__(self,rows,hp,train):self.rows,self.hp,self.train=rows,hp,train
    def __len__(self):return len(self.rows)*(4 if self.train else 1)
    def __getitem__(self,i):
        r=self.rows[i%len(self.rows)];im=cv2.imread(r['rgb'])
        if im is None:raise RuntimeError(r['rgb'])
        w,h=self.hp['width'],self.hp['height'];_,M=letterbox(im,w,h)
        if self.train:
            A=cv2.getRotationMatrix2D((w/2,h/2),random.uniform(-180,180),random.uniform(.8,1.15))
            A[:,2]+=np.array([random.uniform(-50,50),random.uniform(-35,35)])
            M=(A@np.vstack([M,[0,0,1]])).astype(np.float32)
            if random.random()<.5:M=(np.array([[-1,0,w-1],[0,1,0]],np.float32)@np.vstack([M,[0,0,1]])).astype(np.float32)
        image=cv2.warpAffine(im,M,(w,h),borderValue=(114,114,114))
        if self.train:
            image=np.clip(image.astype(np.float32)*random.uniform(.7,1.3)+random.uniform(-30,30),0,255).astype(np.uint8)
            if random.random()<.3:image=cv2.GaussianBlur(image,(3,3),0)
        stride=self.hp['stride'];ys,xs=np.mgrid[:h//stride,:w//stride];hm=np.zeros((3,h//stride,w//stride),np.float32)
        valid=np.ones(3,np.float32)
        for ci,keys in enumerate([POINT_KEYS[:4],['corner_hinge'],['corner_latch']]):
            # Missing or explicitly invisible points are not asserted to be absent.
            # Supervise a channel only when its complete required point set is annotated.
            if any(r['points'].get(k) is None or not r.get('visible',{}).get(k,True) for k in keys):valid[ci]=0.;continue
            for key in keys:
                p=M@np.array([*r['points'][key],1.]);x,y=p/stride
                if 0<=x<w/stride and 0<=y<h/stride:
                    hm[ci]=np.maximum(hm[ci],np.exp(-((xs-x)**2+(ys-y)**2)/(2*self.hp['sigma']**2)))
        return tensor_image(image),torch.from_numpy(hm),torch.from_numpy(valid)


def heatmap_loss(pred,target,mask):
    per=((pred-target)**2*(1+20*target)).mean((2,3))
    return (per*mask).sum()/mask.sum().clamp(min=1)


def train_holes(name,seed,run,epochs=None,workers=2,pilot=False):
    sp,summary,_,_=snapshot(name);cfg=read_json(sp/'protocol.json');hp=cfg['holes']
    labels=read_jsonl(sp/'annotations_lab.jsonl');split=read_json(sp/'annotations_lab_split.json')
    for r in labels:
        if r['source']!='human_manual_lab' or sha256(r['rgb'])!=r['rgb_sha256']:raise ValueError('LAB annotation source mismatch')
    tr=[r for r in labels if r['annotation_id'] in split['train']];va=[r for r in labels if r['annotation_id'] in split['val']]
    if pilot:tr=tr[:8];va=va[:4]
    runpath=require_fresh_dir('runs/'+run);seed_all(seed);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net=HoleNet().to(dev);opt=torch.optim.AdamW(net.parameters(),lr=hp['learning_rate'])
    dl=make_loader(HoleData(tr,hp,True),hp['batch_size'],True,workers,seed)
    vl=make_loader(HoleData(va,hp,False),hp['batch_size'],False,workers,seed)
    total_epochs=epochs or hp['epochs'];scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,total_epochs)
    scaler=torch.amp.GradScaler('cuda',enabled=dev.type=='cuda')
    meta={'model_kind':'holes','snapshot':name,'snapshot_sha256':sha256(sp/'COMPLETE.json'),'protocol_sha256':summary['protocol_sha256'],
          'seed':seed,'training_regime':'human_manual_lab_only','initialization':'ImageNet1K_V1',
          'initialization_sha256':sha256(pretrained_resnet()),'train_annotation_ids':[r['annotation_id'] for r in tr],
          'val_annotation_ids':[r['annotation_id'] for r in va],'train_group_ids':[],'val_group_ids':[],
          'pilot':bool(pilot or total_epochs!=hp['epochs']),'requested_epochs':total_epochs,'hyperparameters':hp,'status':'running','started_at':now(),
          'code_sha256':sha256(__file__)}
    dump(runpath/'run.json',meta);hist=[];best=float('inf');t0=time.perf_counter()
    for ep in range(total_epochs):
        net.train();tl=[]
        for x,y,m in dl:
            x,y,m=x.to(dev),y.to(dev),m.to(dev);opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type,enabled=dev.type=='cuda'):loss=heatmap_loss(net(x),y,m)
            scaler.scale(loss).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_(net.parameters(),1.)
            scaler.step(opt);scaler.update();tl.append(float(loss.detach()))
        net.eval();vloss=[]
        with torch.no_grad():
            for x,y,m in vl:
                with torch.autocast(device_type=dev.type,enabled=dev.type=='cuda'):loss=heatmap_loss(net(x.to(dev)),y.to(dev),m.to(dev))
                vloss.append(float(loss))
        v=float(np.mean(vloss));rec={'epoch':ep+1,'train_loss':float(np.mean(tl)),'val_loss':v};hist.append(rec)
        if v<best:best=v;meta['best_epoch']=ep+1;torch.save(net.state_dict(),outpath(runpath/'model.pth'))
        scheduler.step();dump(runpath/'history.json',hist);dump(runpath/'run.json',meta);print(json.dumps(rec),flush=True)
    meta.update(status='complete',finished_at=now(),seconds=time.perf_counter()-t0,weights_sha256=sha256(runpath/'model.pth'))
    dump(runpath/'run.json',meta);return meta


def load_run(run,sp,allow_pilot=False):
    rp=outpath('runs/'+run);meta=read_json(rp/'run.json')
    if meta['status']!='complete':raise ValueError('Training is incomplete')
    if meta.get('pilot') and not allow_pilot:raise ValueError('Pilot weights cannot enter a publication experiment')
    if meta['snapshot_sha256']!=sha256(sp/'COMPLETE.json'):raise ValueError('Model/snapshot mismatch')
    if sha256(rp/'model.pth')!=meta['weights_sha256']:raise ValueError('Weights changed after training')
    return rp,meta
