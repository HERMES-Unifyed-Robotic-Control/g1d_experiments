#!/usr/bin/env python3
"""G1D v2.1 video -> stride5 DINO patches -> train PCA64 -> C01 world targets."""
from pathlib import Path
import sys,os,argparse,json
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'third_party/dexjoco'))
import numpy as np
import torch
import cv2
from g1d_experiments.c01_contract import read_episodes
from afce_all11.build_world import match_intervals
from effect_vla.effect.dino_extractor import FrozenDINOv3
p=argparse.ArgumentParser();p.add_argument('--train',type=Path,required=True);p.add_argument('--validation',type=Path,required=True);p.add_argument('--dino',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.manual_seed(42)
contract={'schema':'g1d_c01_visual_v1','train':str(a.train.resolve()),'validation':str(a.validation.resolve()),'dino':str(a.dino.resolve()),'view':'egocentric','stride':5,'mask_protocol':'legacy-none_explicitly_unavailable','pca_fit':'train_only','dino_image_size':224}
cp=a.output/'visual_contract.json'
if cp.exists() and json.loads(cp.read_text())!=contract:raise ValueError('Evidence contract changed; choose new output')
cp.write_text(json.dumps(contract,indent=2)+'\n');backbone=FrozenDINOv3(a.dino,image_size=224,device=a.device);paths={'train':[],'validation':[]}
for split,root in [('train',a.train),('validation',a.validation)]:
 info,episodes=read_episodes(root)
 for ep in episodes:
  e=ep['episode_index'];dest=a.output/split/f'episode_{e:06d}';dest.mkdir(parents=True,exist_ok=True);paths[split].append(dest)
  if (dest/'features.npy').exists() and (dest/'frames.npy').exists():continue
  ids=np.arange(0,ep['length'],5);video=root/info['video_path'].format(episode_index=e,episode_chunk=e//info.get('chunks_size',1000),video_key='observation.images.egocentric')
  cap=cv2.VideoCapture(str(video));imgs=[];features=[];cursor=0
  try:
   for idx in ids:
    while cursor<=idx:
     ok,bgr=cap.read();cursor+=1
     if not ok:raise ValueError(f'Missing recorded frame:{video}/{idx}')
    # DINO has its own 224 resize and normalization, independent of pi letterbox.
    imgs.append(cv2.resize(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),(224,224)))
    if len(imgs)==16:
     features.append(backbone.encode_numpy(np.stack(imgs)));imgs=[]
   if imgs:features.append(backbone.encode_numpy(np.stack(imgs)))
  finally:cap.release()
  temp=dest/'features.tmp.npy';np.save(temp,np.concatenate(features));temp.replace(dest/'features.npy');np.save(dest/'frames.npy',ids)
  print(json.dumps({'event':'features','split':split,'episode':e,'sampled_frames':len(ids)}),flush=True)
del backbone
if torch.cuda.is_available():torch.cuda.empty_cache()
pp=a.output/'fitted_pca.npz'
if not pp.exists():
 rng=np.random.default_rng(42);samples=[]
 for path in paths['train']:
  f=np.load(path/'features.npy',mmap_mode='r')
  for anchor in rng.integers(0,len(f),2):samples.append(f[anchor,rng.choice(f.shape[1],8,replace=False)].astype(np.float32))
 x=torch.from_numpy(np.concatenate(samples));mean=x.mean(0);_,_,v=torch.pca_lowrank(x-mean,q=64,center=False,niter=4)
 np.savez(pp,mean=mean.numpy(),weight=v.T.numpy(),fit_split='train',seed=42)
with np.load(pp) as z:weight=torch.tensor(z['weight'],device=a.device);mean=torch.tensor(z['mean'],device=a.device)
xy=torch.stack(torch.meshgrid(torch.arange(14),torch.arange(14),indexing='ij'),-1).reshape(-1,2)[:,[1,0]].to(a.device).float()/14+1/28;nearby=torch.cdist(xy,xy)<=.2
for split,items in paths.items():
 for path in items:
  if (path/'world_intervals.npz').exists():continue
  f=np.load(path/'features.npy',mmap_mode='r');frames=np.load(path/'frames.npy');raws=[];valids=[]
  if f.shape[1]!=196:raise ValueError('Expected224px/16px196 DINO patches')
  with torch.inference_mode():
   for lo in range(0,len(f)-1,32):
    hi=min(lo+32,len(f)-1);batch=torch.tensor(f[lo:hi+1].astype(np.float32),device=a.device)
    # Missing masks are explicitly recorded; legacy-none matches upstream protocol.
    raw,valid=match_intervals(batch[:-1],batch[1:],torch.zeros((hi-lo,196),device=a.device),weight,xy,nearby,mean)
    raws.append(raw.cpu().numpy());valids.append(valid.cpu().numpy())
  raw=np.concatenate(raws) if raws else np.empty((0,8,136),np.float32);valid=np.concatenate(valids) if valids else np.empty((0,8),np.float32)
  np.savez_compressed(path/'world_intervals.npz',start=frames[:-1],end=frames[1:],raw=raw,valid=valid)
  print(json.dumps({'event':'world','split':split,'episode':path.name}),flush=True)
(a.output/'world_complete.json').write_text(json.dumps(dict(contract,complete=True),indent=2)+'\n')
