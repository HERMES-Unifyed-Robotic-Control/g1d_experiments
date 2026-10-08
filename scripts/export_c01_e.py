#!/usr/bin/env python3
"""Export native16 trained E cache and decoder; E normalization from train only."""
from pathlib import Path
import argparse,sys,json,hashlib
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from g1d_experiments.c01_codec import G1DC01Codec
from g1d_experiments.c01_evidence import G1DEvidence
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--train',type=Path,required=True);p.add_argument('--validation',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--batch-size',type=int,default=32);a=p.parse_args()
blob=torch.load(a.checkpoint,map_location='cpu',weights_only=False);model=G1DC01Codec(blob['statistics']);model.load_state_dict(blob['model']);model.eval().to(a.device);a.output.mkdir(parents=True,exist_ok=True)
sha=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest();cp=a.output/'codec.pt'
if cp.exists() and hashlib.sha256(cp.read_bytes()).hexdigest()!=sha:raise ValueError('Existing cache has different E checkpoint')
if not cp.exists():cp.write_bytes(a.checkpoint.read_bytes())
count=0;total=np.zeros(256,np.float64);squared=total.copy();manifest={'complete':False,'schema':blob['schema'],'checkpoint_sha256':sha,'E_shape':[30,256],'action_dim':16,'state_dim':16,'splits':{}}
for split,root in [('train',a.train),('validation',a.validation)]:
 data=G1DEvidence(root,a.evidence,split);dest=a.output/split;dest.mkdir(exist_ok=True);offset=0;records=[]
 for ep in data.episodes:
  n=ep['length'];path=dest/f'episode_{ep["episode_index"]:06d}.npy'
  if not path.exists():
   array=np.lib.format.open_memmap(path.with_suffix('.tmp.npy'),mode='w+',dtype=np.float32,shape=(n,30,256))
   with torch.inference_mode():
    for start in range(0,n,a.batch_size):
     ids=list(range(offset+start,offset+min(start+a.batch_size,n)));b=data.batch(ids,a.device);array[start:start+len(ids)]=model.encode(b).cpu().numpy()
   array.flush();del array;path.with_suffix('.tmp.npy').replace(path)
  x=np.load(path,mmap_mode='r')
  if x.shape!=(n,30,256) or not np.isfinite(x).all():raise ValueError('Invalid E cache')
  if split=='train':
   for start in range(0,n,32):
    xx=x[start:start+32].astype(np.float64);valid=(start+np.arange(len(xx)))[:,None]+np.arange(30)[None]<n;xx=xx[valid];count+=len(xx);total+=xx.sum(0);squared+=np.square(xx).sum(0)
  records.append({'episode_index':ep['episode_index'],'source_episode_index':ep['source_episode_index'],'length':n});offset+=n
  print(json.dumps({'event':'export','split':split,'episode':ep['episode_index'],'frames':n}),flush=True)
 manifest['splits'][split]={'root':str(root.resolve()),'frames':offset,'episodes':records}
mean=(total/count).astype(np.float32);std=np.sqrt(np.maximum(squared/count-np.square(total/count),0)).clip(1e-3).astype(np.float32);np.savez(a.output/'normalization.npz',mean=mean,std=std,fit_split='train',valid_tokens=count)
arrays={'decoder_16/'+k:v.cpu().numpy() for k,v in model.action_decoder.decoder.state_dict().items()}
for key in ('state_mean','state_std','action_mean','action_std'):arrays[key+'_16']=getattr(model.action_decoder,key).cpu().numpy()
arrays.update(effect_mean=mean,effect_std=std);np.savez(a.output/'decoder_init.npz',**arrays)
manifest.update(complete=True,normalization_sha256=hashlib.sha256((a.output/'normalization.npz').read_bytes()).hexdigest(),decoder_sha256=hashlib.sha256((a.output/'decoder_init.npz').read_bytes()).hexdigest());(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
