#!/usr/bin/env python3
"""Train native G1D Query-ASV E codec with grouped episode holdout."""
from pathlib import Path
import sys,argparse,json,random,time,os,hashlib
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from g1d_experiments.c01_evidence import G1DEvidence
from g1d_experiments.c01_codec import G1DC01Codec
from g1d_experiments.c01_contract import SCHEMA
p=argparse.ArgumentParser();p.add_argument('--train',type=Path,required=True);p.add_argument('--validation',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--statistics',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=30000);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--device',default='cuda:0');p.add_argument('--resume',action='store_true');a=p.parse_args()
if a.steps<1 or a.batch_size<1:raise ValueError('Positive training steps/batch required')
random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(4)
train=G1DEvidence(a.train,a.evidence,'train');val=G1DEvidence(a.validation,a.evidence,'validation')
if {e['source_episode_index'] for e in train.episodes}&{e['source_episode_index'] for e in val.episodes}:raise ValueError('Source episode leakage')
a.output.mkdir(parents=True,exist_ok=True);stats=json.loads(a.statistics.read_text())
contract={'schema':SCHEMA,'train':str(a.train.resolve()),'validation':str(a.validation.resolve()),'evidence':str(a.evidence.resolve()),'stats_sha256':hashlib.sha256(a.statistics.read_bytes()).hexdigest(),'batch_size':a.batch_size,'seed':42,'fusion':'query','E_shape':[30,256],'robot_response':'consecutive_measured_native16_joint_delta','mask_protocol':'legacy-none_explicitly_unavailable','lr':[1e-4,3e-5],'switch_step':20000}
if (a.output/'contract.json').exists() and json.loads((a.output/'contract.json').read_text())!=contract:raise ValueError('Resume contract changed')
(a.output/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
# World statistics use valid training intervals only.
if a.resume:
 blob=torch.load(a.output/'last.pt',map_location='cpu',weights_only=False);stats=blob['statistics']
else:
 if (a.output/'last.pt').exists():raise ValueError('Existing run requires --resume')
 count=0;total=np.zeros(136,np.float64);squared=total.copy()
 for ep in train.episodes:
  with np.load(train.evidence/f'episode_{ep["episode_index"]:06d}'/'world_intervals.npz') as z:
   x=z['raw'][z['valid']>0].astype(np.float64);count+=len(x);total+=x.sum(0);squared+=np.square(x).sum(0)
 if count==0:raise ValueError('No valid training visual consequences')
 mean=total/count;std=np.sqrt(np.maximum(squared/count-mean*mean,0)).clip(1e-3);mean[70:72]=0;std[70:72]=1
 stats.update(world_mean=mean.tolist(),world_std=std.tolist())
model=G1DC01Codec(stats).to(a.device);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.9,.99),weight_decay=.01)
rng=np.random.default_rng(42);step=0;best=float('inf')
if a.resume:
 model.load_state_dict(blob['model']);optimizer.load_state_dict(blob['optimizer']);step=blob['step'];best=blob['best'];rng.bit_generator.state=blob['sample_rng'];random.setstate(blob['python_rng']);np.random.set_state(blob['numpy_rng']);torch.set_rng_state(blob['torch_rng'])
 if torch.cuda.is_available() and blob.get('cuda_rng') is not None:torch.cuda.set_rng_state_all(blob['cuda_rng'])
fixed=[];offset=0
for ep in val.episodes:
 for fraction in (.25,.75):fixed.append(offset+int(max(0,ep['length']-30)*fraction))
 offset+=ep['length']
(a.output/'validation_indices.json').write_text(json.dumps(fixed))
def save(name):
 state={'schema':SCHEMA,'step':step,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'statistics':stats,'contract':contract,'best':best,'sample_rng':rng.bit_generator.state,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
 temp=a.output/(name+'.tmp');torch.save(state,temp);os.replace(temp,a.output/name)
while step<a.steps:
 step+=1;lr=1e-4 if step<=20000 else 3e-5
 for group in optimizer.param_groups:group['lr']=lr
 model.train();b=train.batch(rng.integers(0,len(train),a.batch_size),a.device);optimizer.zero_grad(set_to_none=True)
 weight=.05+(.2-.05)*min(step/1000,1);losses=model.losses(model(b),b,weight,weight)
 if not all(torch.isfinite(x) for x in losses.values()):raise FloatingPointError('Nonfinite E loss')
 losses['loss'].backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
 if not torch.isfinite(grad):raise FloatingPointError('Nonfinite E gradient')
 optimizer.step()
 if step%10==0:
  record={'step':step,'lr':lr,**{k:float(v.detach()) for k,v in losses.items()}}
  with (a.output/'training_metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
  print(json.dumps(record),flush=True)
 if step%1000==0 or step==a.steps:
  model.eval();values=[]
  with torch.inference_mode():
   for i in range(0,len(fixed),a.batch_size):
    ids=fixed[i:i+a.batch_size];b=val.batch(ids,a.device);v=model.losses(model(b),b);values.append((len(ids),{k:float(x) for k,x in v.items()}))
  record={'step':step,'observations':len(fixed),**{k:sum(n*v[k] for n,v in values)/len(fixed) for k in values[0][1]}}
  with (a.output/'validation_metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
  if record['loss']<best:best=record['loss'];save('best.pt')
 if step%250==0 or step==a.steps:save('last.pt')
(a.output/'TRAINING_COMPLETE.json').write_text(json.dumps({'steps':step,'best_validation_loss':best}))
