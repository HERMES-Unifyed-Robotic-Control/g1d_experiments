#!/usr/bin/env python3
"""Fixed holdout action error for native16 C01 pi; no future inputs at inference."""
from pathlib import Path
import os,sys,argparse,dataclasses,json,hashlib
ROOT=Path(__file__).parents[1];os.environ.setdefault('PSI_HOME',str(ROOT/'vendor/openpi'));os.environ.setdefault('DATA_HOME',str(ROOT/'artifacts/data'));os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false');sys.path[:0]=[str(ROOT),str(ROOT/'vendor/openpi/src'),str(ROOT/'vendor/openpi/src/openpi/openpi-client/src')]
import numpy as np
import jax
from openpi.training import config
from openpi.models import model as model_lib
from g1d_experiments.c01_pi import G1DC01PiConfig
from g1d_experiments.c01_pi_data import G1DPiData
p=argparse.ArgumentParser();p.add_argument('--params',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--validation',type=Path,required=True);p.add_argument('--norm-stats',type=Path,required=True);p.add_argument('--variant',choices=['lora','fullft'],default='lora');p.add_argument('--output',type=Path,required=True);p.add_argument('--denoise-steps',type=int,default=10);a=p.parse_args()
base=config.get_config('g1d_subtasks_pi05_official_lora_holdout_v1');v=dataclasses.asdict(base.model);v.update(action_horizon=30,action_dim=32,decoder_init=str(a.cache/'decoder_init.npz'))
if a.variant=='fullft':v.update(paligemma_variant='gemma_2b',action_expert_variant='gemma_300m')
model=G1DC01PiConfig(**v).load(model_lib.restore_params(a.params));model.eval();data=G1DPiData(a.validation,a.cache,'validation',a.norm_stats)
# Runtime model receives only current images, normalized32D token state and language.
@jax.jit
def infer(obs,raw,key,noise):
 effect=model.sample_actions(key,obs,num_steps=a.denoise_steps,noise=noise)
 return model.decoder.physical(effect*model.effect_std.value+model.effect_mean.value,raw)
anchors=json.loads((ROOT/'data/validation_samples.json').read_text())
fixed=list(zip(anchors['indices'],anchors['task_indices']))
errors=[];preds=[];targets=[];tasks=[]
for i,(index,task) in enumerate(fixed):
 sample=data.sample(index);truth=sample['target_actions'];valid=sample['action_valid'];raw=sample['raw_state'][None]
 # Strip supervision before creating the inference Observation.
 obs=model_lib.Observation(**{k:jax.tree.map(lambda x:x[None],sample[k]) for k in ('images','image_masks','state','tokenized_prompt','tokenized_prompt_mask')})
 noise=np.random.default_rng(20260924+i).standard_normal((1,30,256),dtype=np.float32)
 pred=np.asarray(infer(obs,raw,jax.random.key(20260924+i),noise))[0]
 if pred.shape!=(30,16) or not np.isfinite(pred).all():raise ValueError('C01 decoder output violated16D contract')
 errors.append(np.where(valid[:,None],pred-truth,np.nan));preds.append(pred);targets.append(truth);tasks.append(task)
e=np.stack(errors);tasks=np.asarray(tasks)
def metrics(x):return {'mae':float(np.nanmean(np.abs(x))),'rmse':float(np.sqrt(np.nanmean(x*x))),'per_dim_mae':np.nanmean(np.abs(x),axis=(0,1)).tolist(),'horizon_mae':np.nanmean(np.abs(x),axis=(0,2)).tolist()}
a.output.mkdir(parents=True,exist_ok=True);report={'params':str(a.params),'variant':a.variant,'observations':len(e),'horizon':30,'native_state_dim':16,'action_dim':16,'denoise_steps':a.denoise_steps,'noise_protocol':'numpy.default_rng(20260924 + observation_index), standard_normal_float32; different latent dimensions','anchor_sha256':hashlib.sha256((ROOT/'data/validation_samples.json').read_bytes()).hexdigest(),'overall':metrics(e),'per_task':{str(t):metrics(e[tasks==t]) for t in sorted(set(tasks))},'limitation':'Selected holdout, not independent test set; compare baseline first30 of32 horizon with same anchors/noise protocol; not closed-loop success'}
(a.output/'action_metrics.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(a.output/'predictions.npz',predictions=np.stack(preds),targets=np.stack(targets),task_indices=tasks,indices=np.asarray(anchors['indices']),action_valid=np.isfinite(e).all(-1))
print(json.dumps(report,indent=2))
