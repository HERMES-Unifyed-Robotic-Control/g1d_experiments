#!/usr/bin/env python3
"""Matched current-observation action generation; no supervision passed to models."""
from pathlib import Path
import os,sys,argparse,dataclasses,json,hashlib,time
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('PSI_HOME',str(ROOT/'vendor/openpi'));os.environ.setdefault('DATA_HOME',str(ROOT/'artifacts/data'));os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
sys.path[:0]=[str(ROOT),str(ROOT/'vendor/openpi/src'),str(ROOT/'vendor/openpi/src/openpi/openpi-client/src')]
import numpy as np
import jax
import jax.numpy as jnp
from openpi.training import config
from openpi.models import model as model_lib
from openpi.shared import nnx_utils,normalize
from g1d_experiments.c01_pi import G1DC01PiConfig
from g1d_experiments.c01_pi_data import G1DPiData
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--params',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--validation',type=Path,required=True);p.add_argument('--norm-stats',type=Path,required=True);p.add_argument('--variant',choices=['baseline','lora','fullft'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=200);p.add_argument('--denoise-steps',type=int,default=10);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
base=config.get_config('g1d_subtasks_pi05_official_lora_holdout_v1')
if a.variant=='baseline':
 cfg=base.model;params=model_lib.restore_params(a.params,dtype=jnp.bfloat16)
else:
 v=dataclasses.asdict(base.model);v.update(action_horizon=30,action_dim=32,decoder_init=str(a.cache/'decoder_init.npz'))
 if a.variant=='fullft':v.update(paligemma_variant='gemma_2b',action_expert_variant='gemma_300m')
 cfg=G1DC01PiConfig(**v);params=model_lib.restore_params(a.params)
leaves=jax.tree.leaves(params);print('PARAM_BYTES',sum(x.size*x.dtype.itemsize for x in leaves),flush=True)
model=cfg.load(params);model.eval();sample_fn=nnx_utils.module_jit(model.sample_actions)
if a.variant!='baseline':decode_fn=nnx_utils.module_jit(model.decoder.physical)
data=G1DPiData(a.validation,a.cache,'validation',a.norm_stats);action_stats=normalize.load(a.norm_stats.parent)['actions']
anchors_path=ROOT/'data/validation_samples.json';anchors=json.loads(anchors_path.read_text());indices=anchors['indices'][:a.limit];tasks=anchors['task_indices'][:a.limit]
preds=[];targets=[];valids=[];states=[];future_states=[];digests=[];source_ids=[];episode_ids=[];frames=[];prompts=[];latencies=[];seen=set()
for i,(index,task) in enumerate(zip(indices,tasks)):
 s=data.sample(index);ei=int(np.searchsorted(data.ends,index,side='right'));frame=index-(int(data.ends[ei-1]) if ei else 0);ep=data.episodes[ei];assert ep['task_index']==task
 inputs={k:s[k] for k in ('images','image_masks','state','tokenized_prompt','tokenized_prompt_mask')}
 h=hashlib.sha256()
 for k,x in sorted(jax.tree_util.tree_flatten_with_path(inputs)[0],key=lambda v:str(v[0])):
  x=np.ascontiguousarray(x);h.update(str(k).encode());h.update(str(x.dtype).encode());h.update(str(x.shape).encode());h.update(x.tobytes())
 digests.append(h.hexdigest())
 obs=model_lib.Observation(**jax.tree.map(lambda x:x[None],inputs));raw=s['raw_state'][None]
 shape=(1,32,32) if a.variant=='baseline' else (1,30,256)
 noise=np.random.default_rng(20260924+i).standard_normal(shape,dtype=np.float32)
 start=time.monotonic();value=sample_fn(jax.random.key(20260924+i),obs,num_steps=a.denoise_steps,noise=noise)
 if a.variant=='baseline':
  z=np.asarray(value)[0,:30,:16];pred=(z+1)/2*(action_stats.q99-action_stats.q01+1e-6)+action_stats.q01
 else:pred=np.asarray(decode_fn(value*model.effect_std.value+model.effect_mean.value,raw))[0]
 latencies.append(time.monotonic()-start)
 assert pred.shape==(30,16) and np.isfinite(pred).all()
 preds.append(pred);targets.append(s['target_actions']);valids.append(s['action_valid']);states.append(s['raw_state']);future_states.append(ep['states'][np.minimum(frame+np.arange(30),ep['length']-1)]);source_ids.append(ep['source_episode_index']);episode_ids.append(ep['episode_index']);frames.append(frame);prompts.append(data.tasks[task])
 if task not in seen:
  seen.add(task)
  for view,im in s['images'].items():Image.fromarray(np.clip((im+1)*127.5,0,255).astype(np.uint8)).save(a.output/f'obs_task{task}_anchor{i}_{view}.png')
 if (i+1)%10==0 or i+1==len(indices):print(json.dumps({'evaluated':i+1,'total':len(indices),'latest_infer_seconds':latencies[-1]}),flush=True)
np.savez_compressed(a.output/'predictions.npz',predictions=np.stack(preds),targets=np.stack(targets),task_indices=np.asarray(tasks),indices=np.asarray(indices),action_valid=np.stack(valids),raw_states=np.stack(states),future_states=np.stack(future_states),source_episode_indices=np.asarray(source_ids),subtask_episode_indices=np.asarray(episode_ids),frame_indices=np.asarray(frames),input_sha256=np.asarray(digests),prompts=np.asarray(prompts))
e=np.where(np.stack(valids)[...,None],np.stack(preds)-np.stack(targets),np.nan)
report={'params':str(a.params),'variant':a.variant,'observations':len(preds),'horizon':30,'native_model_horizon':32 if a.variant=='baseline' else 30,'denoise_steps':a.denoise_steps,'anchor_sha256':hashlib.sha256(anchors_path.read_bytes()).hexdigest(),'state_stats_sha256':hashlib.sha256(a.norm_stats.read_bytes()).hexdigest(),'parameter_bytes':sum(x.size*x.dtype.itemsize for x in leaves),'parameter_precision':'all bfloat16 (official baseline policy)' if a.variant=='baseline' else 'checkpoint-native (same as existing C01 evaluation)','overall':{'mae':float(np.nanmean(np.abs(e))),'rmse':float(np.sqrt(np.nanmean(e*e)))},'noise_protocol':'numpy.default_rng(20260924+anchor_ordinal), standard_normal float32; baseline32x32, C01 30x256','timing_note':'includes first JIT compile; concurrent GPU training for fullFT, not a hardware-fair latency benchmark','infer_seconds':latencies,'limitation':'selected holdout, not independent test; teacher current observations; no robot executed'}
(a.output/'action_metrics.json').write_text(json.dumps(report,indent=2)+'\n');(a.output/'EVALUATION_COMPLETE.json').write_text(json.dumps({'complete':True,'observations':len(preds),'finished_unix':time.time()}));print('EVALUATION_COMPLETE',report['overall'],flush=True)
