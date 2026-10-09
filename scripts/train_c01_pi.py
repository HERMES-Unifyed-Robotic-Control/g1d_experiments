#!/usr/bin/env python3
"""G1D C01 E flow + joint decoder; optional multi-process data parallelism."""
from pathlib import Path
import os,sys,argparse,dataclasses,functools,importlib.util,json,hashlib,time
ROOT=Path(__file__).parents[1];os.environ.setdefault('PSI_HOME',str(ROOT/'vendor/openpi'));os.environ.setdefault('DATA_HOME',str(ROOT/'artifacts/data'));os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
sys.path[:0]=[str(ROOT),str(ROOT/'vendor/openpi/src'),str(ROOT/'vendor/openpi/src/openpi/openpi-client/src')]
import numpy as np
import jax
from g1d_experiments import c01_distributed as distributed
distributed_parser=distributed.arguments();distributed.initialize(distributed_parser)
import jax.numpy as jnp
from flax import nnx
import optax
from openpi.training import config,checkpoints,sharding
from openpi.shared import normalize
from g1d_experiments.c01_pi import G1DC01PiConfig,G1DC01Weights
from g1d_experiments.c01_pi_data import G1DPiData
from train_subtask import LowMemoryAdamW
p=argparse.ArgumentParser(parents=[distributed_parser]);p.add_argument('--cache',type=Path,required=True);p.add_argument('--train',type=Path,required=True);p.add_argument('--validation',type=Path,required=True);p.add_argument('--base-params',type=Path,required=True);p.add_argument('--norm-stats',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--variant',choices=['lora','fullft'],default='lora');p.add_argument('--steps',type=int,default=40000);p.add_argument('--batch-size',type=int,default=1);p.add_argument('--gtaux-weight',type=float,default=None,help='G1D calibrated weight, or explicit exploratory choice');p.add_argument('--delta-weight',type=float,default=.05);p.add_argument('--calibrate-only',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
if jax.local_device_count()!=1:raise RuntimeError('Expose one GPU per process')
if a.batch_size%jax.device_count():raise ValueError('Global batch size must be divisible by total GPU count')
if a.calibrate_only and jax.process_count()>1:raise ValueError('Reuse single-GPU GT-AUX calibration for the distributed comparison')
if a.gtaux_weight is None:
 if not a.calibrate_only:raise ValueError('Run --calibrate-only first and provide its chosen_gtaux_weight')
 a.gtaux_weight=0.
if min(a.gtaux_weight,a.delta_weight)<0:raise ValueError('Auxiliary weights must be nonnegative')
a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True);manifest=json.loads((a.cache/'manifest.json').read_text())
if not manifest.get('complete'):raise ValueError('Export E cache first')
for name,expected in [('decoder_init.npz','decoder_sha256'),('normalization.npz','normalization_sha256')]:
 if hashlib.sha256((a.cache/name).read_bytes()).hexdigest()!=manifest[expected]:raise ValueError('E cache asset changed:'+name)
contract={'model':'g1d_c01_native16_joint_response_v1','variant':a.variant,'decoder_sha256':manifest['decoder_sha256'],'codec_sha256':manifest['checkpoint_sha256'],'E_norm_sha256':manifest['normalization_sha256'],'state_norm_sha256':hashlib.sha256(a.norm_stats.read_bytes()).hexdigest(),'train':str(a.train.resolve()),'validation':str(a.validation.resolve()),'state_dim':16,'padded_state_dim':32,'E_shape':[30,256],'output_action_dim':16,'gtaux_weight':a.gtaux_weight,'delta_weight':a.delta_weight,'decoder_lr_scale':.25,'seed':42,'batch_size':a.batch_size}
if jax.process_count()>1:contract.update(process_count=jax.process_count(),global_device_count=jax.device_count(),parallelism='data_parallel',samples_per_step=a.batch_size)
if (a.output/'contract.json').exists() and json.loads((a.output/'contract.json').read_text())!=contract:raise ValueError('Resume contract changed')
distributed.write_json(a.output/'contract.json',contract);distributed.synchronize('c01_contract')
train=G1DPiData(a.train,a.cache,'train',a.norm_stats);val=G1DPiData(a.validation,a.cache,'validation',a.norm_stats)
if {e['source_episode_index'] for e in train.episodes}&{e['source_episode_index'] for e in val.episodes}:raise ValueError('Source episode leakage')
base=config.get_config('g1d_subtasks_pi05_official_lora_holdout_v1');values=dataclasses.asdict(base.model);values.update(action_horizon=30,action_dim=32,decoder_init=str((a.cache/'decoder_init.npz').resolve()),gtaux_weight=a.gtaux_weight,delta_weight=a.delta_weight)
if a.variant=='fullft':values.update(paligemma_variant='gemma_2b',action_expert_variant='gemma_300m')
model_config=G1DC01PiConfig(**values);cfg=dataclasses.replace(base,model=model_config,freeze_filter=model_config.get_freeze_filter() if a.variant=='lora' else nnx.Nothing,weight_loader=G1DC01Weights(str(a.base_params)),optimizer=LowMemoryAdamW(**dataclasses.asdict(base.optimizer)),num_train_steps=a.steps,batch_size=a.batch_size,fsdp_devices=1,ema_decay=None,lr_schedule=dataclasses.replace(base.lr_schedule,decay_steps=40000),resume=a.resume)
spec=importlib.util.spec_from_file_location('official',ROOT/'historical/tools/openpi_train_official_215abfb.py');official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
mesh=sharding.make_mesh(1);ds=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS));repl=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
manager,resuming=distributed.checkpoint_manager(checkpoints,a.output/'checkpoints',a.resume)
if jax.process_count()>1:cfg=dataclasses.replace(cfg,weight_loader=distributed.ReplicatedWeights(cfg.weight_loader,repl))
state,state_sharding=official.init_train_state(cfg,distributed.key(42,repl) if jax.process_count()>1 else jax.random.key(42),mesh,resume=resuming);jax.block_until_ready(state)
class Assets:
 def data_config(self):return dataclasses.replace(cfg.data.create(cfg.assets_dirs,cfg.model),norm_stats=normalize.load(a.norm_stats.parent),asset_id='g1d_subtasks_train_v1')
assets=Assets();rng=np.random.default_rng(42)
if resuming:
 state=checkpoints.restore_state(manager,state,assets);cursor=json.loads((a.output/f'cursor_{int(state.step)}.json').read_text());rng.bit_generator.state=cursor['sample_rng']
fixed=[];off=0
for e in val.episodes:
 for frac in (.25,.75):fixed.append(off+int(max(0,e['length']-30)*frac))
 off+=e['length']
distributed.write_json(a.output/'validation_indices.json',fixed)
def step_fn(state,batch):
 m=nnx.merge(state.model_def,state.params);m.train();obs,_=batch
 def objective(m):return m.losses(jax.random.fold_in(jax.random.key(42),state.step),obs,state.step,train=True)
 (loss,metrics),grads=nnx.value_and_grad(objective,argnums=nnx.DiffState(0,cfg.trainable_filter),has_aux=True)(m)
 params=state.params.filter(cfg.trainable_filter);updates,opt=state.tx.update(grads,state.opt_state,params)
 def scale(path,value):return value*.25 if 'decoder' in str(path) else value
 updates=jax.tree_util.tree_map_with_path(scale,updates);nnx.update(m,optax.apply_updates(params,updates));return dataclasses.replace(state,step=state.step+1,params=nnx.state(m),opt_state=opt),dict(metrics,grad_norm=optax.global_norm(grads))
learning_rate=cfg.lr_schedule.create()
step_jit=jax.jit(step_fn,in_shardings=(state_sharding,ds),out_shardings=(state_sharding,repl),donate_argnums=(0,))
if a.calibrate_only:
 def measure(st,batch,key):
  m=nnx.merge(st.model_def,st.params);m.eval();return m.losses(key,batch[0],jnp.asarray(10000),train=False,calibration=True)[1]
 measure=jax.jit(measure);rows=[];diagnostic=np.random.default_rng(20261008)
 for i in range(16):
  batch=jax.device_put(train.batch(diagnostic.integers(0,len(train),a.batch_size)),ds)
  v=measure(state,batch,jax.random.fold_in(jax.random.key(20261008),i));rows.append({k:float(x) for k,x in jax.device_get(v).items()})
 ratio=float(np.median([r['gtaux_unweighted_output_gradient_ratio'] for r in rows]));weight=.1 if .05<=.1*ratio<=.2 else .1/max(ratio,1e-12)
 if not np.isfinite(weight) or not 0<weight<=10:raise ValueError('Implausible GT-AUX calibration; inspect before training')
 (a.output/'calibration.json').write_text(json.dumps({'chosen_gtaux_weight':weight,'median_unweighted_output_gradient_ratio':ratio,'optimizer_updates':0,'mode':'train-only,current-policy,output-gradient-calibration','rows':rows},indent=2)+'\n');manager.close();print('Calibration complete, gtaux_weight=',weight);sys.exit(0)

def evaluate(st,batch,key):
 m=nnx.merge(st.model_def,st.params);m.eval();return m.losses(key,batch[0],st.step,train=False)[1]
eval_jit=jax.jit(evaluate)
while int(state.step)<a.steps:
 batch=distributed.batch(train,rng.integers(0,len(train),a.batch_size),ds);state,metrics=step_jit(state,batch);step=int(state.step)
 if step%10==0:
  record={'step':step,'lr':float(learning_rate(step-1)),**{k:float(v) for k,v in jax.device_get(metrics).items()}}
  if not all(np.isfinite(v) for v in record.values()):raise FloatingPointError('Nonfinite pi metrics')
  distributed.append_json(a.output/'training_metrics.jsonl',record)
  if jax.process_index()==0:print(json.dumps(record),flush=True)
 if step%2500==0 or step==a.steps:
  values=[]
  for start in range(0,len(fixed),a.batch_size):
   ids=fixed[start:start+a.batch_size];n=len(ids);ids=ids+[ids[-1]]*(a.batch_size-n)
   v=eval_jit(state,distributed.batch(val,ids,ds),jax.random.fold_in(distributed.key(20260924,repl) if jax.process_count()>1 else jax.random.key(20260924),start));values.append((n,{k:float(x) for k,x in jax.device_get(v).items()}))
  record={'step':step,'observations':len(fixed),**{k:sum(n*v[k] for n,v in values)/len(fixed) for k in values[0][1]}}
  distributed.append_json(a.output/'validation_metrics.jsonl',record)
  checkpoints.save_state(manager,state,assets,step);manager.wait_until_finished();distributed.write_json(a.output/f'cursor_{step}.json',{'sample_rng':rng.bit_generator.state});distributed.synchronize('c01_cursor_'+str(step))
manager.close();distributed.write_json(a.output/'TRAINING_COMPLETE.json',{'steps':int(state.step)});distributed.synchronize('c01_complete')
