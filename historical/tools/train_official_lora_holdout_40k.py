#!/usr/bin/env python3
"""Official OpenPI JAX training functions plus grouped-heldout evaluation/reporting."""
from dotenv import load_dotenv
load_dotenv('/mnt/data/yangky/test/Psi0/.env')
import argparse,dataclasses,functools,hashlib,importlib.util,json,logging,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from openpi.training import config as configs, data_loader, checkpoints, sharding
from openpi.models import model as model_api

REPORT=Path('/mnt/data/yangky/test/xr_teleoperate_g1d/reports/pi05_official_holdout_40k')
CONFIG='g1d_subtasks_pi05_official_lora_holdout_v1'

def load_official():
 path=Path(__file__).with_name('openpi_train_official_215abfb.py')
 spec=importlib.util.spec_from_file_location('official_openpi_training',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
 logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
 base_cfg=configs.get_config(CONFIG)
 cfg=dataclasses.replace(base_cfg, resume=args.resume, num_train_steps=40000, checkpoint_base_dir='/mnt/data/yangky/test/Psi0/.runs/openpi-05-jax-40k', lr_schedule=dataclasses.replace(base_cfg.lr_schedule, decay_steps=40000))
 split=json.loads((REPORT/'split_manifest.json').read_text());assert set(split['train_source_episodes']).isdisjoint(split['validation_source_episodes'])
 assert Path(cfg.data.repo_id)==Path(split['datasets']['train']['path'])
 assert cfg.model.pi05 and cfg.model.paligemma_variant.endswith('_lora') and cfg.model.action_expert_variant.endswith('_lora')
 assert cfg.ema_decay is None and cfg.fsdp_devices==2 and not cfg.data.base_config.legacy_quantile_norm
 assert jax.device_count()==2, jax.devices()
 stats_path=cfg.assets_dirs / cfg.data.assets.asset_id / 'norm_stats.json'
 assert hashlib.sha256(stats_path.read_bytes()).hexdigest()==split['stats']['sha256']
 REPORT.mkdir(parents=True,exist_ok=True)
 run={'config':cfg.name,'backend':'jax','adaptation':'official gemma_2b_lora + gemma_300m_lora; official get_freeze_filter',
      'official_reference':'215abfb217dbac7d5f1273282331b9b1866c0479','train_dataset':cfg.data.repo_id,
      'validation_dataset':split['datasets']['validation']['path'],'split_manifest_sha256':hashlib.sha256((REPORT/'split_manifest.json').read_bytes()).hexdigest(),
      'stats_sha256':split['stats']['sha256'],'batch_size':cfg.batch_size,'num_train_steps':cfg.num_train_steps,'continuation_from':'/mnt/data/yangky/test/Psi0/.runs/openpi-05-jax-40k/g1d_subtasks_pi05_official_lora_holdout_v1/subtasks_official_lora_holdout_v1/10000',
      'fsdp_devices':cfg.fsdp_devices,'seed':cfg.seed,'validation_seed':20260924,'optimizer':dataclasses.asdict(cfg.optimizer),'model':dataclasses.asdict(cfg.model),'lr_schedule':dataclasses.asdict(cfg.lr_schedule),
      'checkpoint_dir':str(cfg.checkpoint_dir),'devices':[str(d) for d in jax.devices()], 'validation_interval':500}
 run['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('openpi_train_official_215abfb.py'),Path(configs.__file__),Path(data_loader.__file__),Path('/mnt/data/yangky/test/Psi0/src/openpi/transforms.py')]}
 (REPORT/'resolved_run.json').write_text(json.dumps(run,indent=2))
 cfg.checkpoint_dir.mkdir(parents=True,exist_ok=True) if args.resume else None
 official=load_official();official.init_wandb(cfg,resuming=args.resume,enabled=False)
 jax.config.update('jax_compilation_cache_dir','/home/sys01/.cache/jax_g1d_official')
 mesh=sharding.make_mesh(cfg.fsdp_devices)
 ds_sharding=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS))
 repl=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
 manager,resuming=checkpoints.initialize_checkpoint_dir(cfg.checkpoint_dir,keep_period=500,overwrite=False,resume=args.resume)
 train_loader=data_loader.create_data_loader(cfg,sharding=ds_sharding,shuffle=True)
 train_iter=iter(train_loader);batch=next(train_iter)
 logging.info('Training batch loaded: actions=%s, train source episodes=%d, validation source episodes=%d',batch[1].shape,len(split['train_source_episodes']),len(split['validation_source_episodes']))
 # Validation uses only held-out source episodes, with TRAINING norm stats and fixed frames.
 val_cfg=dataclasses.replace(cfg.data,repo_id=split['datasets']['validation']['path'])
 val_dc=val_cfg.create(cfg.assets_dirs,cfg.model)
 val_ds=data_loader.create_torch_dataset(val_dc,cfg.model.action_horizon,cfg.model)
 val_ds=data_loader.transform_dataset(val_ds,val_dc)
 rows=[json.loads(line) for line in (Path(val_cfg.repo_id)/'meta/episodes.jsonl').read_text().splitlines()]
 indices=[];tasks=[]
 for r in rows:
  n=r['length'];usable=max(0,n-cfg.model.action_horizon)
  for frac in (.25,.75):
   indices.append(r['dataset_from_index']+int(usable*frac));tasks.append(r['task_index'])
 (REPORT/'validation_samples.json').write_text(json.dumps({'indices':indices,'task_indices':tasks,'frames_per_episode':2,'note':'Fixed heldout observations; no training or stats fit'},indent=2))
 val_batches=[]
 for start in range(0,len(indices),cfg.batch_size):
  chosen=indices[start:start+cfg.batch_size];valid=len(chosen);chosen+=chosen[-1:]*(cfg.batch_size-valid)
  samples=[val_ds[i] for i in chosen]
  d=jax.tree.map(lambda *x:np.stack(x),*samples)
  obs=model_api.Observation.from_dict(d);actions=d['actions']
  val_batches.append((jax.tree.map(np.asarray,obs),actions,valid))
 logging.info('Validation prepared: %d fixed held-out observations, %d batches',len(indices),len(val_batches))
 rng=jax.random.key(cfg.seed);train_rng,init_rng=jax.random.split(rng)
 state,state_sharding=official.init_train_state(cfg,init_rng,mesh,resume=resuming)
 jax.block_until_ready(state)
 if resuming:state=checkpoints.restore_state(manager,state,train_loader)
 trainable=state.params.filter(cfg.trainable_filter)
 trainable_count=sum(x.size for x in jax.tree.leaves(trainable));total_count=sum(x.size for x in jax.tree.leaves(state.params))
 run.update(trainable_parameters=trainable_count,total_parameters=total_count);(REPORT/'resolved_run.json').write_text(json.dumps(run,indent=2))
 logging.info('Initialized official LoRA: %d trainable / %d total parameters',trainable_count,total_count)
 step_fn=jax.jit(functools.partial(official.train_step,cfg),in_shardings=(repl,state_sharding,ds_sharding),out_shardings=(state_sharding,repl),donate_argnums=(1,))
 def eval_loss(st,key,b):
  model=nnx.merge(st.model_def,st.params);model.eval()
  loss=model.compute_loss(key,b[0],b[1],train=False)
  return jnp.mean(loss,axis=tuple(range(1,loss.ndim)))
 eval_fn=jax.jit(eval_loss,in_shardings=(state_sharding,repl,ds_sharding),out_shardings=repl)
 def validate(step):
  values=[]
  for i,(obs,actions,valid) in enumerate(val_batches):
   b=jax.device_put((obs,actions),ds_sharding)
   with sharding.set_mesh(mesh):v=eval_fn(state,jax.random.fold_in(jax.random.key(20260924),i),b)
   values.extend(np.asarray(v)[:valid].tolist())
  metrics={'step':step,'validation_loss':float(np.mean(values)),'observations':len(values),
           'per_task_loss':{str(t):float(np.mean([v for v,k in zip(values,tasks) if k==t])) for t in sorted(set(tasks))}}
  assert np.isfinite(metrics['validation_loss'])
  with (REPORT/'validation_metrics.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
  logging.info('VALIDATION step=%d loss=%.6f per_task=%s',step,metrics['validation_loss'],metrics['per_task_loss'])
  return metrics
 start_step=int(state.step)
 validate(start_step)
 best_path=REPORT/'best_checkpoint.json';best=float('inf')
 if args.resume and best_path.exists():best=json.loads(best_path.read_text())['validation_loss']
 training_log=REPORT/'training_metrics.jsonl'
 for step in range(start_step+1,cfg.num_train_steps+1):
  begun=time.perf_counter()
  with sharding.set_mesh(mesh):state,info=step_fn(train_rng,state,batch)
  info=jax.device_get(info)
  assert np.isfinite(info['loss']) and np.isfinite(info['grad_norm']),info
  elapsed=time.perf_counter()-begun
  if step<=5 or step%10==0:
   metrics={'step':step,'loss':float(info['loss']),'grad_norm':float(info['grad_norm']),'seconds':elapsed}
   with training_log.open('a') as f:f.write(json.dumps(metrics)+'\n')
   logging.info('TRAIN step=%d/%d loss=%.6f grad_norm=%.4f seconds=%.2f',step,cfg.num_train_steps,metrics['loss'],metrics['grad_norm'],elapsed)
  if step%500==0 or step==cfg.num_train_steps:
   metrics=validate(step)
   checkpoints.save_state(manager,state,train_loader,step);manager.wait_until_finished()
   checkpoint=cfg.checkpoint_dir/str(step)
   (checkpoint/'g1d_contract.json').write_text(json.dumps({**run,'step':step,'normalization':'official_quantile_unclipped','validation':metrics},indent=2))
   if metrics['validation_loss']<best:
    best=metrics['validation_loss'];best_path.write_text(json.dumps({**metrics,'checkpoint':str(checkpoint)},indent=2))
  if step<cfg.num_train_steps:batch=next(train_iter)
 manager.wait_until_finished()
 (REPORT/'TRAINING_COMPLETE').write_text(json.dumps({'steps':cfg.num_train_steps,'best':json.loads(best_path.read_text())},indent=2))
 logging.info('Training complete; best checkpoint: %s',best_path.read_text())

if __name__=='__main__':main()
