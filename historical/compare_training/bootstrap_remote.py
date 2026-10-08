from pathlib import Path
import os, json, shutil, subprocess
base=Path('/HOME/sysu_xdliang/sysu_xdliang_2/HDD_POOL/yangky')
old=base/'g1d_pi05_lora_subtasks_20260930'
root=base/'g1d_subtask_compare40k_20261007'
root.mkdir(exist_ok=True)
source=(old/'code/train_subtasks_lora_40k.py').read_text()
for variant in ['lora','fullft']:
 r=root/variant
 if r.exists():raise RuntimeError('Refusing to overwrite existing comparison '+str(r))
 (r/'code').mkdir(parents=True);report=r/'logs/report';report.mkdir(parents=True)
 for directory in ['data','models','venv']:(r/directory).symlink_to(old/directory,target_is_directory=True)
 for package in ['openpi','psi']:
  if (old/'code'/package).exists():(r/'code'/package).symlink_to(old/'code'/package,target_is_directory=True)
 shutil.copy2(old/'code/openpi_train_official_215abfb.py',r/'code/openpi_train_official_215abfb.py')
 shutil.copy2(old/'logs/report/subtasks_lora_40k_20260930/split_manifest.json',report/'split_manifest.json')
 matches=[p for p in (old/'assets').rglob('norm_stats.json') if __import__('hashlib').sha256(p.read_bytes()).hexdigest()=='5801d57c3c40374d7882ef4edbe2ebfa0e3d9a103ae18cb1fd679e79deb770b9']
 assert matches
 asset=r/'assets'/f'g1d_subtasks_pi05_compare_{variant}_40k_v1'/'g1d_subtasks_train_v1'
 asset.mkdir(parents=True);shutil.copy2(matches[0],asset/'norm_stats.json')
 s=source
 s=s.replace("ROOT=Path(os.environ.get('G1D_LORA_ROOT',", "ROOT=Path(os.environ.get('G1D_LORA_ROOT',",1)
 # Explicit environment supplies new roots; no source/dataset changes.
 s=s.replace(' model=base.model', ' model=base.model' if variant=='lora' else " model=dataclasses.replace(base.model,paligemma_variant='gemma_2b',action_expert_variant='gemma_300m')",1)
 s=s.replace(' opt=base.optimizer', ' opt=base.optimizer\n low_mem_opt=LowMemoryAdamW(b1=opt.b1,b2=opt.b2,eps=opt.eps,weight_decay=opt.weight_decay,clip_gradient_norm=opt.clip_gradient_norm)',1)
 s=s.replace("name='g1d_subtasks_pi05_official_lora_v1'",f"name='g1d_subtasks_pi05_compare_{variant}_40k_v1'",1)
 s=s.replace('model=model, fsdp_devices=1,', 'model=model, fsdp_devices=1, freeze_filter=model.get_freeze_filter(), ema_decay=None,' if variant=='lora' else 'model=model, fsdp_devices=1, freeze_filter=nnx.Nothing, ema_decay=None,',1)
 s=s.replace('optimizer=opt, lr_schedule=lr_schedule','optimizer=low_mem_opt, lr_schedule=lr_schedule',1).replace('batch_size=8, resume=args.resume','batch_size=1, resume=args.resume',1)
 if variant=='fullft':s=s.replace("cfg.model.paligemma_variant=='gemma_2b_lora' and cfg.model.action_expert_variant=='gemma_300m_lora'", "cfg.model.paligemma_variant=='gemma_2b' and cfg.model.action_expert_variant=='gemma_300m'",1)
 s=s.replace("'memory_adaptation':'A800 single-GPU official LoRA; standard AdamW; frozen base parameters'",f"'comparison_variant':'{variant}', 'memory_adaptation':'matched AdamW mu_dtype=bfloat16, batch=1, EMA=None'",1)
 s=s.replace("'adaptation':'official 7-subtask LoRA; gemma_2b_lora + gemma_300m_lora; official freeze filter'",f"'adaptation':'matched seven-subtask {variant}; fresh pi05 base'",1)
 s=s.replace("'training_mode':'fresh_official_7subtask_lora_40k_from_pi05_base'",f"'training_mode':'fresh_matched_subtask_{variant}_40k_from_pi05_base', 'prompt_mode':'subtask'",1)
 s=s.replace("'validation_interval':500","'validation_interval':2500",1).replace('keep_period=500','keep_period=2500',1).replace('if step%500==0 or step==cfg.num_train_steps:','if step%2500==0 or step==cfg.num_train_steps:',1)
 (r/'code/train.py').write_text(s)
runner='''from pathlib import Path
import subprocess,os,json,time
root=Path(__file__).parent
status=root/'QUEUE_STATUS.json'
for variant in ['lora','fullft']:
 r=root/variant
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',G1D_LORA_ROOT=str(r),G1D_LORA_REPORT=str(r/'logs/report'),G1D_LORA_EXP_NAME='matched_subtasks_40k',PYTHONPATH=str(r/'code')+':/HOME/sysu_xdliang/sysu_xdliang_2/HDD_POOL/yangky/g1d_pi05_lora_subtasks_20260930/venv/lib/python3.12/site-packages:/HOME/sysu_xdliang/sysu_xdliang_2/HDD_POOL/yangky/runtime/venv/lib/python3.12/site-packages',JAX_PLATFORMS='cuda',PSI_HOME=str(r/'code'),DATA_HOME=str(r/'data'),HF_HOME=str(r/'cache/huggingface'),HF_DATASETS_CACHE=str(r/'cache/huggingface/datasets'),HF_HUB_CACHE=str(r/'cache/huggingface/hub'),WANDB_MODE='disabled',XLA_PYTHON_CLIENT_PREALLOCATE='false')
 with (r/'logs/train.log').open('w') as log:
  p=subprocess.Popen([str(r/'venv/bin/python'),str(r/'code/train.py'),'--steps','40000','--resume'],env=env,cwd=r,stdout=log,stderr=log)
  status.write_text(json.dumps({'active':variant,'pid':p.pid,'queued':['fullft'] if variant=='lora' else [],'started_unix':time.time()},indent=2))
  code=p.wait()
 if code:status.write_text(json.dumps({'failed':variant,'exit_code':code,'log':str(r/'logs/train.log')},indent=2));raise SystemExit(code)
status.write_text(json.dumps({'complete':['lora','fullft'],'finished_unix':time.time()},indent=2))
'''
(root/'run_queue.py').write_text(runner)
manifest={'data_split':'original seven-subtask grouped source-episode holdout','runs':['lora','fullft'],'fresh_initialization':True,'steps':40000,'batch_size':1,'seed':42,'ema_decay':None,'optimizer_momentum_dtype':'bfloat16','validation_seed':20260924,'validation_interval':2500,'checkpoint_interval':2500,'control_evaluation':'current aligned client fixed atomic prompt, identical per-task height and initial scene','historical_confounds':'old LoRA batch8/no EMA/standard AdamW; old full FT batch1/EMA .99/bfloat16 AdamW'}
(root/'EXPERIMENT.json').write_text(json.dumps(manifest,indent=2))
with (root/'queue.log').open('w') as log:
 p=subprocess.Popen(['/usr/bin/python3',str(root/'run_queue.py')],cwd=root,stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True)
(root/'queue.pid').write_text(str(p.pid))
print(json.dumps({'root':str(root),'queue_pid':p.pid,'status':str(root/'QUEUE_STATUS.json')}))
