from pathlib import Path
import shutil,json,os,signal,subprocess
r=Path('/HOME/sysu_xdliang/sysu_xdliang_2/HDD_POOL/yangky/g1d_subtask_compare40k_20261007')
status=json.loads((r/'QUEUE_STATUS.json').read_text());assert status.get('complete')==['lora','fullft'],status
shutil.copy2(r/'QUEUE_STATUS.json',r/'QUEUE_STATUS_40000.json')
for v in ['lora','fullft']:
 p=r/v/'code/train.py';s=p.read_text();assert 'decay_steps=args.steps' in s
 shutil.copy2(p,p.with_name('train_40000.py'))
 report=r/v/'logs/report'
 for name in ['resolved_run.json','best_checkpoint.json','TRAINING_COMPLETE']:
  shutil.copy2(report/name,report/(name+'.40000_snapshot'))
 (report/'TRAINING_COMPLETE').rename(report/'TRAINING_COMPLETE_40000')
 cp=r/v/'runs'/f'g1d_subtasks_pi05_compare_{v}_40k_v1'/'matched_subtasks_40k/40000'
 assert (cp/'train_state').is_dir() and (cp/'params').is_dir(),cp
 s=s.replace('decay_steps=args.steps','decay_steps=40000',1)
 s=s.replace("'training_mode':'fresh_matched_subtask_", "'training_mode':'resumed_matched_subtask_",1)
 s=s.replace("'prompt_mode':'subtask'", "'prompt_mode':'subtask', 'continuation_from':str(cfg.checkpoint_dir/'40000'), 'continuation_lr_policy':'original 40k schedule; plateau at 1e-6 to total100k'",1)
 s=s.replace(' if resuming:state=checkpoints.restore_state(manager,state,train_loader)', ' if resuming:state=checkpoints.restore_state(manager,state,train_loader)\n if args.resume:assert 40000<=int(state.step)<cfg.num_train_steps,int(state.step)',1)
 p.write_text(s)
s=(r/'run_queue.py').read_text().replace("'40000'","'100000'").replace("logs/train.log","logs/train_100k.log")
(r/'run_queue_100k.py').write_text(s)
(r/'EXTENSION_100K.json').write_text(json.dumps({'from_step':40000,'total_steps':100000,'additional_steps_each':60000,'restore':'params + optimizer train_state','lr_policy':'unchanged original40k schedule, terminal1e-6','validation_checkpoint_interval':2500,'variants':['lora','fullft']},indent=2))
# Restart only this comparison's metrics writer; the TensorBoard server reloads files.
pid=int((r/'tensorboard_writer.pid').read_text());proc=Path(f'/proc/{pid}/cmdline')
if proc.exists():
 assert str(r/'tensorboard_sync.py') in proc.read_bytes().decode();os.kill(pid,signal.SIGTERM)
with (r/'tensorboard_writer.log').open('a') as log:
 writer=subprocess.Popen([str(r/'lora/venv/bin/python'),str(r/'tensorboard_sync.py'),'--root',str(r)],stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True)
(r/'tensorboard_writer.pid').write_text(str(writer.pid))
with (r/'queue_100k.log').open('a') as log:
 q=subprocess.Popen(['/usr/bin/python3',str(r/'run_queue_100k.py')],cwd=r,stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True)
(r/'queue_100k.pid').write_text(str(q.pid))
print(json.dumps({'queue_pid':q.pid,'writer_pid':writer.pid,'target':100000}))
