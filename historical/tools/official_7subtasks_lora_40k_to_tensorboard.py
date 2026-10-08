#!/usr/bin/env python3
"""Mirror persisted holdout metrics without importing the GPU training runtime."""
import json,math,time,fcntl
from pathlib import Path
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
REPORT=Path('/mnt/data/yangky/test/xr_teleoperate_g1d/reports/pi05_official_7subtasks_lora_40k_20260929')
LOGDIR=Path('/mnt/data/yangky/test/xr_teleoperate_g1d/reports/pi05_finetune/tensorboard/g1d_subtasks_official_lora_7subtasks_40k_20260929')
LOGDIR.mkdir(parents=True,exist_ok=True)
lock=(REPORT/'tensorboard_writer.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
seen=set()
acc=EventAccumulator(str(LOGDIR),size_guidance={'scalars':0});acc.Reload()
for tag in acc.Tags()['scalars']:
 for value in acc.Scalars(tag):seen.add((tag,value.step))
writer=EventFileWriter(str(LOGDIR));total=json.loads((REPORT/'resolved_run.json').read_text())['num_train_steps']
def emit(tag,value,step):
 key=(tag,step)
 if key in seen or not math.isfinite(value):return
 writer.add_event(Event(wall_time=time.time(),step=step,summary=Summary(value=[Summary.Value(tag=tag,simple_value=value)])))
 seen.add(key)
def records(name):
 p=REPORT/name
 if not p.exists():return
 for line in p.read_text().splitlines():
  try:yield json.loads(line)
  except json.JSONDecodeError:continue
try:
 while True:
  for r in records('training_metrics.jsonl'):
   step=r['step']
   for key in ('loss','grad_norm','seconds'):emit('train/'+key,float(r[key]),step)
   emit('progress/completed_percent',100*step/total,step)
  for r in records('validation_metrics.jsonl'):
   step=r['step'];emit('validation/loss',r['validation_loss'],step)
   for task,value in r['per_task_loss'].items():emit('validation/subtask_'+task+'/loss',value,step)
  writer.flush()
  status={'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'logdir':str(LOGDIR),'latest_train_step':max((step for tag,step in seen if tag=='train/loss'),default=0),'latest_validation_step':max((step for tag,step in seen if tag=='validation/loss'),default=0)}
  (REPORT/'tensorboard_status.json').write_text(json.dumps(status,indent=2))
  time.sleep(5)
finally:writer.close()
