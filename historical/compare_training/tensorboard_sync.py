"""Continuously mirror persisted comparison metrics; no GPU runtime imports."""
from pathlib import Path
import argparse, json, math, time, fcntl
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args();root=args.root
lock=(root/'tensorboard_sync.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
specs={'lora':('lora',40000,0,40000),'fullft':('fullft',40000,0,40000),'lora_to100k':('lora',100000,40000,100000),'fullft_to100k':('fullft',100000,40000,100000)}
writers={};seen={}
for variant in specs:
 directory=root/'tensorboard'/variant;directory.mkdir(parents=True,exist_ok=True)
 acc=EventAccumulator(str(directory),size_guidance={'scalars':0});acc.Reload()
 seen[variant]={(tag,event.step) for tag in acc.Tags().get('scalars',[]) for event in acc.Scalars(tag)}
 writers[variant]=EventFileWriter(str(directory))
def records(path):
 if not path.exists():return
 for line in path.read_text().splitlines():
  try:yield json.loads(line)
  except json.JSONDecodeError:continue

def emit(variant,tag,value,step):
 if (tag,step) in seen[variant] or not math.isfinite(value):return
 writers[variant].add_event(Event(wall_time=time.time(),step=step,summary=Summary(value=[Summary.Value(tag=tag,simple_value=value)])))
 seen[variant].add((tag,step))
try:
 while True:
  status={'updated_unix':time.time(),'logdir':str(root/'tensorboard'),'runs':{}}
  for variant in writers:
   source,total,minimum,maximum=specs[variant]
   report=root/source/'logs/report';latest=minimum;latest_val=None
   emit(variant,'progress/completed_percent',minimum/total*100.0,minimum)
   for record in records(report/'training_metrics.jsonl'):
    step=int(record['step'])
    if not minimum<=step<=maximum:continue
    latest=max(latest,step)
    for key in ('loss','grad_norm','seconds'):
     if key in record:emit(variant,'train/'+key,float(record[key]),step)
    emit(variant,'progress/completed_percent',step/total*100.0,step)
   for record in records(report/'validation_metrics.jsonl'):
    step=int(record['step'])
    if not minimum<=step<=maximum:continue
    latest_val=step if latest_val is None else max(step,latest_val)
    emit(variant,'validation/loss',float(record['validation_loss']),step)
    for task,value in record.get('per_task_loss',{}).items():emit(variant,'validation/subtask_'+str(task)+'/loss',float(value),step)
   writers[variant].flush()
   status['runs'][variant]={'latest_train_step':latest,'latest_validation_step':latest_val,'complete':latest>=total}
  queue=root/'QUEUE_STATUS.json'
  if queue.exists():
   try:status['queue']=json.loads(queue.read_text())
   except json.JSONDecodeError:pass
  tmp=root/'tensorboard_status.tmp';tmp.write_text(json.dumps(status,indent=2));tmp.replace(root/'tensorboard_status.json')
  time.sleep(5)
finally:
 for writer in writers.values():writer.close()
