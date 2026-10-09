"""Export transfer status as scalars; these are not model training metrics."""
import argparse
import json
from pathlib import Path
import time
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--logdir',type=Path,required=True);a=p.parse_args()
w=EventFileWriter(str(a.logdir/'asset_transfer'))
try:
 while True:
  try:
   s=json.loads((a.root/'ASSET_SYNC_STATUS.json').read_text())
   r=json.loads((a.root/'AUTO_SYNC_RETRY_STATUS.json').read_text())
   now=time.time();total=max(s['total_bytes'],1)
   m={'verified_GB':s['bytes']/1e9,'inflight_GB':s.get('inflight_bytes',0)/1e9,
      'verified_percent':100*s['bytes']/total,
      'written_percent_unverified_included':100*s.get('written_bytes_including_unverified',s['bytes'])/total,
      'verified_files':s['files'],'retry_attempt':r.get('attempt',0),
      'active_files':len(s.get('active_files',[])),
      'status_age_seconds':max(0,now-s['updated_unix']),
      'controller_age_seconds':max(0,now-r['updated_unix']),
      'complete':int(s['state']=='complete'),'last_status_failed':int(s['state']=='failed')}
   w.add_event(Event(wall_time=now,step=int(now),summary=Summary(value=[Summary.Value(tag='transfer/'+k,simple_value=v) for k,v in m.items()])))
   w.flush()
  except (OSError,ValueError,KeyError) as e:
   print(type(e).__name__,str(e),flush=True)
  time.sleep(5)
finally:w.close()
