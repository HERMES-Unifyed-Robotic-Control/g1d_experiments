from pathlib import Path
import json,shutil,hashlib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from openpi.shared import normalize
src=Path('/mnt/data/yangky/test/datasets/g1d_food_sorting_123_subtasks_lerobot')
base=src.parent
outdir=Path('/mnt/data/yangky/test/xr_teleoperate_g1d/reports/pi05_official_holdout')
rows=[json.loads(x) for x in (src/'meta/episodes.jsonl').read_text().splitlines()]
stats={x['episode_index']:x for x in map(json.loads,(src/'meta/episodes_stats.jsonl').read_text().splitlines())}
groups={}
for r in rows:
 sid=r['source_episode_index'];tid=r['source_task_index'];groups.setdefault(tid,set()).add(sid)
rng=np.random.default_rng(20260924);val=set();counts={}
for tid,ids in sorted(groups.items()):
 ids=sorted(ids);n=int(np.floor(.1*len(ids)+.5));chosen=set(map(int,rng.choice(ids,n,replace=False)));val.update(chosen);counts[tid]={'total':len(ids),'validation':n,'train':len(ids)-n}
all_ids={r['source_episode_index'] for r in rows};train=all_ids-val
assert train.isdisjoint(val) and len(train)+len(val)==197
manifest={'seed':20260924,'group_key':'source_episode_index','stratify':'source_task_index','source':str(src),'source_meta_sha256':hashlib.sha256((src/'meta/episodes.jsonl').read_bytes()).hexdigest(),'source_counts':counts,'train_source_episodes':sorted(train),'validation_source_episodes':sorted(val),'datasets':{}}
running={k:normalize.RunningStats() for k in ['state','actions']}
for split,ids in [('train',train),('validation',val)]:
 dst=base/f'g1d_subtasks_official_{split}_v1'
 assert not dst.exists(),f'Refusing overwrite: {dst}'
 (dst/'meta').mkdir(parents=True)
 info=json.loads((src/'meta/info.json').read_text());newrows=[];newstats=[];idx=0
 for old in rows:
  if old['source_episode_index'] not in ids:continue
  ep=len(newrows);old_ep=old['episode_index'];oldchunk=old_ep//1000;chunk=ep//1000
  table=pq.read_table(src/f'data/chunk-{oldchunk:03d}/episode_{old_ep:06d}.parquet');n=table.num_rows
  for name,values in [('episode_index',np.full(n,ep,dtype=np.int64)),('index',np.arange(idx,idx+n,dtype=np.int64))]:
   table=table.set_column(table.schema.get_field_index(name),name,pa.array(values))
  data=dst/f'data/chunk-{chunk:03d}/episode_{ep:06d}.parquet';data.parent.mkdir(parents=True,exist_ok=True);pq.write_table(table,data,compression='zstd')
  for view in ['egocentric','left_wrist','right_wrist']:
   target=src/f'videos/chunk-{oldchunk:03d}/observation.images.{view}/episode_{old_ep:06d}.mp4';assert target.exists()
   link=dst/f'videos/chunk-{chunk:03d}/observation.images.{view}/episode_{ep:06d}.mp4';link.parent.mkdir(parents=True,exist_ok=True);link.symlink_to(target)
  nr={**old,'episode_index':ep,'dataset_from_index':idx,'dataset_to_index':idx+n-1,'original_subtask_episode_index':old_ep}
  newrows.append(nr);newstats.append({**stats[old_ep],'episode_index':ep});idx+=n
  if split=='train':
   state=np.asarray(table['observation.state'].to_pylist(),dtype=np.float32);act=np.asarray(table['action'].to_pylist(),dtype=np.float32)
   running['state'].update(state)
   # Same 32-step, episode-end padded targets as the official LeRobot training loader.
   windows=act[np.minimum(np.arange(n)[:,None]+np.arange(32)[None,:],n-1)]
   running['actions'].update(windows)
 info.update(total_episodes=len(newrows),total_frames=idx,total_chunks=(len(newrows)+999)//1000,total_videos=len(newrows)*3,splits={'train':f'0:{len(newrows)}'})
 (dst/'meta/info.json').write_text(json.dumps(info,indent=2))
 for name,items in [('episodes.jsonl',newrows),('episodes_stats.jsonl',newstats),('subtask_segments.jsonl',newrows)]:
  (dst/'meta'/name).write_text(''.join(json.dumps(x)+'\n' for x in items))
 for name in ['tasks.jsonl','g1d_schema.json']:shutil.copy2(src/'meta'/name,dst/'meta'/name)
 manifest['datasets'][split]={'path':str(dst),'episodes':len(newrows),'frames':idx,'task_episode_counts':{str(k):sum(x['task_index']==k for x in newrows) for k in range(7)}}
 print(split,manifest['datasets'][split],flush=True)
asset=Path('/mnt/data/yangky/test/Psi0/assets/g1d_subtasks_pi05_official_lora_holdout_v1/g1d_subtasks_train_v1')
normalize.save(asset,{k:v.get_statistics() for k,v in running.items()})
manifest['stats']={'path':str(asset/'norm_stats.json'),'sha256':hashlib.sha256((asset/'norm_stats.json').read_bytes()).hexdigest(),'method':'official RunningStats, training states and 32-step action windows with episode-end padding; validation excluded'}
(outdir/'split_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2),flush=True)
