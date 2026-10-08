#!/usr/bin/env python3
"""Prepare train-only native16 normalization and an explicit embodiment report."""
from pathlib import Path
import argparse,json,hashlib,sys
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT))
from g1d_experiments.c01_contract import read_episodes,statistics,NAMES,SCHEMA
p=argparse.ArgumentParser();p.add_argument('--train',type=Path,required=True);p.add_argument('--validation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
_,train=read_episodes(a.train);_,val=read_episodes(a.validation)
tr={e['source_episode_index'] for e in train};va={e['source_episode_index'] for e in val}
if tr&va:raise ValueError('Source episode leakage')
split=json.loads((ROOT/'data/split_manifest.json').read_text())
if tr!=set(split['train_source_episodes']) or va!=set(split['validation_source_episodes']):raise ValueError('Dataset differs from original grouped split')
a.output.mkdir(parents=True,exist_ok=True)
st=statistics(train);text=json.dumps(st,indent=2)+'\n';(a.output/'statistics.json').write_text(text)
report={'schema':SCHEMA,'state_dim':16,'action_dim':16,'state_order':list(NAMES),'decoder_raw_state_dim':16,
        'policy_state_padding':'separate from raw state; never pad to DexJoCo23/46D',
        'upstream_state_dims':[23,46],'upstream_action_dims':[22,44],
        'train_episodes':len(train),'validation_episodes':len(val),'train_frames':sum(e['length'] for e in train),'validation_frames':sum(e['length'] for e in val),
        'train_source_episodes':len(tr),'validation_source_episodes':len(va),'group_overlap':0,
        'statistics_sha256':hashlib.sha256(text.encode()).hexdigest(),
        'response':'consecutive measured qpos delta, native16; no synthetic fingertip/tactile data'}
(a.output/'STATE_CONTRACT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
