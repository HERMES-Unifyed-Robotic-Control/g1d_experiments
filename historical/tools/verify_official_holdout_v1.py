from dotenv import load_dotenv
load_dotenv('/mnt/data/yangky/test/Psi0/.env')
import json,dataclasses,hashlib
from pathlib import Path
import numpy as np
from openpi.training import config,data_loader
from openpi import transforms
report=Path('/mnt/data/yangky/test/xr_teleoperate_g1d/reports/pi05_official_holdout');m=json.loads((report/'split_manifest.json').read_text());c=config.get_config('g1d_subtasks_pi05_official_lora_holdout_v1');dc=c.data.create(c.assets_dirs,c.model)
assert not dc.legacy_quantile_norm and dc.use_quantile_norm
summary={}
for split in ['train','validation']:
 root=Path(m['datasets'][split]['path']);rows=[json.loads(l) for l in (root/'meta/episodes.jsonl').read_text().splitlines()]
 ids={r['source_episode_index'] for r in rows};assert ids==set(m['train_source_episodes'] if split=='train' else m['validation_source_episodes'])
 d=dataclasses.replace(dc,repo_id=str(root));ds=data_loader.create_torch_dataset(d,32,c.model);td=data_loader.transform_dataset(ds,d)
 a=td[0];assert a['actions'].shape==(32,32) and np.isfinite(a['actions']).all() and np.all(a['actions'][:,16:]==0)
 summary[split]={'episodes':len(rows),'source_episodes':len(ids),'frames':len(ds),'first_action_shape':list(a['actions'].shape)}
assert set(m['train_source_episodes']).isdisjoint(m['validation_source_episodes'])
asset=c.assets_dirs/dc.asset_id/'norm_stats.json';assert hashlib.sha256(asset.read_bytes()).hexdigest()==m['stats']['sha256']
x={'state':np.zeros(16,dtype=np.float32),'actions':np.zeros((1,16),dtype=np.float32)}
y=transforms.Normalize(dc.norm_stats,use_quantiles=True,legacy_quantile=False)(x)
z=transforms.Unnormalize(dc.norm_stats,use_quantiles=True)(y)
np.testing.assert_allclose(z['actions'],x['actions'],atol=1e-6)
summary.update(status='PASS',no_source_episode_overlap=True,train_only_stats_sha256=m['stats']['sha256'],official_zero_gripper_roundtrip=True)
(report/'preflight.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
