#!/usr/bin/env python3
"""Prepare a deployable profile for a verified JAX checkpoint; never activate it."""
from pathlib import Path
import json,hashlib,shutil
r=Path('/mnt/data/yangky/test/xr_teleoperate_g1d');report=r/'reports/pi05_official_holdout'
assert (report/'EVALUATION_COMPLETE').exists(),'Heldout evaluation must complete first'
best=json.loads((report/'best_checkpoint.json').read_text());split=json.loads((report/'split_manifest.json').read_text());cp=Path(best['checkpoint'])
profile=json.loads((r/'deploy/g1d_profiles.json').read_text())
profile['dataset']=split['datasets']['train']['path'];profile['asset_id']='g1d_subtasks_train_v1'
profile['profiles']['official']={'config':'g1d_subtasks_pi05_official_lora_holdout_v1','training_config':'g1d_subtasks_pi05_official_lora_holdout_v1','checkpoint':str(cp),'normalization':'official_quantile_unclipped','norm_stats_sha256':split['stats']['sha256'],'backend':'jax_lora'}
path=r/'deploy/releases/official_lora_holdout_v1';path.mkdir(parents=True,exist_ok=True)
(path/'g1d_profiles.json').write_text(json.dumps(profile,indent=2))
(path/'selection.json').write_text(json.dumps(best,indent=2))
print('Prepared, NOT activated:',path)
