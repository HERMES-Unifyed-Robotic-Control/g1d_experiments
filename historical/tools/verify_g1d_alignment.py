#!/usr/bin/env python3
"""Real-data transform and contract regression checks, without robot commands."""
import ast
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import cv2
import numpy as np
from dotenv import load_dotenv
load_dotenv('/mnt/data/yangky/test/Psi0/.env')
from openpi import transforms as t
from openpi.training import config, data_loader
from openpi_client import image_tools

root = Path(__file__).resolve().parents[1]
profiles_path = root / 'deploy/g1d_profiles.json'
profiles = json.loads(profiles_path.read_text())
# Extract pure client definitions without importing cameras/DDS or initializing hardware.
client_path = root / 'deploy/deploy_g1d_food_sorting_pi05.py'
tree = ast.parse(client_path.read_text())
names = {'rgb224', 'PromptSchedule', 'validate_contract'}
selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
constants = [n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(x,ast.Name) and x.id in {'SUBTASK_NAMES','SUBTASK_FRACTIONS','INGREDIENTS'} for x in n.targets)]
ns = {'__file__': str(client_path), 'np':np, 'cv2':cv2, 'image_tools':image_tools,
      'dataclass':dataclasses.dataclass, 'Path':Path, 'hashlib':hashlib, 'json':json}
exec(compile(ast.Module(body=constants+selected,type_ignores=[]),str(client_path),'exec'),ns)

dataset_root=Path(profiles['dataset'])
info=json.loads((dataset_root/'meta/info.json').read_text())
assert info['fps']==30
assert info['features']['action']['shape']==[16]
assert info['features']['observation.state']['shape']==[16]
assert info['features']['action']['names']==info['features']['observation.state']['names']
task_texts={json.loads(line)['task'] for line in (dataset_root/'meta/tasks.jsonl').read_text().splitlines()}
client_prompts=set()
for ingredient in ['chrysanthemum','red_dates','lemon_slices']:
    schedule=ns['PromptSchedule']('subtask',ingredient,300,(.220,.355,.134,.089,.202),'unused')
    client_prompts.update(schedule._subtask_prompt(i) for i in range(5))
assert client_prompts==task_texts
cfg = config.get_config(profiles['profiles']['official']['config'])
dc = cfg.data.create(cfg.assets_dirs,cfg.model)
assert dc.legacy_quantile_norm is False and dc.use_quantile_norm
assert dc.asset_id == profiles['asset_id'] and not Path(dc.asset_id).is_absolute()
stat_file = cfg.assets_dirs / dc.asset_id / 'norm_stats.json'
assert hashlib.sha256(stat_file.read_bytes()).hexdigest() == profiles['profiles']['official']['norm_stats_sha256']
# Values outside q01/q99 must be preserved by the official round-trip.
st = t.NormStats(mean=np.zeros(2),std=np.ones(2),q01=np.array([1.,2.]),q99=np.array([3.,4.]))
x=np.array([[-2.,8.]],dtype=np.float32); before=st.q99.copy()
y=t.Normalize({'actions':st},use_quantiles=True,legacy_quantile=False)({'actions':x})
np.testing.assert_allclose(y['actions'],(x-st.q01)/(st.q99-st.q01+1e-6)*2-1)
z=t.Unnormalize({'actions':st},use_quantiles=True)(y)['actions']
np.testing.assert_allclose(z,x,atol=1e-6);np.testing.assert_array_equal(before,st.q99)
y_old=t.Normalize({'actions':st},use_quantiles=True)({'actions':x})['actions']
assert np.max(np.abs(y_old)) <= 1
# Exercise the actual LeRobot dataset and training adapter/model transforms.
ds=data_loader.create_torch_dataset(dc,cfg.model.action_horizon,cfg.model)
raw=ds[0]
repacked=t.compose(dc.repack_transforms.inputs)(raw)
normalize=t.Normalize(dc.norm_stats,use_quantiles=True,legacy_quantile=False)
train=t.compose([*dc.data_transforms.inputs,normalize,*dc.model_transforms.inputs])(repacked)
wire={k:v for k,v in repacked.items() if k!='actions'}
server=t.compose([*dc.data_transforms.inputs,normalize,*dc.model_transforms.inputs])(wire)
for key in ['state','tokenized_prompt','tokenized_prompt_mask']:
 np.testing.assert_array_equal(train[key],server[key])
for key in train['image']:
 np.testing.assert_array_equal(train['image'][key],server['image'][key])
assert train['actions'].shape==(32,32) and np.all(train['actions'][:,16:]==0)
assert np.all(train['state'][16:]==0) and np.isfinite(train['actions']).all()
for key in ['observation/base_image','observation/left_wrist_image','observation/right_wrist_image']:
 rgb=np.asarray(repacked[key]);rgb=(rgb*255).astype(np.uint8) if np.issubdtype(rgb.dtype,np.floating) else rgb
 if rgb.shape[0]==3: rgb=np.transpose(rgb,(1,2,0))
 np.testing.assert_array_equal(ns['rgb224'](cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)),image_tools.resize_with_pad(rgb,224,224))
schedule=ns['PromptSchedule']('subtask','chrysanthemum',300,(.220,.355,.134,.089,.202),'unused')
assert schedule.prompt_for_step(0)==schedule.prompt_for_step(299)
meta=dict(cfg.policy_metadata);meta.update(client_sha256=hashlib.sha256(client_path.read_bytes()).hexdigest(),checkpoint=profiles['profiles']['official']['checkpoint'],norm_stats_sha256=profiles['profiles']['official']['norm_stats_sha256'],image_tools_sha256=hashlib.sha256(Path(image_tools.__file__).read_bytes()).hexdigest(),profiles_sha256=hashlib.sha256(profiles_path.read_bytes()).hexdigest())
args=SimpleNamespace(profile='official',fps=30,prompt_mode='subtask',execute_chunk_steps=8)
ns['validate_contract'](args,meta)
for key,value in [('profile','legacy'),('fps',20),('norm_stats_sha256','bad'),('image_tools_sha256','bad')]:
 bad={**meta,key:value}
 try: ns['validate_contract'](args,bad)
 except ValueError: pass
 else: raise AssertionError(f'contract mismatch not rejected: {key}')
# Exercise the actual checkpoint writer with a tiny CPU model (no training/GPU).
import tempfile
import torch
from openpi.train_pytorch import save_checkpoint
with tempfile.TemporaryDirectory(prefix="g1d-checkpoint-test-") as tmp:
    tiny=torch.nn.Linear(2,2)
    opt=torch.optim.AdamW(tiny.parameters())
    test_cfg=dataclasses.replace(cfg,checkpoint_base_dir=tmp,exp_name="contract_test")
    save_checkpoint(tiny,opt,500,test_cfg,True,dc)
    saved=test_cfg.checkpoint_dir / "500"
    frozen=saved / "assets" / dc.asset_id / "norm_stats.json"
    assert frozen.is_file()
    assert hashlib.sha256(frozen.read_bytes()).hexdigest()==hashlib.sha256(stat_file.read_bytes()).hexdigest()
    provenance=json.loads((saved / "g1d_contract.json").read_text())
    assert provenance["normalization"]=="official_quantile_unclipped"
    assert provenance["norm_stats_sha256"]==hashlib.sha256(frozen.read_bytes()).hexdigest()
    assert hashlib.sha256(stat_file.read_bytes()).hexdigest()==profiles["profiles"]["official"]["norm_stats_sha256"]
print(json.dumps({'status':'PASS','dataset_frames':len(ds),'training_actions_shape':list(train['actions'].shape),'official_roundtrip':'PASS','legacy_behavior':'PASS','train_server_transform_equality':'PASS','client_image_equality':'PASS','fixed_subtask':'PASS','checkpoint_stats_isolation':'PASS','mismatch_rejection':'PASS'},indent=2))
