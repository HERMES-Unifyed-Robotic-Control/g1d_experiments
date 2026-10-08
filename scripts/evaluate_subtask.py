#!/usr/bin/env python3
"""Fixed 200-observation physical-action error evaluation; not robot success."""
from pathlib import Path
import dataclasses,json,hashlib,argparse,os,sys
ROOT=Path(__file__).parents[1]
os.environ.setdefault("PSI_HOME",str(ROOT/"vendor/openpi"));os.environ.setdefault("DATA_HOME",str(ROOT/"artifacts/data"))
sys.path.insert(0,str(ROOT/"vendor/openpi/src"))
import numpy as np
from openpi.training import config,data_loader
from openpi.policies import policy_config
from openpi import transforms
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--variant',choices=['lora','fullft'],default='lora');p.add_argument('--output',type=Path,default=ROOT/'artifacts/evaluation');a=p.parse_args()
REPORT=a.output;REPORT.mkdir(parents=True,exist_ok=True)
cfg=config.get_config('g1d_subtasks_pi05_official_lora_holdout_v1')
split=json.loads((ROOT/'data/split_manifest.json').read_text());samples=json.loads((ROOT/'data/validation_samples.json').read_text())
if a.variant=='fullft':cfg=dataclasses.replace(cfg,model=dataclasses.replace(cfg.model,paligemma_variant='gemma_2b',action_expert_variant='gemma_300m'))
checkpoint=a.checkpoint.resolve();stats=checkpoint/'assets/g1d_subtasks_train_v1/norm_stats.json'
assert hashlib.sha256(stats.read_bytes()).hexdigest()==split['stats']['sha256']
from openpi.shared import normalize
policy=policy_config.create_trained_policy(cfg,checkpoint,norm_stats=normalize.load(stats.parent))
vc=dataclasses.replace(cfg.data,repo_id=str(ROOT/'artifacts/data/g1d_subtasks_official_validation_v1'));dc=vc.create(cfg.assets_dirs,cfg.model)
ds=data_loader.create_torch_dataset(dc,32,cfg.model)
repack=transforms.compose(dc.repack_transforms.inputs)
errors=[];preds=[];targets=[];task_indices=[]
for i,(idx,task) in enumerate(zip(samples['indices'],samples['task_indices'])):
 raw=repack(ds[idx]);gt=np.asarray(raw.pop('actions'))[:,:16]
 noise=np.random.default_rng(20260924+i).standard_normal((32,32),dtype=np.float32)
 pred=policy.infer(raw,noise=noise)['actions'];assert pred.shape==(32,16) and np.isfinite(pred).all()
 errors.append(pred-gt);preds.append(pred);targets.append(gt);task_indices.append(task)
 if (i+1)%10==0:print(f'evaluated {i+1}/{len(samples["indices"])} heldout observations',flush=True)
e=np.stack(errors);tasks=np.asarray(task_indices)
def metrics(a):
 return {'mae':float(np.abs(a).mean()),'rmse':float(np.sqrt((a*a).mean())),'per_dim_mae':np.abs(a).mean(axis=(0,1)).tolist(),'left_arm_mae':float(np.abs(a[:,:,:7]).mean()),'right_arm_mae':float(np.abs(a[:,:,7:14]).mean()),'left_gripper_mae':float(np.abs(a[:,:,14]).mean()),'right_gripper_mae':float(np.abs(a[:,:,15]).mean()),'horizon_mae':np.abs(a).mean(axis=(0,2)).tolist()}
result={'checkpoint':str(checkpoint),'selection':'explicit checkpoint argument','variant':a.variant,'observations':len(e),'normalization':'official_quantile_unclipped','stats_sha256':split['stats']['sha256'],'physical_units':'joint radians; gripper motor radians','overall':metrics(e),'per_task':{str(t):metrics(e[tasks==t]) for t in sorted(set(tasks.tolist()))},'limitation':'Heldout demonstrations, not closed-loop robot success rate. This validation set was used for checkpoint selection.'}
(REPORT/'heldout_action_metrics.json').write_text(json.dumps(result,indent=2));np.savez_compressed(REPORT/'heldout_predictions.npz',predictions=np.stack(preds),targets=np.stack(targets),task_indices=tasks)
(REPORT/'EVALUATION_COMPLETE').write_text(str(checkpoint)+'\n');print(json.dumps(result,indent=2))
