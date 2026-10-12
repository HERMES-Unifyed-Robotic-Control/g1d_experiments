#!/usr/bin/env python3
from pathlib import Path
import os,sys,json,dataclasses,hashlib
ROOT=Path(__file__).resolve().parents[1];os.environ.setdefault('PSI_HOME',str(ROOT/'vendor/openpi'));os.environ.setdefault('DATA_HOME',str(ROOT/'artifacts/data'));sys.path[:0]=[str(ROOT),str(ROOT/'vendor/openpi/src'),str(ROOT/'vendor/openpi/src/openpi/openpi-client/src')]
import numpy as np
import jax
from openpi.training import config,data_loader
from openpi import transforms
from openpi.shared import normalize
from openpi.models import model as model_lib
from g1d_experiments.c01_pi_data import G1DPiData
root=Path(sys.argv[1]);norm=Path(sys.argv[2]);out=root/'run/matched_20261012/baseline7500';out.mkdir(parents=True,exist_ok=True)
base=config.get_config('g1d_subtasks_pi05_official_lora_holdout_v1');vc=dataclasses.replace(base.data,repo_id=str(root/'data/g1d_subtasks_official_validation_v1'));dc=vc.create(base.assets_dirs,base.model);ds=data_loader.create_torch_dataset(dc,32,base.model);repack=transforms.compose(dc.repack_transforms.inputs)
stats=normalize.load(norm.parent);inp=transforms.compose([*dc.data_transforms.inputs,transforms.Normalize(stats,use_quantiles=dc.use_quantile_norm,legacy_quantile=dc.legacy_quantile_norm),*dc.model_transforms.inputs]);data=G1DPiData(root/'data/g1d_subtasks_official_validation_v1',root/'run/cache','validation',norm)
anchors=json.loads((ROOT/'data/validation_samples.json').read_text());rows=[];raw=[];futures=[];valid=[];truth=[];sources=[];episodes=[];frames=[];prompts=[];hashes=[]
def digest(v):
 h=hashlib.sha256()
 for k,x in sorted(jax.tree_util.tree_flatten_with_path(v)[0],key=lambda p:str(p[0])):
  x=np.ascontiguousarray(x);h.update(str(k).encode());h.update(str(x.dtype).encode());h.update(str(x.shape).encode());h.update(x.tobytes())
 return h.hexdigest()
for n,(index,task) in enumerate(zip(anchors['indices'],anchors['task_indices'])):
 item=repack(ds[index]);item.pop('actions');obs=model_lib.Observation.from_dict(jax.tree.map(lambda x:jax.numpy.asarray(x),inp(item)));base_inputs={k:getattr(obs,k) for k in ['images','image_masks','state','tokenized_prompt','tokenized_prompt_mask']};s=data.sample(index);c_inputs={k:s[k] for k in base_inputs};diff={str(k):float(np.max(np.abs(np.asarray(a).astype(float)-np.asarray(b).astype(float)))) for (k,a),(_,b) in zip(jax.tree_util.tree_flatten_with_path(base_inputs)[0],jax.tree_util.tree_flatten_with_path(c_inputs)[0])}
 pixels_equal=all(np.array_equal(np.rint((np.asarray(base_inputs['images'][k])+1)*127.5).astype(np.uint8),np.rint((np.asarray(c_inputs['images'][k])+1)*127.5).astype(np.uint8)) for k in c_inputs['images']);nonimages_equal=all(np.array_equal(np.asarray(base_inputs[k]),np.asarray(c_inputs[k])) if not isinstance(c_inputs[k],dict) else all(np.array_equal(np.asarray(base_inputs[k][q]),np.asarray(v)) for q,v in c_inputs[k].items()) for k in ['state','image_masks','tokenized_prompt','tokenized_prompt_mask']);exact=pixels_equal and nonimages_equal and max(diff.values())<=2e-7;rows.append({'ordinal':n,'index':index,'equal':exact,'raw_rgb_pixels_equal':pixels_equal,'state_masks_prompt_tokens_equal':nonimages_equal,'max_abs_diffs':diff,'baseline_hash':digest(base_inputs),'c01_hash':digest(c_inputs)});hashes.append(digest(c_inputs))
 ei=int(np.searchsorted(data.ends,index,side='right'));ep=data.episodes[ei];frame=index-(int(data.ends[ei-1]) if ei else 0);raw.append(s['raw_state']);valid.append(s['action_valid']);truth.append(s['target_actions']);futures.append(ep['states'][np.minimum(frame+np.arange(30),ep['length']-1)]);sources.append(ep['source_episode_index']);episodes.append(ep['episode_index']);frames.append(frame);prompts.append(data.tasks[task])
 if (n+1)%20==0:print('INPUT_AUDIT',n+1,'matched',sum(x['equal'] for x in rows),flush=True)
report={'all_equal':all(x['equal'] for x in rows),'observations':len(rows),'rows':rows,'baseline_source_npz_sha256':hashlib.sha256((ROOT/'baseline7500/heldout_predictions.npz').read_bytes()).hexdigest(),'note':'Same uint8 RGB pixels, state/masks/prompt tokens exact; normalized image floats allowed<=2e-7 for NumPy divide vs JAX reciprocal arithmetic. Original baseline tensor hash is preserved separately; not claimed bitwise equal to C01 float image tensors.'};(out/'INPUT_AUDIT.json').write_text(json.dumps(report,indent=2));assert report['all_equal'],'Baseline input tensors differ; do not attach historical predictions'
with np.load(ROOT/'baseline7500/heldout_predictions.npz') as z:
 pred=z['predictions'][:,:30,:16];target=z['targets'][:,:30,:16];tasks=z['task_indices'];assert np.array_equal(tasks,anchors['task_indices']);assert np.allclose(target,np.stack(truth),atol=1e-6,rtol=0)
 np.savez_compressed(out/'predictions.npz',predictions=pred,targets=target,task_indices=tasks,indices=np.asarray(anchors['indices']),action_valid=np.stack(valid),raw_states=np.stack(raw),future_states=np.stack(futures),source_episode_indices=np.asarray(sources),subtask_episode_indices=np.asarray(episodes),frame_indices=np.asarray(frames),input_sha256=np.asarray([x['baseline_hash'] for x in rows]),canonical_observation_sha256=np.asarray(hashes),prompts=np.asarray(prompts))
metadata=json.loads((ROOT/'baseline7500/heldout_action_metrics.json').read_text());metadata.update(reuse='historical baseline7500 predictions with current baseline-vs-C01 tensor equivalence audit',observations=200,anchor_sha256=hashlib.sha256((ROOT/'data/validation_samples.json').read_bytes()).hexdigest());(out/'action_metrics.json').write_text(json.dumps(metadata,indent=2));(out/'EVALUATION_COMPLETE.json').write_text(json.dumps({'complete':True,'observations':200,'input_audit_all_equal':True}));print('BASELINE_ATTACHED_INPUTS_VERIFIED',flush=True)
