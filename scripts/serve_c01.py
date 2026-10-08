#!/usr/bin/env python3
"""Current-observation-only C01 policy server with native16 physical outputs."""
from pathlib import Path
import argparse, dataclasses, hashlib, json, os, sys
import threading
ROOT = Path(__file__).parents[1]
os.environ.setdefault('PSI_HOME', str(ROOT/'vendor/openpi'))
os.environ.setdefault('DATA_HOME', str(ROOT/'artifacts/data'))
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
sys.path[:0] = [str(ROOT), str(ROOT/'vendor/openpi/src'), str(ROOT/'vendor/openpi/src/openpi/openpi-client/src')]
import jax
import numpy as np
from openpi.models import model as model_lib
from openpi.training import config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi_client import image_tools
from g1d_experiments.c01_pi import G1DC01PiConfig
from g1d_experiments.c01_pi_data import G1DPolicyInputs

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', type=Path, required=True, help='pi/checkpoints/<step> directory')
p.add_argument('--cache', type=Path, required=True)
p.add_argument('--port', type=int, default=9021)
p.add_argument('--variant', choices=['lora', 'fullft'], default='lora')
p.add_argument('--denoise-steps', type=int, default=10)
a = p.parse_args()
cp = a.checkpoint.resolve()
stats = cp/'assets/g1d_subtasks_train_v1/norm_stats.json'
manifest = json.loads((a.cache/'manifest.json').read_text())
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()
for name, key in [('decoder_init.npz', 'decoder_sha256'), ('normalization.npz', 'normalization_sha256')]:
    if sha(a.cache/name) != manifest[key]: raise ValueError('C01 asset hash mismatch: '+name)
values = dataclasses.asdict(config.get_config('g1d_subtasks_pi05_official_lora_holdout_v1').model)
values.update(action_horizon=30, action_dim=32, decoder_init=str(a.cache/'decoder_init.npz'))
if a.variant == 'fullft': values.update(paligemma_variant='gemma_2b', action_expert_variant='gemma_300m')
model = G1DC01PiConfig(**values).load(model_lib.restore_params(cp/'params'))
model.eval()
inputs = G1DPolicyInputs(stats)

@jax.jit
def predict(obs, raw, key):
    effect = model.sample_actions(key, obs, num_steps=a.denoise_steps)
    return model.decoder.physical(effect*model.effect_std.value+model.effect_mean.value, raw)

class C01Policy:
    def __init__(self): self.key = jax.random.key(42); self.lock = threading.Lock()
    def infer(self, obs):
        raw = np.asarray(obs['state'], dtype=np.float32)
        rgb = [obs[k] for k in ('observation/base_image', 'observation/left_wrist_image', 'observation/right_wrist_image')]
        data = inputs.inputs(rgb, raw, str(obs['prompt']))
        observation = model_lib.Observation(**jax.tree.map(lambda x: x[None], data))
        with self.lock:
            self.key, key = jax.random.split(self.key)
            actions = np.asarray(predict(observation, raw[None], key))[0]
        if actions.shape != (30, 16) or not np.isfinite(actions).all(): raise ValueError('Invalid C01 physical actions')
        return {'actions': actions}

client = ROOT/'deploy/c01_client'
pf = client/'g1d_profiles_baseline.json'
profiles = json.loads(pf.read_text())
ref = profiles['profiles']['official']
if sha(stats) != ref['norm_stats_sha256']: raise ValueError('State normalization differs from baseline')
ref['checkpoint'] = str(cp)
pf.write_text(json.dumps(profiles, indent=2)+'\n')
meta = dict(contract_version='g1d-c01-native16-v1', profile='official', dataset=profiles['dataset'],
    action_layout='left_arm_7,right_arm_7,left_gripper,right_gripper', action_dim=16,
    model_action_dim=256, action_horizon=30, fps=30, prompt_mode='subtask',
    image_preprocess='openpi_client.resize_with_pad.PIL_BILINEAR', normalization='official_quantile_unclipped',
    checkpoint=str(cp), backend='jax_c01_'+a.variant, norm_stats_sha256=sha(stats),
    profiles_sha256=sha(pf), client_sha256=sha(client/'deploy_g1d_food_sorting_pi05.py'),
    runtime_helper_sha256=sha(client/'robot_alignment.py'), image_tools_sha256=sha(image_tools.__file__),
    codec_sha256=manifest['checkpoint_sha256'], decoder_init_sha256=manifest['decoder_sha256'],
    denoise_steps=a.denoise_steps, raw_state_dim=16, padded_state_dim=32)
WebsocketPolicyServer(C01Policy(), host='0.0.0.0', port=a.port, metadata=meta).serve_forever()
