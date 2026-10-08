"""Serve either archived Sep 26 baseline or QC weights with the aligned client contract."""
from pathlib import Path
import argparse
import hashlib
import json
import logging
import os
import time
import numpy as np
from dotenv import load_dotenv
load_dotenv('/mnt/data/yangky/test/Psi0/.env')
from openpi.training import config
from openpi.policies import policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.shared import normalize
from openpi_client import image_tools

CLIENT_ROOT=Path('/mnt/data2/yangky/test/g1d_client_alignment_20261007')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',choices=('baseline','qc','joint197'),default='baseline')
    parser.add_argument('--port',type=int,default=None)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    pf=CLIENT_ROOT/f'g1d_profiles_{args.variant}.json'
    profiles=json.loads(pf.read_text());p=profiles['profiles']['official'];cp=Path(p['checkpoint'])
    if not (cp/'params').is_dir():raise FileNotFoundError(cp/'params')
    stats=list((cp/'assets').glob('*/norm_stats.json'))
    if len(stats)!=1 or sha(stats[0])!=p['norm_stats_sha256']:raise ValueError('checkpoint norm stats mismatch')
    contract=json.loads((cp/'g1d_contract.json').read_text())
    cfg=config.get_config(p['config'])
    meta={
        'contract_version':'g1d-pi05-v1','profile':'official','dataset':profiles['dataset'],
        'action_layout':'left_arm_7,right_arm_7,left_gripper,right_gripper','action_dim':16,
        'model_action_dim':32,'action_horizon':32,'fps':30,'prompt_mode':p['prompt_mode'],
        'image_preprocess':'openpi_client.resize_with_pad.PIL_BILINEAR',
        'normalization':'official_quantile_unclipped','backend':p['backend'],'checkpoint':str(cp),
        'training_config':p['training_config'],'norm_stats_sha256':sha(stats[0]),'profiles_sha256':sha(pf),
        'client_sha256':sha(CLIENT_ROOT/'deploy_g1d_food_sorting_pi05.py'),
        'runtime_helper_sha256':sha(CLIENT_ROOT/'robot_alignment.py'),
        'image_tools_sha256':sha(Path(image_tools.__file__)),
        'official_reference':profiles['official_reference'],'source_contract':contract,
        'control_release':'alignment_20261007','variant':args.variant,
    }
    print(json.dumps(meta,indent=2),flush=True)
    if args.check_only:return
    logging.basicConfig(level=logging.INFO,force=True)
    os.environ['OPENPI_DETERMINISTIC_NOISE']='0';os.environ['OPENPI_DIAGNOSTIC_OUTPUT']='0'
    policy=policy_config.create_trained_policy(cfg,cp,norm_stats=normalize.load(stats[0].parent))
    prompt='Move the black cup from the right side to the middle position.' if args.variant=='baseline' else 'Move the black cup from the right to the middle, place chrysanthemum into it, then pour water from the white cup from the left into the black cup.'
    obs={'state':np.zeros(16,np.float32),'prompt':prompt,
         **{k:np.zeros((224,224,3),np.uint8) for k in ('observation/base_image','observation/left_wrist_image','observation/right_wrist_image')}}
    for i in range(2):
        begin=time.monotonic();actions=np.asarray(policy.infer(obs)['actions'])
        if actions.shape!=(32,16) or not np.isfinite(actions).all():raise RuntimeError('invalid warmup actions')
        print(f'warmup {i+1}: {(time.monotonic()-begin)*1000:.1f} ms',flush=True)
    port=args.port or profiles['client']['policy_port']
    print(f'READY aligned variant={args.variant} checkpoint={cp} port={port}',flush=True)
    WebsocketPolicyServer(policy,host='0.0.0.0',port=port,metadata=meta).serve_forever()

if __name__=='__main__':main()
