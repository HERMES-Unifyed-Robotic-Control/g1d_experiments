"""Portable step7500 policy server; use matching client from this repository."""
from pathlib import Path
import argparse,os,json,hashlib,sys,dataclasses
ROOT=Path(__file__).parents[1]
os.environ.setdefault('PSI_HOME',str(ROOT/'vendor/openpi'));os.environ.setdefault('DATA_HOME',str(ROOT/'artifacts/data'))
sys.path.insert(0,str(ROOT/'vendor/openpi/src'))
from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import normalize
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi_client import image_tools
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,default=ROOT/'artifacts/baseline/7500');p.add_argument('--port',type=int,default=9016);a=p.parse_args()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
cp=a.checkpoint.resolve();pf=ROOT/'deploy/client/g1d_profiles_baseline.json';profiles=json.loads(pf.read_text());ref=profiles['profiles']['official'];ref['checkpoint']=str(cp);pf.write_text(json.dumps(profiles,indent=2)+'\n');stats=list((cp/'assets').glob('*/norm_stats.json'))
assert len(stats)==1 and sha(stats[0])==ref['norm_stats_sha256']
cfg=config.get_config(ref['config']);policy=policy_config.create_trained_policy(cfg,cp,norm_stats=normalize.load(stats[0].parent))
meta={'contract_version':'g1d-pi05-v1','profile':'official','dataset':profiles['dataset'],'action_layout':'left_arm_7,right_arm_7,left_gripper,right_gripper','action_dim':16,'model_action_dim':32,'action_horizon':32,'fps':30,'prompt_mode':'subtask','image_preprocess':'openpi_client.resize_with_pad.PIL_BILINEAR','normalization':'official_quantile_unclipped','backend':'jax_lora','checkpoint':str(cp),'norm_stats_sha256':sha(stats[0]),'profiles_sha256':sha(pf),'client_sha256':sha(ROOT/'deploy/client/deploy_g1d_food_sorting_pi05.py'),'runtime_helper_sha256':sha(ROOT/'deploy/client/robot_alignment.py'),'image_tools_sha256':sha(Path(image_tools.__file__))}
WebsocketPolicyServer(policy,host='0.0.0.0',port=a.port,metadata=meta).serve_forever()
