"""Native G1D embodiment contract; padding is never an embodiment conversion."""
from pathlib import Path
import json
import numpy as np
NAMES=('left_shoulder_pitch','left_shoulder_roll','left_shoulder_yaw','left_elbow',
       'left_wrist_roll','left_wrist_pitch','left_wrist_yaw',
       'right_shoulder_pitch','right_shoulder_roll','right_shoulder_yaw','right_elbow',
       'right_wrist_roll','right_wrist_pitch','right_wrist_yaw','left_gripper','right_gripper')
GROUPS=((0,7),(7,14),(14,15),(15,16))
HORIZON=30
SCHEMA='g1d_c01_native16_joint_response_v1'
def native_state(state):
    q=np.asarray(state,np.float32)
    if q.shape[-1]!=16 or not np.isfinite(q).all():raise ValueError('G1D decoder requires finite raw16D state, never DexJoCo23/46D or padded state')
    return q

def read_episodes(root):
    import pyarrow.parquet as pq
    root=Path(root);info=json.loads((root/'meta/info.json').read_text())
    if info['fps']!=30:raise ValueError('Expected G1D30Hz')
    for key in ('action','observation.state'):
        spec=info['features'][key]
        if spec['shape']!=[16] or tuple(spec['names'])!=NAMES:raise ValueError(f'{key} G1D order/dim mismatch')
    episodes=[]
    for line in (root/'meta/episodes.jsonl').read_text().splitlines():
        e=json.loads(line);ep=e['episode_index'];p=root/info['data_path'].format(episode_index=ep,episode_chunk=ep//info.get('chunks_size',1000))
        t=pq.read_table(p,columns=['observation.state','action','frame_index','timestamp'])
        q=native_state(t['observation.state'].to_pylist());a=native_state(t['action'].to_pylist());n=e['length']
        if q.shape!=(n,16) or a.shape!=(n,16):raise ValueError(f'Length mismatch:{ep}')
        if not np.array_equal(np.asarray(t['frame_index']),np.arange(n)):raise ValueError(f'Frame index mismatch:{ep}')
        ts=np.asarray(t['timestamp']);
        if not np.allclose(ts,np.arange(n)/30,rtol=0,atol=2e-5):raise ValueError(f'Timestamp mismatch:{ep}')
        episodes.append(dict(e,states=q,actions=a))
    return info,episodes

def statistics(episodes):
    out={'schema':SCHEMA,'state_dim':16,'action_dim':16,'state_order':list(NAMES),'action_order':list(NAMES),'fit':'train_source_episodes_only'}
    for key,field in [('state','states'),('action','actions')]:
        x=np.concatenate([e[field] for e in episodes]).astype(np.float64)
        out[key+'_mean']=x.mean(0).astype(np.float32).tolist();out[key+'_std']=x.std(0).clip(1e-3).astype(np.float32).tolist()
    x=np.concatenate([np.diff(e['states'],axis=0) for e in episodes]).astype(np.float64)
    out['response_mean']=x.mean(0).astype(np.float32).tolist();out['response_std']=x.std(0).clip(1e-3).astype(np.float32).tolist()
    out['train_source_episodes']=sorted({e['source_episode_index'] for e in episodes})
    return out
