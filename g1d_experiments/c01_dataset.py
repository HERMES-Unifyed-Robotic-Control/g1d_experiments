"""Native G1D windows for C01 E supervision; these are not DexJoCo 44D actions.

The observation is available to pi at deployment. Supervision is offline-only.
Measured joint response is a G1D adaptation, not the upstream FK fingertip target.
"""
from pathlib import Path
import json
import numpy as np
import pyarrow.parquet as pq
import cv2
import sys
sys.path.insert(0,str(Path(__file__).parents[1]/"vendor/openpi/src/openpi/openpi-client/src"))
from openpi_client.image_tools import resize_with_pad
from torch.utils.data import Dataset
VIEWS=('egocentric','left_wrist','right_wrist')
class G1DC01Windows(Dataset):
    def __init__(self,root,horizon=30,stride=8,image_size=224):
        self.root=Path(root);self.horizon=horizon;self.image_size=image_size
        self.info=json.loads((self.root/'meta/info.json').read_text())
        if self.info['fps']!=30:raise ValueError('C01 G1D contract requires 30Hz')
        self.episodes=[json.loads(x) for x in (self.root/'meta/episodes.jsonl').read_text().splitlines()]
        self.by_id={e['episode_index']:e for e in self.episodes}
        self.tasks={e['task_index']:e['task'] for e in map(json.loads,(self.root/'meta/tasks.jsonl').read_text().splitlines())}
        self.index=[(e['episode_index'],t) for e in self.episodes for t in range(0,e['length']-horizon,stride)]
    def __len__(self):return len(self.index)
    def path(self,template,ep,**kw):return self.root/template.format(episode_index=ep,episode_chunk=ep//self.info.get('chunks_size',1000),**kw)
    def frame(self,ep,view,t):
        cap=cv2.VideoCapture(str(self.path(self.info['video_path'],ep,video_key='observation.images.'+view)))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES,t);ok,bgr=cap.read()
            if not ok:raise RuntimeError(f'Missing frame: {ep}/{view}/{t}')
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            return resize_with_pad(rgb,self.image_size,self.image_size).transpose(2,0,1)
        finally:cap.release()
    def __getitem__(self,i):
        ep,t=self.index[i];table=pq.read_table(self.path(self.info['data_path'],ep),columns=['observation.state','action','task_index'])
        state=np.asarray(table['observation.state'].to_pylist(),np.float32);action=np.asarray(table['action'].to_pylist(),np.float32)
        if state.shape[-1]!=16 or action.shape[-1]!=16:raise ValueError('Expected native G1D 16D qpos')
        future=np.arange(t+1,t+self.horizon+1)
        return {'observation':{'rgb':np.stack([self.frame(ep,v,t) for v in VIEWS]),'state':state[t],
                                'prompt':self.tasks[int(table['task_index'][t].as_py())]},
                'supervision':{'action':action[t:t+self.horizon],'state_future':state[future],
                               'joint_response':state[future]-state[future-1], 'valid':np.ones(self.horizon,bool),
                               'future_frame_indices':future},
                'metadata':{'episode_index':ep,'source_episode_index':self.by_id[ep].get('source_episode_index',ep),
                            'anchor_frame':t,'fps':30,'views':VIEWS,'response_protocol':'g1d_measured_joint_delta_v1'}}

def collate_c01_windows(samples):
    import torch
    def stack(group,key):return torch.from_numpy(np.stack([s[group][key] for s in samples]))
    return {'observation':{'rgb':stack('observation','rgb'),'state':stack('observation','state'),
                           'prompt':[s['observation']['prompt'] for s in samples]},
            'supervision':{key:stack('supervision',key) for key in ('action','state_future','joint_response','valid','future_frame_indices')},
            'metadata':[s['metadata'] for s in samples]}
