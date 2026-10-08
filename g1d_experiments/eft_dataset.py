"""G1D LeRobot -> history/language inputs and separate future supervision.

This is a data adapter for EFT experiments, not an implementation of EFT training.
Never pass the returned targets dict to an encoder/U predictor.
"""
from pathlib import Path
import json
import numpy as np
import pyarrow.parquet as pq
import cv2
from torch.utils.data import Dataset

VIEWS=('egocentric','left_wrist','right_wrist')
class G1DPredictiveWindowDataset(Dataset):
    def __init__(self, root, history_frames=4, history_seconds=.5, horizon=45, stride=8, image_size=224):
        self.root=Path(root);self.info=json.loads((self.root/'meta/info.json').read_text())
        self.episodes=[json.loads(x) for x in (self.root/'meta/episodes.jsonl').read_text().splitlines()]
        self.tasks={x['task_index']:x['task'] for x in map(json.loads,(self.root/'meta/tasks.jsonl').read_text().splitlines())}
        self.history_frames=history_frames;self.horizon=horizon;self.image_size=image_size
        self.offsets=np.round(np.linspace(-history_seconds*self.info['fps'],0,history_frames)).astype(int)
        self.index=[]
        for e in self.episodes:
            for t in range(-int(self.offsets[0]),e['length']-horizon,stride):self.index.append((e['episode_index'],t))
        self.by_id={e['episode_index']:e for e in self.episodes}
    def __len__(self):return len(self.index)
    def path(self, template, ep, **extra):
        return self.root/template.format(episode_index=ep,episode_chunk=ep//self.info.get('chunks_size',1000),**extra)
    def frames(self, ep, view, indices):
        path=self.path(self.info['video_path'],ep,video_key='observation.images.'+view)
        cap=cv2.VideoCapture(str(path));images=[]
        try:
            for i in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES,int(i));ok,bgr=cap.read()
                if not ok:raise RuntimeError(f'Missing frame {i} in {path}')
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                # Letterbox preserves camera geometry; unnormalized uint8 RGB.
                h,w=rgb.shape[:2];scale=self.image_size/max(h,w)
                small=cv2.resize(rgb,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_LINEAR)
                out=np.zeros((self.image_size,self.image_size,3),np.uint8)
                y=(self.image_size-small.shape[0])//2;x=(self.image_size-small.shape[1])//2
                out[y:y+small.shape[0],x:x+small.shape[1]]=small;images.append(out.transpose(2,0,1))
        finally:cap.release()
        return np.stack(images)
    def __getitem__(self,index):
        ep,t=self.index[index];entry=self.by_id[ep]
        table=pq.read_table(self.path(self.info['data_path'],ep),columns=['observation.state','action','task_index'])
        state=np.asarray(table['observation.state'].to_pylist(),np.float32)
        action=np.asarray(table['action'].to_pylist(),np.float32)
        hist=t+self.offsets;future=t+np.arange(1,self.horizon+1)
        return {'inputs':{'rgb_history':np.stack([self.frames(ep,v,hist) for v in VIEWS],axis=1),
                          'state_history':state[hist],'language':self.tasks[int(table['task_index'][t].as_py())]},
                'targets':{'rgb_future':np.stack([self.frames(ep,v,future) for v in VIEWS],axis=1),
                           'state_future':state[future],
                           # action[t] is first policy command; future state begins t+1.
                           'action_future':action[t:t+self.horizon]},
                'metadata':{'episode_index':ep,'anchor_frame':t,'source_episode_index':entry.get('source_episode_index',ep),
                            'fps':self.info['fps'],'views':VIEWS}}


def collate_g1d_windows(samples):
    """Batch tensors without merging future supervision into predictor inputs."""
    import torch
    def stack(group,key):
        return torch.from_numpy(np.stack([sample[group][key] for sample in samples]))
    return {'inputs':{'rgb_history':stack('inputs','rgb_history'),
                      'state_history':stack('inputs','state_history'),
                      'language':[sample['inputs']['language'] for sample in samples]},
            'targets':{key:stack('targets',key) for key in ('rgb_future','state_future','action_future')},
            'metadata':[sample['metadata'] for sample in samples]}
