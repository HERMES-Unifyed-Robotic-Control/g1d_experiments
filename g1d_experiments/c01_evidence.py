"""Native16 low-dimensional windows with real sparse DINO consequences."""
from pathlib import Path
import json
import numpy as np
from .c01_contract import read_episodes,HORIZON
class G1DEvidence:
    def __init__(self,root,evidence,split):
        self.root=Path(root);self.evidence=Path(evidence)/split;self.split=split
        _,self.episodes=read_episodes(root)
        m=json.loads((Path(evidence)/'world_complete.json').read_text())
        if not m.get('complete') or m['schema']!='g1d_c01_visual_v1':raise ValueError('Complete G1D visual evidence required')
        self.ends=np.cumsum([e['length'] for e in self.episodes]);self.world={}
    def __len__(self):return int(self.ends[-1])
    def locate(self,index):
        if not 0<=index<len(self):raise IndexError(index)
        ei=int(np.searchsorted(self.ends,index,side='right'));return ei,index-(int(self.ends[ei-1]) if ei else 0)
    def sample(self,index):
        ei,t=self.locate(index);ep=self.episodes[ei];n=ep['length'];ids=t+np.arange(HORIZON)
        take=np.minimum(ids,n-1);nxt=np.minimum(ids+1,n-1)
        delta=ep['states'][nxt]-ep['states'][take]
        b={'action':ep['actions'][take],'state':ep['states'][t],'response':delta,
           'valid':ids<n,'response_valid':ids+1<n,
           'world':np.zeros((6,8,136),np.float32),'world_valid':np.zeros((6,8),bool),
           'world_query':np.zeros((6,8,4),np.float32),'world_target':np.zeros((6,8,66),np.float32)}
        if ei not in self.world:
            with np.load(self.evidence/f'episode_{ep["episode_index"]:06d}'/'world_intervals.npz') as z:self.world[ei]={k:z[k] for k in z.files}
        w=self.world[ei];selected=np.flatnonzero((w['start']>=t)&(w['end']<=min(t+30,n-1)))
        if len(selected)>6:raise ValueError('Unexpected visual interval count')
        for k,j in enumerate(selected):
            raw=w['raw'][j].copy();raw[:,70]=(int(w['start'][j])-t)/30;raw[:,71]=(int(w['end'][j])-t)/30
            b['world'][k]=raw;b['world_valid'][k]=w['valid'][j];b['world_target'][k]=raw[:,:66];b['world_query'][k]=raw[:,[66,67,70,71]]
        return b
    def batch(self,indices,device):
        import torch
        rows=[self.sample(int(i)) for i in indices]
        return {k:torch.from_numpy(np.stack([r[k] for r in rows])).to(device) for k in rows[0]}
