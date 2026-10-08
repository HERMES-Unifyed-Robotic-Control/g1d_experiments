"""G1D joint-response adaptation of C01 Query ASV, E30x256, native16 decoder."""
from pathlib import Path
import sys
import torch
from torch import nn
import torch.nn.functional as F
from .c01_decoder import G1DC01Decoder
sys.path.insert(0,str(Path(__file__).parents[1]/'third_party/dexjoco'))
from afce_all11.codec import QueryFusion,SparseWorld,WorldDecoder
from effect_afce_v21.model.codec import ActionBranch,TemporalEffect
class G1DC01Codec(nn.Module):
    def __init__(self,statistics):
        super().__init__();self.action_encoder=ActionBranch(action_dim=16)
        self.robot_encoder=nn.Sequential(nn.Linear(16,64),nn.GELU(),nn.Linear(64,64))
        self.world=SparseWorld();self.fuser=QueryFusion();self.temporal=TemporalEffect()
        self.action_decoder=G1DC01Decoder(statistics)
        self.response_decoder=nn.Sequential(nn.Linear(256,256),nn.GELU(),nn.Linear(256,16));self.world_decoder=WorldDecoder()
        for key in ('response_mean','response_std','world_mean','world_std'):
            self.register_buffer(key,torch.as_tensor(statistics[key],dtype=torch.float32))
    def encode(self,b):
        d=self.action_decoder;a=self.action_encoder((b['action']-d.action_mean)/d.action_std)
        s=self.robot_encoder((b['response']-self.response_mean)/self.response_std)
        v=self.world((b['world']-self.world_mean)/self.world_std,b['world_valid'])
        return self.temporal(self.fuser(a,s,v))
    def forward(self,b):
        e=self.encode(b)
        return {'E':e,'action_normalized':self.action_decoder.normalized(e,b['state']),
                'response_normalized':self.response_decoder(e),'world_normalized':self.world_decoder(e,b['world_query'])}
    def losses(self,out,b,response_weight=.2,world_weight=.2):
        d=self.action_decoder;truth=(b['action']-d.action_mean)/d.action_std
        def masked(pred,target,valid):
            v=F.smooth_l1_loss(pred,target,reduction='none');mask=valid[...,None].expand_as(v)
            return (v*mask).sum()/mask.sum().clamp_min(1)
        groups=[masked(out['action_normalized'][...,lo:hi],truth[...,lo:hi],b['valid']) for lo,hi in ((0,7),(7,14),(14,15),(15,16))]
        la=torch.stack(groups).mean();ls=masked(out['response_normalized'],(b['response']-self.response_mean)/self.response_std,b['response_valid'])
        wt=(b['world_target']-self.world_mean[:66])/self.world_std[:66]
        lv=masked(out['world_normalized'][...,:64],wt[...,:64],b['world_valid'])+masked(out['world_normalized'][...,64:],wt[...,64:],b['world_valid'])
        return {'loss':la+response_weight*ls+world_weight*lv,'action':la,'response':ls,'world':lv}
