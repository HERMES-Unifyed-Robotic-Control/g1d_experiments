"""16D G1D action decoder and GT-AUX / gripper-delta objective components.

A decoder component, not a trained G1D C01 policy or complete E/pi training loop.
Uses the upstream decoder architecture, with newly initialized G1D dimensions.
"""
from pathlib import Path
import sys
import torch
from torch import nn
import torch.nn.functional as F
ROOT=Path(__file__).parents[1]
sys.path.insert(0,str(ROOT/'third_party/dexjoco'))
from effect_afce_v21.model.codec import ActionDecoder
GROUPS=(slice(0,7),slice(7,14),slice(14,15),slice(15,16))
class G1DC01Decoder(nn.Module):
    def __init__(self,statistics,horizon=30):
        super().__init__();self.decoder=ActionDecoder(effect_dim=256,state_dim=16,action_dim=16,horizon=horizon)
        for key in ('state_mean','state_std','action_mean','action_std'):
            value=torch.as_tensor(statistics[key],dtype=torch.float32)
            if value.shape!=(16,):raise ValueError(f'{key} must be training-only 16D statistics')
            if 'std' in key and torch.any(value<=0):raise ValueError('Nonpositive normalization scale')
            self.register_buffer(key,value)
    def normalized(self,e,state):return self.decoder(e,(state-self.state_mean)/self.state_std)
    def forward(self,e,state):return self.normalized(e,state)*self.action_std+self.action_mean
    def auxiliary_losses(self,e,state,target,valid,delta_scale_floor=.1):
        pred=self.normalized(e,state);truth=((target-self.action_mean)/self.action_std).detach()
        groups=[]
        for sl in GROUPS:
            values=F.smooth_l1_loss(pred[...,sl],truth[...,sl],reduction='none');mask=valid[...,None].expand_as(values)
            groups.append((values*mask).sum((1,2))/mask.sum((1,2)).clamp_min(1))
        gtaux=torch.stack(groups).mean(0)
        dp=pred[:,1:,14:16]-pred[:,:-1,14:16];dt=truth[:,1:,14:16]-truth[:,:-1,14:16]
        mask=(valid[:,1:]&valid[:,:-1])[...,None].expand_as(dt)
        scale=((dt.square()*mask).sum((0,1),keepdim=True)/mask.sum((0,1),keepdim=True).clamp_min(1)+1e-8).sqrt().clamp_min(delta_scale_floor).detach()
        values=F.smooth_l1_loss(dp/scale,dt/scale,reduction='none')
        delta=(values*mask).sum((1,2))/mask.sum((1,2)).clamp_min(1)
        return {'gtaux_per_sample':gtaux,'gripper_delta_per_sample':delta}
