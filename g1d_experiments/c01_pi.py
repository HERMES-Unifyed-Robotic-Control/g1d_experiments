"""Native16 G1D pi05 predicting E30x256 with one jointly trained decoder."""
from pathlib import Path
import sys,dataclasses
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/'vendor/openpi/src'));sys.path.insert(0,str(ROOT/'third_party/dexjoco'))
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx,struct
from openpi.models import model as model_lib,gemma
from openpi.models.pi0 import Pi0,make_attn_mask
from openpi.models.pi0_config import Pi0Config
from openpi.shared import array_typing as at
from openpi.training import weight_loaders
from afce_all11.jax_action_decoder import ActionDecoder,load_arrays
@struct.dataclass
class C01Observation(model_lib.Observation):
    effect_target: object=None
    raw_state: object=None
    target_actions: object=None
    action_valid: object=None
class G1DC01Pi(Pi0):
    def __init__(self,config,rngs):
        super().__init__(config,rngs);width=gemma.get_config(config.action_expert_variant).width
        self.action_dim=256;self.action_in_proj=nnx.Linear(256,width,rngs=rngs);self.action_out_proj=nnx.Linear(width,256,rngs=rngs)
        arrays=load_arrays(config.decoder_init);self.decoder=ActionDecoder(arrays,16)
        if self.decoder.state_mean.value.shape!=(16,):raise ValueError('Native16 raw state statistics required')
        self.effect_mean=nnx.Variable(jnp.asarray(arrays['effect_mean']));self.effect_std=nnx.Variable(jnp.asarray(arrays['effect_std']))
        self.gtaux_weight=config.gtaux_weight;self.delta_weight=config.delta_weight;self.ramp_steps=config.ramp_steps
    def losses(self,rng,obs,step,train=False,calibration=False):
        # Capture all training-only fields before normal image preprocessing.
        target=jax.lax.stop_gradient(obs.effect_target);raw=obs.raw_state;actions=obs.target_actions;valid=obs.action_valid
        if target.shape[-2:]!=(30,256) or raw.shape[-1]!=16 or actions.shape[-2:]!=(30,16):raise ValueError('C01 G1D E/state/action dimensions differ from contract')
        pre,noise_key,time_key=jax.random.split(rng,3);o=model_lib.preprocess_observation(pre,obs,train=train)
        noise=jax.random.normal(noise_key,target.shape);time=jax.random.beta(time_key,1.5,1,target.shape[:-2])*.999+.001;t=time[:,None,None]
        noisy=t*noise+(1-t)*target;truth=noise-target
        prefix,pm,pa=self.embed_prefix(o);suffix,sm,sa,condition=self.embed_suffix(o,noisy,time)
        mask=jnp.concatenate([pm,sm],1);ar=jnp.concatenate([pa,sa]);(_,out),_=self.PaliGemma.llm([prefix,suffix],mask=make_attn_mask(mask,ar),positions=jnp.cumsum(mask,1)-1,adarms_cond=[None,condition])
        velocity=self.action_out_proj(out[:,-30:]).astype(jnp.float32)
        mask_e=jnp.broadcast_to(valid[...,None],velocity.shape);fm=jnp.sum(jnp.where(mask_e,jnp.square(velocity-truth),0))/jnp.maximum(mask_e.sum(),1)
        clean=noisy-t*velocity;physical=clean*self.effect_std.value+self.effect_mean.value;prediction=self.decoder(physical,raw)
        target_a=jax.lax.stop_gradient((actions-self.decoder.action_mean.value)/self.decoder.action_std.value)
        def huber(error):
            x=jnp.abs(error);return jnp.where(x<1,.5*error**2,x-.5)
        def per_sample(values,mask):
            mask=jnp.broadcast_to(mask[...,None],values.shape);return jnp.sum(jnp.where(mask,values,0),(1,2))/jnp.maximum(mask.sum((1,2)),1)
        gt=jnp.stack([per_sample(huber(prediction[...,lo:hi]-target_a[...,lo:hi]),valid) for lo,hi in ((0,7),(7,14),(14,15),(15,16))]).mean(0)
        dp=jnp.diff(prediction[...,14:16],axis=1);dt=jnp.diff(target_a[...,14:16],axis=1);dv=valid[:,1:]&valid[:,:-1];dm=jnp.broadcast_to(dv[...,None],dt.shape)
        scale=jax.lax.stop_gradient(jnp.maximum(jnp.sqrt(jnp.sum(jnp.where(dm,dt**2,0),(0,1),keepdims=True)/jnp.maximum(dm.sum((0,1),keepdims=True),1)+1e-8),.1))
        delta=per_sample(huber((dp-dt)/scale),dv);prob=((.7-.001)/.999)**1.5-((.2-.001)/.999)**1.5;weight=((time>=.2)&(time<=.7))/prob
        ramp=jnp.clip(step/self.ramp_steps,0,1);aux=ramp*jnp.mean(weight*(self.gtaux_weight*gt+self.delta_weight*delta))
        metrics={'loss':fm+aux,'effect_loss':fm,'gtaux':gt.mean(),'gripper_delta':delta.mean(),'ramp':ramp}
        if calibration:
            def gt_objective(value):
                pred=self.decoder(value*self.effect_std.value+self.effect_mean.value,raw)
                groups=jnp.stack([per_sample(huber(pred[...,lo:hi]-target_a[...,lo:hi]),valid) for lo,hi in ((0,7),(7,14),(14,15),(15,16))]).mean(0)
                return jnp.mean(weight*groups)
            grad=jax.grad(gt_objective)(clean)
            fm_grad=2*jnp.where(mask_e,velocity-truth,0)/jnp.maximum(mask_e.sum(),1)
            metrics['gtaux_unweighted_output_gradient_ratio']=jnp.linalg.norm(t*grad)/jnp.maximum(jnp.linalg.norm(fm_grad),1e-12)
        return fm+aux,metrics
    def compute_loss(self,rng,observation,actions,train=False):
        del actions;loss,_=self.losses(rng,observation,jnp.asarray(self.ramp_steps),train);return jnp.broadcast_to(loss,observation.effect_target.shape[:-1])
@dataclasses.dataclass(frozen=True)
class G1DC01PiConfig(Pi0Config):
    decoder_init: str=''
    gtaux_weight: float=0.25628781345139295
    delta_weight: float=.05
    ramp_steps: int=10000
    def create(self,rng):return G1DC01Pi(self,nnx.Rngs(rng))
    def inputs_spec(self,batch_size=1):
        obs,actions=super().inputs_spec(batch_size=batch_size)
        with at.disable_typechecking():
            obs=C01Observation(**{f.name:getattr(obs,f.name) for f in dataclasses.fields(obs)},effect_target=jax.ShapeDtypeStruct((batch_size,30,256),jnp.float32),raw_state=jax.ShapeDtypeStruct((batch_size,16),jnp.float32),target_actions=jax.ShapeDtypeStruct((batch_size,30,16),jnp.float32),action_valid=jax.ShapeDtypeStruct((batch_size,30),jnp.bool_))
        return obs,actions
@dataclasses.dataclass(frozen=True)
class G1DC01Weights:
    params_path: str
    def load(self,params):
        loaded=model_lib.restore_params(self.params_path,restore_type=np.ndarray)
        loaded={k:v for k,v in loaded.items() if k not in ('action_in_proj','action_out_proj')}
        return weight_loaders._merge_params(loaded,params,missing_regex=r'.*(lora|action_in_proj|action_out_proj|decoder|effect_mean|effect_std).*')
