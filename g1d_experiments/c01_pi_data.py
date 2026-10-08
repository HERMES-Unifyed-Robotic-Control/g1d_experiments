"""Same native16/quantile/padded32 state input for G1D pi train and inference."""
from pathlib import Path
import json
import numpy as np
import cv2
from .c01_contract import native_state,read_episodes
from .c01_dataset import VIEWS
class G1DPiData:
    def __init__(self,root,cache,split,norm_stats):
        self.root=Path(root);self.cache=Path(cache);self.split=split
        self.info,self.episodes=read_episodes(root);self.ends=np.cumsum([e['length'] for e in self.episodes]);self.effects={}
        self.manifest=json.loads((self.cache/'manifest.json').read_text())
        if not self.manifest.get('complete') or self.manifest['state_dim']!=16 or self.manifest['action_dim']!=16:raise ValueError('Complete native16 E cache required')
        record=self.manifest['splits'][split]
        if record['frames']!=int(self.ends[-1]) or [(e['episode_index'],e['source_episode_index'],e['length']) for e in record['episodes']]!=[(e['episode_index'],e['source_episode_index'],e['length']) for e in self.episodes]:raise ValueError('E cache episode/frame alignment mismatch')
        self.tasks={e['task_index']:e['task'] for e in map(json.loads,(self.root/'meta/tasks.jsonl').read_text().splitlines())}
        with np.load(self.cache/'normalization.npz') as z:self.mean=z['mean'];self.std=z['std']
        from openpi.shared import normalize
        self.norm=normalize.load(Path(norm_stats).parent)['state']
        from openpi.models.tokenizer import PaligemmaTokenizer
        self.tokenizer=PaligemmaTokenizer(200)
    def __len__(self):return int(self.ends[-1])
    def inputs(self,rgb,state,prompt):
        from openpi_client.image_tools import resize_with_pad
        state=native_state(state)
        q=(state-self.norm.q01)/(self.norm.q99-self.norm.q01+1e-6)*2-1
        padded=np.pad(q,(0,16)).astype(np.float32);tokens,mask=self.tokenizer.tokenize(prompt,q)
        keys=('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')
        images={k:resize_with_pad(np.asarray(im),224,224).astype(np.float32)/255*2-1 for k,im in zip(keys,rgb)}
        return dict(images=images,image_masks={k:np.True_ for k in keys},state=padded,tokenized_prompt=tokens.astype(np.int32),tokenized_prompt_mask=mask)
    def sample(self,index):
        if not 0<=index<len(self):raise IndexError(index)
        ei=int(np.searchsorted(self.ends,index,side='right'));t=index-(int(self.ends[ei-1]) if ei else 0);ep=self.episodes[ei];e=ep['episode_index'];n=ep['length'];rgb=[]
        for view in VIEWS:
            path=self.root/self.info['video_path'].format(episode_index=e,episode_chunk=e//self.info.get('chunks_size',1000),video_key='observation.images.'+view)
            cap=cv2.VideoCapture(str(path))
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES,t);ok,bgr=cap.read()
                if not ok:raise RuntimeError(f'Missing policy frame:{e}/{t}/{view}')
                rgb.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
            finally:cap.release()
        if e not in self.effects:self.effects[e]=np.load(self.cache/self.split/f'episode_{e:06d}.npy',mmap_mode='r')
        effect=self.effects[e][t];raw=ep['states'][t];ids=t+np.arange(30);actions=ep['actions'][np.minimum(ids,n-1)]
        out=self.inputs(rgb,raw,self.tasks[ep['task_index']]);out.update(effect_target=(effect-self.mean)/self.std,raw_state=raw,target_actions=actions,action_valid=ids<n)
        return out
    def batch(self,indices):
        import jax
        from .c01_pi import C01Observation
        samples=[self.sample(int(i)) for i in indices];d=jax.tree.map(lambda *x:np.stack(x),*samples)
        return C01Observation(**d),np.zeros((len(samples),30,32),np.float32)
