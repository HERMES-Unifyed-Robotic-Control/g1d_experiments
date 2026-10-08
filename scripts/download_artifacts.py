"""Fetch pinned artifacts; token resolved by Hugging Face login/env, never stored."""
from pathlib import Path
import argparse,json,hashlib,tarfile
from huggingface_hub import hf_hub_download,snapshot_download
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('artifacts'));p.add_argument('--baseline',action='store_true');p.add_argument('--dataset',choices=['whole_197','subtasks_all','subtasks_train','subtasks_validation'],action='append');a=p.parse_args()
m=json.loads((Path(__file__).parents[1]/'ARTIFACTS.json').read_text());a.output.mkdir(exist_ok=True,parents=True)
for name in a.dataset or []:
 e=m['datasets'][name]
 if not e.get('revision'):raise RuntimeError(name+' upload pending; update repository after upload completion')
 f=Path(hf_hub_download(m['dataset_repo'],e['path'],repo_type='dataset',revision=e['revision']))
 h=hashlib.sha256()
 with f.open('rb') as stream:
  for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
 if h.hexdigest()!=e['sha256']:raise RuntimeError('Checksum mismatch '+name)
 with tarfile.open(f) as tf:tf.extractall(a.output/'data',filter='data')
if a.baseline:
 if not m['baseline'].get('revision'):raise RuntimeError('Baseline upload pending; update repository after completion')
 snapshot_download(m['model_repo'],revision=m['baseline']['revision'],allow_patterns=['7500/**'],local_dir=a.output/'baseline')
