"""Fetch pinned artifacts; token resolved by Hugging Face login/env, never stored."""
from pathlib import Path
import argparse,json,hashlib,tarfile
from huggingface_hub import hf_hub_download,snapshot_download
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('artifacts'));p.add_argument('--baseline',action='store_true');p.add_argument('--afce-fullft',action='store_true');p.add_argument('--comparison',action='append');p.add_argument('--dataset',choices=['whole_197','subtasks_all','subtasks_train','subtasks_validation'],action='append');a=p.parse_args()
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
 e=m['baseline'];prefix=e.get('path','7500')
 root=Path(snapshot_download(e.get('repo',m['model_repo']),revision=e['revision'],allow_patterns=[prefix+'/**']))
 dest=a.output/'baseline'/'7500';dest.parent.mkdir(parents=True,exist_ok=True)
 if not dest.exists():dest.symlink_to(root/prefix,target_is_directory=True)
 elif dest.resolve()!=(root/prefix).resolve():raise RuntimeError('Existing baseline directory differs; choose a fresh --output')
if a.afce_fullft:
 e=m['afce_fullft']
 if e.get('phase')!='complete' or not e.get('revision'):raise RuntimeError('AFCE upload not yet verified complete; update code after publication')
 root=Path(snapshot_download(e['repo'],revision=e['revision'],allow_patterns=['afce_fullft/**']))
 dest=a.output/'afce_fullft'
 if not dest.exists():dest.symlink_to(root/'afce_fullft',target_is_directory=True)
 elif dest.resolve()!=(root/'afce_fullft').resolve():raise RuntimeError('Existing AFCE directory differs; choose a fresh --output')

for name in a.comparison or []:
 e=m["comparisons"].get(name)
 if not e or not e.get("revision"):raise RuntimeError("Comparison not yet published: "+name)
 snapshot_download(e.get("repo",m.get("legacy_model_repo",m["model_repo"])),revision=e["revision"],allow_patterns=[e["path"]+"/**"],local_dir=a.output)
