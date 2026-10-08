from pathlib import Path
import json,urllib.request,urllib.parse,hashlib,base64,google_crc32c,shutil
r=Path('/mnt/data/yangky/test/Psi0/cache/checkpoints/openpi/pi05_base_jax');objects=json.loads((r/'download_manifest.json').read_text());results=[]
for o in objects:
 p=r/o['key'].removeprefix('checkpoints/pi05_base/');part=p.with_suffix(p.suffix+'.part')
 if not p.exists():
  pieces=sorted(p.parent.glob(p.name+'.range_*'),key=lambda x:int(x.name.split('_')[-2]))
  if part.stat().st_size!=o['size']:
   with part.open('ab') as f:
    for piece in pieces:
     a,b=map(int,piece.name.split('_')[-2:]);assert f.tell()==a
     with piece.open('rb') as src:shutil.copyfileobj(src,f,8<<20)
  source=part
 else:source=p
 assert source.stat().st_size==o['size']
 meta=json.load(urllib.request.urlopen('https://storage.googleapis.com/storage/v1/b/openpi-assets/o/'+urllib.parse.quote(o['key'],safe='')))
 crc=google_crc32c.Checksum();md5=hashlib.md5()
 with source.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):crc.update(b);md5.update(b)
 assert base64.b64encode(crc.digest()).decode()==meta['crc32c'],p
 if 'md5Hash' in meta:assert base64.b64encode(md5.digest()).decode()==meta['md5Hash'],p
 if source!=p:source.replace(p)
 results.append(dict(key=o['key'],size=o['size'],crc32c=meta['crc32c'],local_md5=md5.hexdigest(),generation=meta['generation']))
 print('VERIFIED',p.name,flush=True)
(r/'verified_manifest.json').write_text(json.dumps(results,indent=2))
(r/'DOWNLOAD_COMPLETE').write_text('All 20 official GCS objects verified by size and CRC32C; MD5 additionally verified where supplied.\n')
print('DOWNLOAD_COMPLETE',flush=True)
