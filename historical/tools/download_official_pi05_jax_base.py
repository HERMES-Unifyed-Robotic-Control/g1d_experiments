from pathlib import Path
import urllib.request,xml.etree.ElementTree as E,json,hashlib,concurrent.futures,time
root=Path('/mnt/data/yangky/test/Psi0/cache/checkpoints/openpi/pi05_base_jax');root.mkdir(parents=True,exist_ok=True)
r=E.fromstring(urllib.request.urlopen('https://storage.googleapis.com/openpi-assets?prefix=checkpoints/pi05_base/params/').read());ns={'s':'http://doc.s3.amazonaws.com/2006-03-01'}
assert r.find('s:IsTruncated',ns).text=='false'
objects=[{'key':x.find('s:Key',ns).text,'size':int(x.find('s:Size',ns).text),'etag':x.find('s:ETag',ns).text.strip('"')} for x in r.findall('s:Contents',ns)]
(root/'download_manifest.json').write_text(json.dumps(objects,indent=2))
def fetch(o):
 p=root/o['key'].removeprefix('checkpoints/pi05_base/');p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists() and p.stat().st_size==o['size']:
  h=hashlib.md5()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(8<<20),b''):h.update(b)
  if h.hexdigest()==o['etag']:return str(p)+' cached verified'
 for attempt in range(3):
  try:
   h=hashlib.md5();part=p.with_suffix(p.suffix+'.part');count=part.stat().st_size if part.exists() else 0
   if count:
    with part.open('rb') as old:
     for b in iter(lambda:old.read(8<<20),b''):h.update(b)
   req=urllib.request.Request('https://storage.googleapis.com/openpi-assets/'+o['key'],headers={'Range':f'bytes={count}-'} if count else {})
   response=urllib.request.urlopen(req,timeout=120)
   if count and response.status!=206:count=0;h=hashlib.md5()
   with response,part.open('ab' if count else 'wb') as f:
    while b:=response.read(8<<20):f.write(b);h.update(b);count+=len(b)
   assert count==o['size'] and h.hexdigest()==o['etag'],(count,o['size'],h.hexdigest(),o['etag'])
   p.with_suffix(p.suffix+'.part').replace(p)
   return f'{p.name}: {count/1e6:.1f} MB verified'
  except Exception:
   if attempt==2:raise
   time.sleep(2)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 for r in pool.map(fetch,objects):print(r,flush=True)
(root/'DOWNLOAD_COMPLETE').write_text('All 20 official GCS objects verified by size and MD5.\n')
print('DOWNLOAD_COMPLETE',flush=True)
