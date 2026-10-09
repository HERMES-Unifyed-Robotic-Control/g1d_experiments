"""Copy manifest assets with SHA256 validation and live progress heartbeats."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import threading
import time
import urllib.request
import urllib.parse

base, dest = sys.argv[1].rstrip('/'), Path(sys.argv[2]).resolve()
dest.mkdir(parents=True, exist_ok=True)
manifest = json.load(urllib.request.urlopen(base + '/manifest.json', timeout=60))
assert manifest['complete']
(dest.parent / 'ASSET_SOURCE_MANIFEST.json').write_text(json.dumps(manifest))
lock = threading.RLock()
active = {}
status = dict(state='running', files=0, bytes=0, total_files=len(manifest['files']), total_bytes=manifest['total_bytes'])
stop = threading.Event()

def write_status():
    with lock:
        snapshot = dict(status, updated_unix=time.time(), active_files=list(active.values()),
                        inflight_bytes=sum(row['downloaded_bytes'] for row in active.values()))
        snapshot['written_bytes_including_unverified'] = snapshot['bytes'] + snapshot['inflight_bytes']
        temp = dest.parent / 'ASSET_SYNC_STATUS.tmp.json'
        temp.write_text(json.dumps(snapshot, indent=2)+'\n')
        temp.replace(dest.parent / 'ASSET_SYNC_STATUS.json')
        print(json.dumps(snapshot), flush=True)

def heartbeat():
    while not stop.wait(5):
        write_status()

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8*1024*1024):
            h.update(block)
    return h.hexdigest()

def copy(row):
    relative = PurePosixPath(row['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe manifest path')
    path = dest.joinpath(*relative.parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.stat().st_size == row['size'] and digest(path) == row['sha256']:
            return row['size']
        raise ValueError('Existing asset differs: '+str(path))
    temp = path.with_name(path.name+'.transfer')
    try:
        for attempt in range(3):
            with lock:
                active[row['path']] = dict(path=row['path'], downloaded_bytes=0, total_bytes=row['size'], last_block_unix=time.time())
            try:
                h = hashlib.sha256(); count = 0
                with urllib.request.urlopen(base+'/'+urllib.parse.quote(row['path']), timeout=120) as source, temp.open('wb') as target:
                    while block := source.read(4*1024*1024):
                        target.write(block); h.update(block); count += len(block)
                        with lock:
                            active[row['path']].update(downloaded_bytes=count, last_block_unix=time.time())
                    target.flush(); os.fsync(target.fileno())
                if count != row['size'] or h.hexdigest() != row['sha256']:
                    raise ValueError('Asset hash mismatch: '+row['path'])
                temp.replace(path)
                return count
            except OSError as error:
                temp.unlink(missing_ok=True)
                if error.errno in (28,122) or attempt == 2:
                    raise
                time.sleep(2)
    finally:
        with lock:
            active.pop(row['path'], None)

write_status()
thread = threading.Thread(target=heartbeat, daemon=True); thread.start()
pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
try:
    futures = [pool.submit(copy,row) for row in manifest['files']]
    for future in concurrent.futures.as_completed(futures):
        size = future.result()
        with lock:
            status['files'] += 1; status['bytes'] += size
    with lock:
        status['state'] = 'complete'
except BaseException as error:
    with lock:
        status.update(state='failed', error=repr(error))
    write_status()
    raise
finally:
    pool.shutdown(wait=True, cancel_futures=True)
    stop.set(); thread.join(); write_status()
