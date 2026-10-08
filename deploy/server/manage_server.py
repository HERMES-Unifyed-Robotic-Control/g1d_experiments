"""Switch the GPU0 alignment server between baseline and QC; never controls the robot."""
from pathlib import Path
import argparse
import os
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parent
SCRIPT=ROOT/'serve_aligned.py'
GPU={'baseline':'0','qc':'0','joint197':'1'}

def live_pid(variant):
    path=ROOT/f'server_{variant}.pid'
    if not path.exists():return None
    pid=int(path.read_text())
    try:
        args=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        if str(SCRIPT) in args and f'--variant {variant}' in args:return pid
    except FileNotFoundError:pass
    return None

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('status','switch','restart'))
    parser.add_argument('variant',choices=('baseline','qc','joint197'),nargs='?',default='baseline')
    args=parser.parse_args()
    if args.action=='status':
        for variant in GPU:print(variant,live_pid(variant))
        return
    busy=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','unitree@192.168.10.104',
                         "pgrep -af '[d]eploy_g1d_food_sorting_pi05.py|[t]eleop_hand_and_arm.py'"],capture_output=True,text=True)
    if busy.returncode not in (0,1) or busy.stdout.strip():
        raise RuntimeError('Cannot switch while robot client is active or state is unavailable: '+busy.stdout+busy.stderr)
    current=live_pid(args.variant)
    if current is not None and args.action=='switch':print(f'{args.variant} already running: {current}');return
    for variant in GPU:
        if GPU[variant] != GPU[args.variant]:continue
        pid=live_pid(variant)
        if pid is None:continue
        os.kill(pid,signal.SIGTERM)
        for _ in range(150):
            if live_pid(variant) is None:break
            time.sleep(.1)
        else:raise RuntimeError(f'{variant} did not stop cleanly')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=GPU[args.variant],PSI_HOME='/mnt/data/yangky/test/Psi0',DATA_HOME='/mnt/data/yangky/test/datasets')
    log_path=ROOT/f'server_{args.variant}.log'
    with log_path.open('w') as log:
        proc=subprocess.Popen(['/mnt/data/yangky/test/Psi0/.venv-openpi/bin/python',str(SCRIPT),'--variant',args.variant],
                               cwd='/mnt/data2/yangky/test/g1d_client_alignment_20261007',env=env,
                               stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True)
    (ROOT/f'server_{args.variant}.pid').write_text(str(proc.pid)+'\n')
    print(f'started {args.variant} pid={proc.pid} log={log_path}',flush=True)
    for _ in range(180):
        if proc.poll() is not None:raise RuntimeError(f'Server exited: {log_path.read_text()[-3000:]}')
        if f'READY aligned variant={args.variant}' in log_path.read_text():print('READY');return
        time.sleep(1)
    raise RuntimeError(f'Server not yet ready; inspect {log_path}')

if __name__=='__main__':main()
