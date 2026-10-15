"""Control-side logging and scheduling; imports never initialize robot hardware."""
import hashlib
import json
from pathlib import Path
import queue
import threading
import time
import numpy as np

REVISION = 'no_static_blend_async_recording_deadline_v2_20261013'

def check_compatibility(client_path, metadata):
    """Pin the unchanged inference service and separately verify this control revision.

    A model service cannot attest the bytes of a newer control-only client. Do not
    represent its old client hash as the running client's hash: record both.
    """
    root = Path(client_path).parent
    spec = json.loads((root / 'control_compatibility.json').read_text())
    if spec['control_revision'] != REVISION:
        raise ValueError('Unknown control compatibility revision')
    for name, expected in spec['control_files_sha256'].items():
        if Path(name).name != name or hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Control file hash mismatch: '+name)
    for key, expected in spec['upstream_contract'].items():
        if metadata.get(key) != expected:
            raise ValueError('Inference contract mismatch: '+key)
    return spec

class DeadlineClock:
    """Monotonic waypoint start deadlines; never burst to recover a missed tick."""
    def __init__(self, period, clock=time.perf_counter, sleep=time.sleep):
        self.period = period; self.clock = clock; self.sleep = sleep
        self.deadline = None
    def reset(self):
        self.deadline = None
    def wait(self):
        now = self.clock()
        if self.deadline is None:
            self.deadline = now
        if now < self.deadline:
            self.sleep(self.deadline-now)
        now = self.clock()
        late = max(0., now-self.deadline)
        # Advance from actual start after an overrun, rather than execute a burst.
        self.deadline = max(self.deadline, now)+self.period
        return late*1000.

class AsyncRecorderMixin:
    """Critical logs and video have separate bounded workers.

    Critical overload/errors fail explicitly. Video overload drops video frames
    with counts, so slow encoding does not stall control or hide data loss.
    """
    def __init__(self, *args, critical_capacity=4096, media_capacity=8, **kwargs):
        super().__init__(*args, **kwargs)
        self._closed_async = False
        self._record_error = None
        self._video_drops = 0
        self._critical_peak = 0; self._media_peak = 0
        self._io_stats = {'critical_count': 0, 'media_count': 0,
                          'critical_max_ms': 0., 'media_max_ms': 0.,
                          'critical_total_ms': 0., 'media_total_ms': 0.}
        if not self.enabled: return
        self._critical = queue.Queue(maxsize=critical_capacity)
        self._media = queue.Queue(maxsize=media_capacity)
        self._events_file = (self.dir/'events.jsonl').open('a', buffering=1)
        self._workers = [threading.Thread(target=self._consume, args=(q, kind), daemon=True,
                         name='record-'+kind) for q,kind in [(self._critical,'critical'),(self._media,'media')]]
        for worker in self._workers: worker.start()
    def _consume(self, q, kind):
        while True:
            item = q.get()
            try:
                if item is None: return
                fn, args = item; start = time.perf_counter()
                fn(*args)
                ms = (time.perf_counter()-start)*1000.
                self._io_stats[kind+'_count'] += 1
                self._io_stats[kind+'_total_ms'] += ms
                self._io_stats[kind+'_max_ms'] = max(ms, self._io_stats[kind+'_max_ms'])
            except Exception as exc:
                self._record_error = repr(exc)
            finally:
                q.task_done()
    def check_health(self):
        if self._record_error is not None:
            raise RuntimeError('Background recording failed: '+self._record_error)
    def _submit(self, fn, *args, media=False):
        if not self.enabled: return
        if self._closed_async: raise RuntimeError('Recorder is closed')
        self.check_health()
        q = self._media if media else self._critical
        try: q.put_nowait((fn,args))
        except queue.Full:
            if media:
                self._video_drops += 1
                return
            raise RuntimeError('Critical recording queue full; no silent log loss')
        if media: self._media_peak = max(self._media_peak,q.qsize())
        else: self._critical_peak = max(self._critical_peak,q.qsize())
    def _write_event(self, event):
        self._events_file.write(json.dumps(event)+'\n')
    def record_event(self, event):
        if self.enabled:
            self._submit(self._write_event, {'wall_time':time.time(),**event})
    def record_images(self, index, base, left, right):
        if self.enabled:
            self._submit(super().record_images,index,*(np.array(x,copy=True) for x in (base,left,right)),media=True)
    def record_policy_chunk(self,index,actions,state,prompt):
        if self.enabled:
            self._submit(super().record_policy_chunk,index,np.array(actions,copy=True),np.array(state,copy=True),str(prompt))
    def record_action(self,step,chunk_index,chunk_step,time_s,infer_ms,obs_state,policy_action,cmd_action,robot_state_before,robot_state_after):
        if not self.enabled: return
        row = dict(step=step,chunk_index=chunk_index,chunk_step=chunk_step,time_s=time_s,infer_ms=infer_ms)
        for name,a in zip(('obs_state','policy_action','cmd_action','robot_state_before','robot_state_after'),
                          (obs_state,policy_action,cmd_action,robot_state_before,robot_state_after)):
            a=np.asarray(a).reshape(-1)
            for i in range(16): row[f'{name}_{i:02d}']=float(a[i]) if i<a.size else float('nan')
        self._submit(self._write_row,row)
        # Snapshot on the control thread, not later on the disk worker.
        if self.telemetry is not None:
            state=self.telemetry.snapshot(); state.pop('arm_q',None)
            self.record_event({'type':'physical_state','step':step,'time_s':time_s,**state,
                'gravity_tau_nm':self.gravity.last_tau.tolist() if self.gravity is not None else [0.]*14})
    def _write_row(self,row):
        self.writer.writerow(row); self.rows.append(row)
    def close(self):
        if not self.enabled or self._closed_async: return
        self._closed_async=True
        # Draining occurs only after robot-control shutdown, never in a control tick.
        for q in (self._critical,self._media): q.put(None)
        for worker in self._workers: worker.join()
        summary={'type':'recording_summary','wall_time':time.time(),
            'video_chunks_dropped':self._video_drops,'critical_queue_peak':self._critical_peak,
            'media_queue_peak':self._media_peak,'background_error':self._record_error,**self._io_stats}
        self._write_event(summary);self._events_file.flush();self._events_file.close()
        (self.dir/'recording_summary.json').write_text(json.dumps(summary,indent=2))
        super().close()
        self.check_health()
