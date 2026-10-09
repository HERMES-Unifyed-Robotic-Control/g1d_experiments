"""Follow C01 JSONL metrics without importing JAX, Torch or touching training."""
import argparse
import json
import math
from pathlib import Path
import time

from tensorboard.compat.proto.event_pb2 import Event, SessionLog
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-root', type=Path, required=True)
parser.add_argument('--logdir', type=Path, required=True)
parser.add_argument('--stages', nargs='+', default=['E', 'pi_lora'])
parser.add_argument('--interval', type=float, default=10)
args = parser.parse_args()
streams = {}

def emit(writer, record, split):
    values = [Summary.Value(tag=f'{split}/{key}', simple_value=float(value))
              for key, value in record.items()
              if key != 'step' and isinstance(value, (int, float))
              and not isinstance(value, bool) and math.isfinite(value)]
    writer.add_event(Event(wall_time=time.time(), step=int(record['step']),
                           summary=Summary(value=values)))

try:
    while True:
        for stage in args.stages:
            for split, filename in [('train', 'training_metrics.jsonl'),
                                    ('validation', 'validation_metrics.jsonl')]:
                path = args.run_root / stage / filename
                if not path.exists():
                    continue
                if path not in streams:
                    writer = EventFileWriter(str(args.logdir / stage / split))
                    # A resumed run may contain duplicate steps. Keep the last
                    # record for each step when backfilling history.
                    records = {}
                    offset = 0
                    with path.open('rb') as source:
                        for line in source:
                            if not line.endswith(b'\n'):
                                break
                            offset += len(line)
                            try:
                                row = json.loads(line)
                                records[int(row['step'])] = row
                            except (ValueError, KeyError):
                                continue
                    for step in sorted(records):
                        emit(writer, records[step], split)
                    writer.flush()
                    streams[path] = [writer, offset, max(records, default=-1)]
                    print(f'Backfilled {path}: {len(records)} steps', flush=True)
                writer, offset, last_step = streams[path]
                if path.stat().st_size < offset:
                    offset, last_step = 0, -1
                    writer.add_event(Event(wall_time=time.time(), step=0,
                                           session_log=SessionLog(status=SessionLog.START)))
                with path.open('rb') as source:
                    source.seek(offset)
                    for line in source:
                        if not line.endswith(b'\n'):
                            break
                        offset += len(line)
                        try:
                            row = json.loads(line)
                            step = int(row['step'])
                        except (ValueError, KeyError):
                            continue
                        if step <= last_step:
                            writer.add_event(Event(wall_time=time.time(), step=step,
                                                   session_log=SessionLog(status=SessionLog.START)))
                        emit(writer, row, split)
                        last_step = step
                writer.flush()
                streams[path] = [writer, offset, last_step]
        time.sleep(args.interval)
finally:
    for writer, _, _ in streams.values():
        writer.close()
