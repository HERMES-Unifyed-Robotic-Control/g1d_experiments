"""JAX helpers for C01 data parallelism across one-GPU job instances."""
import argparse
import dataclasses
import json
from pathlib import Path

import jax
import numpy as np
from jax.experimental import multihost_utils


def arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--num-processes', type=int, default=1)
    parser.add_argument('--process-id', type=int, default=0)
    parser.add_argument('--coordinator-address')
    return parser


def initialize(parser):
    args, _ = parser.parse_known_args()
    if args.num_processes < 1 or not 0 <= args.process_id < args.num_processes:
        raise ValueError('Invalid distributed process count or ID')
    if args.num_processes > 1:
        if not args.coordinator_address:
            raise ValueError('Distributed training requires --coordinator-address')
        jax.distributed.initialize(coordinator_address=args.coordinator_address,
                                   num_processes=args.num_processes,
                                   process_id=args.process_id, local_device_ids=[0],
                                   initialization_timeout=180)


def synchronize(name):
    if jax.process_count() > 1:
        multihost_utils.sync_global_devices(name)


def replicate(value, sharding):
    """Explicitly build global replicated arrays from identical host values."""
    return jax.make_array_from_callback(value.shape, sharding, lambda index: value[index])


@dataclasses.dataclass(frozen=True)
class ReplicatedWeights:
    loader: object
    sharding: object

    def load(self, params):
        loaded = self.loader.load(params)
        return jax.tree.map(lambda value: replicate(value, self.sharding)
                            if isinstance(value, np.ndarray) else value, loaded)


def key(seed, sharding):
    raw = np.asarray(jax.random.key_data(jax.random.key(seed)))
    return jax.random.wrap_key_data(replicate(raw, sharding))


def batch(dataset, global_indices, sharding):
    indices = np.asarray(global_indices)
    if len(indices) % jax.process_count():
        raise ValueError('Global batch must divide evenly between processes')
    if jax.process_count() == 1:
        return jax.device_put(dataset.batch(indices), sharding)
    local = np.split(indices, jax.process_count())[jax.process_index()]
    return jax.tree.map(lambda value: jax.make_array_from_process_local_data(sharding, value),
                        dataset.batch(local))


def checkpoint_manager(checkpoints, output, resume):
    if jax.process_count() == 1:
        return checkpoints.initialize_checkpoint_dir(output, keep_period=2500,
                                                     overwrite=False, resume=resume)
    output = Path(output)
    if jax.process_index() == 0:
        if output.exists() and not resume:
            raise FileExistsError('Checkpoint directory already exists; use --resume')
        output.mkdir(parents=True, exist_ok=True)
    synchronize('c01_checkpoint_directory')
    # The directory was created once by rank 0. All Orbax managers must join
    # save/restore collectives; an empty directory does not imply a saved state.
    return checkpoints.initialize_checkpoint_dir(output, keep_period=2500,
                                                 overwrite=False, resume=True)


def write_json(path, value):
    if jax.process_index() == 0:
        Path(path).write_text(json.dumps(value, indent=2) + '\n')


def append_json(path, value):
    if jax.process_index() == 0:
        with Path(path).open('a') as stream:
            stream.write(json.dumps(value) + '\n')
