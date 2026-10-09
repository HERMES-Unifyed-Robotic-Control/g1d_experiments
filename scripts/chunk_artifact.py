#!/usr/bin/env python3
"""Store artifacts as small parts and restore them with SHA256 verification."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def part_path(directory, index):
    return directory / f"part_{index:08d}.bin"


def atomic_write(path, data):
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def pack(source, directory, chunk_size):
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old["chunk_bytes"] != chunk_size or old["name"] != source.name:
            raise ValueError("Existing manifest belongs to a different artifact/layout")
    digest = hashlib.sha256()
    count = size = 0
    with source.open("rb") as stream:
        before = os.fstat(stream.fileno())
        while data := stream.read(chunk_size):
            destination = part_path(directory, count)
            if destination.exists():
                if destination.read_bytes() != data:
                    raise ValueError(f"Existing part differs: {destination}; use a new directory")
            else:
                atomic_write(destination, data)
            digest.update(data)
            size += len(data)
            count += 1
            if count % 256 == 0:
                print(f"Packed {count} parts, {size} bytes", flush=True)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_size, after.st_mtime_ns, after.st_ctime_ns
    ) or size != before.st_size:
        raise ValueError("Source changed while packing; use a stable artifact")
    manifest = dict(version=1, name=source.name, size_bytes=size,
                    chunk_bytes=chunk_size, parts=count, sha256=digest.hexdigest())
    if manifest_path.exists() and old != manifest:
        raise ValueError("Existing manifest differs; use a new directory")
    atomic_write(manifest_path, (json.dumps(manifest, indent=2) + "\n").encode())
    print(json.dumps(manifest), flush=True)


def unpack(directory, destination):
    manifest = json.loads((directory / "manifest.json").read_text())
    size, chunk, count = (manifest[k] for k in ("size_bytes", "chunk_bytes", "parts"))
    if manifest["version"] != 1 or not all(type(x) is int for x in (size, chunk, count)):
        raise ValueError("Invalid manifest")
    if size < 0 or chunk <= 0 or count != (size + chunk - 1) // chunk:
        raise ValueError("Invalid size/part count")
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".assembling")
    if temporary.exists():
        raise FileExistsError(temporary)
    digest = hashlib.sha256()
    try:
        with temporary.open("xb") as stream:
            for index in range(count):
                data = part_path(directory, index).read_bytes()
                if len(data) != min(chunk, size - index * chunk):
                    raise ValueError(f"Unexpected part size: {index}")
                digest.update(data)
                stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if digest.hexdigest() != manifest["sha256"]:
            raise ValueError("SHA256 mismatch")
        # Publish without overwriting a file another process may have created.
        os.link(temporary, destination)
        print(f"Restored {destination}, SHA256 verified", flush=True)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    packing = sub.add_parser("pack")
    packing.add_argument("source", type=Path)
    packing.add_argument("directory", type=Path)
    packing.add_argument("--chunk-kib", type=int, default=64)
    restoring = sub.add_parser("unpack")
    restoring.add_argument("directory", type=Path)
    restoring.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.command == "pack":
        pack(args.source, args.directory, args.chunk_kib * 1024)
    else:
        unpack(args.directory, args.destination)


if __name__ == "__main__":
    main()
