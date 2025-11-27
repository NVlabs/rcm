#!/usr/bin/env python3
"""
Merge many small TAR shards into new larger shards.
Input: a directory containing shards like shard-00000.tar, shard-00001.tar, ...
Output: new shards with target_shard_size samples each.

Usage:
    python merge_tar_shards.py \
        --input_dir path/to/small_shards \
        --output_dir path/to/large_shards \
        --target_shard_size 5000

Each sample is assumed to be a group of tar members with common prefix, e.g.
  000000.jpg
  000000.txt
  000001.jpg
  000001.txt
This script groups files by prefix before writing.
"""

import os
import tarfile
import argparse
from collections import defaultdict


def read_samples_from_tar(tar_path):
    """Yield samples as {key: {filename: bytes}}.
    Group consecutive members by prefix before '.'
    """
    samples = defaultdict(dict)
    with tarfile.open(tar_path, "r") as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            name = os.path.basename(m.name)
            prefix = name.split(".")[0]
            f = tar.extractfile(m)
            if f is None:
                continue
            samples[prefix][name] = f.read()
    for key, files in samples.items():
        yield key, files


def write_shard(samples, out_path):
    """Write a list of samples to one TAR shard.
    samples: list of (key, {filename: bytes})
    """
    with tarfile.open(out_path, "w") as tar:
        for key, file_dict in samples:
            for fname, data in file_dict.items():
                info = tarfile.TarInfo(name=f"{key}/{fname}")
                info.size = len(data)
                tar.addfile(info, fileobj=BytesIO(data))


from io import BytesIO

def merge_shards(input_dir, output_dir, target_shard_size):
    os.makedirs(output_dir, exist_ok=True)
    shard_paths = sorted(
        [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".tar")]
    )

    current_samples = []
    new_shard_index = 0

    for tar_path in shard_paths:
        for key, files in read_samples_from_tar(tar_path):
            current_samples.append((key, files))
            if len(current_samples) >= target_shard_size:
                out_path = os.path.join(output_dir, f"shard-{new_shard_index:05d}.tar")
                write_shard(current_samples, out_path)
                print(f"Wrote {out_path}, {len(current_samples)} samples")
                new_shard_index += 1
                current_samples = []

    # final remainder
    if current_samples:
        out_path = os.path.join(output_dir, f"shard-{new_shard_index:05d}.tar")
        write_shard(current_samples, out_path)
        print(f"Wrote {out_path}, {len(current_samples)} samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_shard_size", type=int, required=True)
    args = parser.parse_args()

    merge_shards(args.input_dir, args.output_dir, args.target_shard_size)
