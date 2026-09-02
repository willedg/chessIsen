# wp4/replay_buffer.py
"""
Simple replay buffer that scans shard files and yields (state,pi,z) batches.
It does not implement sampling prioritization—keeps it simple.
"""

import os
import numpy as np
from wp4.utils_io import list_shards, load_shard
import random

class ReplayBuffer:
    def __init__(self, shard_dir: str, max_shards=None):
        self.shard_dir = shard_dir
        self.max_shards = max_shards

    def list_shards(self):
        shards = list_shards(self.shard_dir)
        if self.max_shards:
            return shards[-self.max_shards:]
        return shards

    def sample_examples(self):
        shards = self.list_shards()
        random.shuffle(shards)
        for s in shards:
            states, pis, zs = load_shard(s)
            # yield each example individually (or convert to batches in trainer)
            for i in range(states.shape[0]):
                yield states[i], pis[i], zs[i]
