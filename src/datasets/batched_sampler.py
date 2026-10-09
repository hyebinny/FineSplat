import torch
import numpy as np


def round_by(total, multiple, up=False):
    """Round total to the nearest multiple; round up if up=True."""
    if up:
        total = total + multiple - 1
    return (total // multiple) * multiple


class DynamicBatchedMultiFeatureRandomSampler:
    """
    Random sampler with dynamic batch sizes across multiple feature pools.

    Each batch shares the same randomly-chosen feature indices. Batch size is
    determined by a mapping from the scaling feature value to a batch size,
    allowing GPU memory to be managed across different configurations (e.g.
    number of input views).

    Yields batches of tuples: [(sample_idx, feat_idx_0, feat_idx_1, ...), ...]
    """

    def __init__(
        self,
        dataset,
        pool_sizes,
        scaling_feature_idx=0,
        feature_to_batch_size_map=None,
        world_size=1,
        rank=0,
        drop_last=True,
    ):
        """
        Args:
            dataset: The dataset to sample from; only len() is called.
            pool_sizes: List of pool sizes, one per feature dimension.
            scaling_feature_idx: Index into pool_sizes whose value determines batch size.
            feature_to_batch_size_map: Dict or callable mapping scaling feature value
                to per-GPU batch size.
            world_size: Total number of distributed processes.
            rank: This process's distributed rank.
            drop_last: Whether to drop the final incomplete batch.
        """
        self.pool_sizes = pool_sizes if isinstance(pool_sizes, list) else [pool_sizes]
        self.scaling_feature_idx = scaling_feature_idx

        if not (0 <= scaling_feature_idx < len(self.pool_sizes)):
            raise ValueError(
                f"scaling_feature_idx must be between 0 and {len(self.pool_sizes) - 1}"
            )

        self.feature_to_batch_size_map = feature_to_batch_size_map
        self.total_size = len(dataset)
        self.world_size = world_size
        self.rank = rank
        self.epoch = None
        self.drop_last = drop_last

    def __len__(self):
        """Lower-bound estimate of batches per process, using the largest batch size."""
        if callable(self.feature_to_batch_size_map):
            batch_sizes = [
                self.feature_to_batch_size_map(i)
                for i in range(self.pool_sizes[self.scaling_feature_idx])
            ]
            max_batch_size = max(batch_sizes)
        else:
            max_batch_size = max(self.feature_to_batch_size_map.values())

        max_batch_size = max(1, max_batch_size)
        total_batches = self.total_size // max_batch_size
        if not self.drop_last and self.total_size % max_batch_size > 0:
            total_batches += 1
        return total_batches // self.world_size

    def set_epoch(self, epoch):
        """Set the epoch index so each epoch uses a distinct RNG seed.

        Args:
            epoch: The current training epoch index.
        """
        self.epoch = epoch

    def __iter__(self):
        """Yield batches of index tuples assigned to this rank.

        Each batch contains tuples of (sample_idx, feat_idx_0, ...) where all
        tuples share the same randomly chosen feature indices.

        Yields:
            A list of tuples forming one batch for this rank.
        """
        # Derive a deterministic seed from the epoch so all ranks agree on
        # the shuffle order and feature assignments.
        if self.epoch is None:
            assert self.world_size == 1 and self.rank == 0, (
                "call set_epoch() before iterating in distributed mode"
            )
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        else:
            seed = self.epoch + 777
        rng = np.random.default_rng(seed=seed)

        sample_idxs = np.arange(self.total_size)
        rng.shuffle(sample_idxs)

        target_batches = len(self)
        batches_yielded = 0
        idx = 0
        batch_idx = 0

        while idx < len(sample_idxs) and batches_yielded < target_batches:
            feat_idxs = [rng.integers(pool_size) for pool_size in self.pool_sizes]
            scaling_feat = feat_idxs[self.scaling_feature_idx]

            if callable(self.feature_to_batch_size_map):
                batch_size = self.feature_to_batch_size_map(scaling_feat)
            else:
                batch_size = self.feature_to_batch_size_map.get(scaling_feat, 1)
            batch_size = max(1, batch_size)

            remaining = len(sample_idxs) - idx
            if remaining < batch_size:
                if self.drop_last:
                    break
                batch_size = remaining

            batch = [tuple([sample_idxs[idx + i]] + feat_idxs) for i in range(batch_size)]

            # Distribute batches round-robin so each rank receives an equal share.
            if batch_idx % self.world_size == self.rank:
                yield batch
                batches_yielded += 1

            batch_idx += 1
            idx += batch_size
