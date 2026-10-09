import importlib

import torch
import torch.distributed as dist

from src.datasets.dynamic_dataset import DynamicInputDataset


class DynamicBatchDatasetWrapper:
    """
    Adapts a dataset for use with DynamicBatchedMultiFeatureRandomSampler.

    The sampler yields pre-formed batches (lists of index tuples) instead of
    single indices, so DataLoader's default collation is bypassed and each
    batch is assembled here instead.
    """

    def __init__(self, dataset):
        """
        Args:
            dataset: The underlying dataset to wrap.
        """
        self.dataset = dataset

    def __getitem__(self, batch_indices):
        """Return a list of samples for a batch, or a single sample for a scalar index.

        Args:
            batch_indices: A list of index tuples (batch mode) or a single index.

        Returns:
            A list of dataset items when batch_indices is a list of tuples,
            or a single dataset item otherwise.
        """
        if isinstance(batch_indices[0], (list, tuple)):
            return [self.dataset[idx] for idx in batch_indices]
        return self.dataset[batch_indices]

    def __len__(self):
        """Return the number of samples in the underlying dataset."""
        return len(self.dataset)


def get_train_data_loader(
    config,
    num_workers: int,
    shuffle: bool = True,
    drop_last: bool = True,
    pin_mem: bool = True,
) -> torch.utils.data.DataLoader:
    """Build and return a DataLoader for distributed training with dynamic batching.

    Args:
        config: OmegaConf config containing training and data settings.
        num_workers: Number of worker processes for data loading.
        shuffle: Whether to shuffle the dataset each epoch.
        drop_last: Whether to drop the last incomplete batch.
        pin_mem: Whether to pin memory for faster GPU transfers.

    Returns:
        A DataLoader backed by DynamicBatchedMultiFeatureRandomSampler.
    """
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    # training.dataset_name picks the dataset class; the default is the DL3DV one.
    name = config.training.get(
        "dataset_name", "src.datasets.dynamic_dataset.DynamicInputDataset")
    module_name, class_name = name.rsplit(".", 1)
    dataset = importlib.import_module(module_name).__dict__[class_name](config)
    batch_sampler = dataset.make_sampler(
        batch_size_per_gpu=config.training.batch_size_per_gpu,
        shuffle=shuffle,
        world_size=world_size,
        rank=rank,
        drop_last=drop_last,
    )

    return torch.utils.data.DataLoader(
        DynamicBatchDatasetWrapper(dataset),
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )
