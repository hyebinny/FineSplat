import numpy as np

from src.datasets.batched_sampler import DynamicBatchedMultiFeatureRandomSampler


class BaseDataset:
    """Abstract base class for multi-resolution training datasets."""

    def __init__(self, config):
        """
        Args:
            config: OmegaConf config containing training and data settings.
        """
        self.num_views = config.training.num_views
        self._set_resolutions(config.training.res_dict)
        self._set_input_views(config.training.num_views)
        self._set_target_views(config.training.target_views)
        self.seed = config.data.seed
        self.max_num_retries = 10

    def set_epoch(self, epoch):
        """Hook for epoch-level state updates; subclasses may override.

        Args:
            epoch: The current training epoch index.
        """
        pass

    def make_sampler(
        self,
        batch_size_per_gpu,
        shuffle: bool = True,
        world_size: int = 1,
        rank: int = 0,
        drop_last: bool = True
    ):
        """Construct a DynamicBatchedMultiFeatureRandomSampler for this dataset.

        Args:
            batch_size_per_gpu: List of per-GPU batch sizes, one per view-count bucket.
            shuffle: Must be True; non-shuffled iteration is not implemented.
            world_size: Total number of distributed processes.
            rank: This process's distributed rank.
            drop_last: Whether to drop the final incomplete batch.

        Returns:
            A DynamicBatchedMultiFeatureRandomSampler instance.
        """
        if not shuffle:
            raise NotImplementedError("Only shuffle=True is supported for now.")

        num_of_num_views = len(self.num_views)
        feature_to_batch_size_map = {i: bs for i, bs in enumerate(batch_size_per_gpu)}

        return DynamicBatchedMultiFeatureRandomSampler(
            dataset=self,
            pool_sizes=[num_of_num_views],
            scaling_feature_idx=0,
            feature_to_batch_size_map=feature_to_batch_size_map,
            world_size=world_size,
            rank=rank,
            drop_last=drop_last,
        )

    def _scene_len(self):
        """Initialize num_of_scenes from data_path; subclasses must set data_path first."""
        self.data_path = []  # set in subclass
        self.num_of_scenes = len(self.data_path)

    def __len__(self):
        """Return the total number of scenes in the dataset."""
        return self.num_of_scenes

    def _get_views(self, idx, resolution, num_views_to_input, num_views_to_target):
        """Load and return a scene sample. Must be implemented by subclasses.

        Args:
            idx: Scene index into data_path.
            resolution: (height, width) tuple for image loading.
            num_views_to_input: Number of input views to select.
            num_views_to_target: Number of target views to select.

        Returns:
            A dict containing image tensors, intrinsics, and pose tensors.
        """
        raise NotImplementedError()

    def _set_resolutions(self, resolutions):
        """Build an index-keyed dict of (height, width) resolution tuples.

        Args:
            resolutions: List of [height, width] pairs from config.
        """
        self._resolutions = {i: tuple(res) for i, res in enumerate(resolutions)}

    def _set_input_views(self, num_views):
        """Build an index-keyed dict of input view counts.

        Args:
            num_views: List of input view counts from config.
        """
        self._input_views = dict(enumerate(num_views))

    def _set_target_views(self, target_views):
        """Build an index-keyed dict of target view counts.

        Args:
            target_views: List of target view counts from config.
        """
        self._target_views = dict(enumerate(target_views))

    def _getitem_fn(self, idx):
        """Dispatch a sampler-produced index tuple to _get_views.

        Args:
            idx: A tuple of (sample_idx, dict_idx) where dict_idx selects
                the resolution/view-count bucket.

        Returns:
            The result of _get_views for the resolved parameters.
        """
        idx, dict_idx = idx
        resolution = self._resolutions[dict_idx]
        input_view = self._input_views[dict_idx]
        target_view = self._target_views[dict_idx]
        return self._get_views(idx, resolution, input_view, target_view)

    def __getitem__(self, idx):
        """Return a dataset sample, retrying with a random index on failure.

        Args:
            idx: A (sample_idx, dict_idx) tuple as yielded by the sampler.

        Returns:
            A valid sample dict from _getitem_fn.

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        for _ in range(self.max_num_retries + 1):
            try:
                return self._getitem_fn(idx)
            except Exception:
                if isinstance(idx, tuple):
                    idx_list = list(idx)
                    idx_list[0] = np.random.randint(0, len(self))
                    idx = tuple(idx_list)
                else:
                    idx = np.random.randint(0, len(self))
        raise RuntimeError(f"Failed to load a valid sample after {self.max_num_retries} retries.")
