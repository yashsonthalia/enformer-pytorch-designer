import os
import numpy as np
from pathlib import Path
import random
import torch

torch.backends.cudnn.benchmark = True
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from transforms import RandomShift, RandomReverseComplement


class SequenceTargetIterableDataset(IterableDataset):
    """
    IterableDataset for paired sequence/target .npy files with
    multi-GPU and multi-worker support, plus flexible shuffling modes.

    Notes
    -----
    - Sample-level sharding across GPUs & workers is via `idx % total_shards == shard_id`.
    - In single-GPU, single-worker mode, no sharding occurs.
    """

    def __init__(
        self,
        seq_dir,
        tgt_dir,
        split,
        add_shift=False,
        add_rc=False,
        shuffle_files=False,
        shuffle_mode=None,
        shuffle_buffer_size=1024,
        seed=42,  # later set the epoch-specific seeds
        rank=0,
        world_size=1,
    ):
        self.seq_dir = Path(seq_dir)
        self.tgt_dir = Path(tgt_dir)
        self.split = split
        # set up transforms
        self.transforms = []
        if add_shift:
            self.transforms.append(RandomShift(max_shift=3, pad_value=0.0))
        if add_rc:
            self.transforms.append(RandomReverseComplement())
        # collect matching file pairs
        self.seq_paths = sorted(Path(self.seq_dir).glob(f"{split}_seqs.npy"))
        self.tgt_paths = sorted(Path(self.tgt_dir).glob(f"{split}_targets.npy"))
        assert len(self.seq_paths) == len(self.tgt_paths), (
            f"Mismatch: {len(self.seq_paths)} seq files vs {len(self.tgt_paths)} target files"
        )
        self.files = list(zip(self.seq_paths, self.tgt_paths))
        self.shuffle_files = shuffle_files
        self.shuffle_mode = shuffle_mode
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        np.random.seed(self.seed)
        # File-level shuffling is consistent across different workers
        file_rng = random.Random(self.seed)
        if self.split == "train" and self.shuffle_files:
            # shuffle file order between epochs during training
            file_rng.shuffle(self.files)

        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        shard_id = self.rank * num_workers + worker_id
        total_shards = self.world_size * num_workers

        # Specific RNG seed per worker for buffer & transforms
        random.seed(self.seed + shard_id)

        def raw_np_stream():
            for seq_path, tgt_path in self.files:
                seq_array = np.load(seq_path, mmap_mode="r")
                tgt_array = np.load(tgt_path, mmap_mode="r")

                if self.shuffle_mode == "intra_file":
                    indices = np.random.permutation(len(seq_array))
                else:
                    indices = range(len(seq_array))
                for i in indices:
                    yield seq_array[i], tgt_array[i]

        def sharded_raw():
            idx = 0
            for seq_np, tgt_np in raw_np_stream():
                if idx % total_shards == shard_id:
                    yield seq_np, tgt_np
                idx += 1

        if self.shuffle_mode == "buffer":
            src = sharded_raw()
            buffer = []

            for _ in range(self.shuffle_buffer_size):
                try:
                    buffer.append(next(src))
                except StopIteration:
                    break

            def buffered_shuffle():
                while buffer:
                    idx = random.randrange(len(buffer))
                    sample = buffer[idx]
                    try:
                        buffer[idx] = next(src)
                    except StopIteration:
                        buffer.pop(idx)
                    yield sample

            sample_iter = buffered_shuffle()

        else:
            sample_iter = sharded_raw()

        # Convert to tensors and apply transforms only on kept samples
        for seq_np, tgt_np in sample_iter:
            seq_t = torch.from_numpy(seq_np.astype(np.float32))
            tgt_t = torch.from_numpy(tgt_np.astype(np.float32))

            for tf in self.transforms:
                if isinstance(tf, RandomReverseComplement):
                    seq_t, tgt_t = tf(seq_t, tgt_t)
                else:
                    seq_t = tf(seq_t)

            yield seq_t, tgt_t


def get_dataloader(
    seq_dir: str,
    tgt_dir: str,
    split: str = "train",
    batch_size: int = 16,
    num_workers: int = 0,
    shuffle_files: bool = False,
    add_shift: bool = False,
    add_rc: bool = False,
    shuffle_mode: str = None,
    shuffle_buffer_size: int = 1024,
    seed: int = 42,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """
    Create a DataLoader for SequenceTargetIterableDataset with multi-GPU/workers support.

    Parameters
    ----------
    seq_dir : str
        Directory containing sequence .npy files.
    tgt_dir : str
        Directory containing target .npy files.
    split : {'train', 'valid', 'test'}, default='train'
        Data split to load. All splits are sharded across GPU+workers;
        only "train" will shuffle file order when 'shuffle_files=True'.
    batch_size : int, default=16
        Number of samples per batch.
    num_workers : int, default=0
        Number of DataLoader worker processes.
    shuffle_files : bool, default=False
        If True and split=='train', shuffle file order each epoch.
    add_shift : bool, default=False
        Add RandomShift augmentation to sequences.
    add_rc : bool, default=False
        Add RandomReverseComplement augmentation.
    shuffle_mode : {'intra_file', 'buffer', None}, default=None
        Sample-level shuffle method (intra-file or buffer).
    shuffle_buffer_size : int, default=1024
        Buffer size for 'buffer' shuffle mode.
    seed : int, default=42
        Base random seed for reproducibility.

    Returns
    -------
    DataLoader
        Configured DataLoader that works for single- or multi-GPU+workers.

    Notes
    -----
    - In single-GPU/single-worker mode, returns the full dataset (no sharding).
    - For training, each GPU+worker gets a disjoint shard of the data.
    """
    dataset = SequenceTargetIterableDataset(
        seq_dir,
        tgt_dir,
        split,
        add_shift,
        add_rc,
        shuffle_files=shuffle_files,
        shuffle_mode=shuffle_mode,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
        rank=rank,
        world_size=world_size,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
        drop_last=False,
    )
