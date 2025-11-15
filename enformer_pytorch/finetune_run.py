import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import glob
from pathlib import Path
import re
import argparse
import time
import math

import random
import numpy as np
import torch
import torch.multiprocessing as mp
from torch import nn
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import torch.distributed as dist

from tqdm import tqdm

from data_loader import get_dataloader
from modeling_enformer import from_pretrained, poisson_loss, pearson_corr_coef
from finetune_modes import HeadAdapterWrapper
from data import seq_indices_to_one_hot

import sys
import logging


def set_reproducibility(seed, ignore_if_cuda=False):
    """
    Set seeds for various randomness and enforces deterministic behavior
    in PyTorch to ensure reproducibility of experiments.

    Args:
        seed (int): The seed value to be used.
        ignore_if_cuda (bool, optional):
            -- If False (default): Enforce deterministic algorithms regardless of CUDA availability
            -- If True: skip enforcing deterministic algorithms if CUDA is available
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if not (ignore_if_cuda and torch.cuda.is_available()):
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError as e:
            print(
                f"Warning: {e}. Some operations might not be made deterministic.",
                file=sys.stderr,
                flush=True,
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Train Enformer PyTorch on human data")
    parser.add_argument(
        "--seed", type=int, default=42, help="Base random seed for reproducibility"
    )
    parser.add_argument(
        "--use_ddp",
        action="store_true",
        help="Spawn one process per GPU and use DistributedDataParallel",
    )
    parser.add_argument(
        "--use_dp",
        action="store_true",
        help="Use DataParallel if multiple GPUs are available",
    )
    parser.add_argument(
        "--shuffle_mode",
        choices=["intra_file", "buffer"],
        default=None,
        help="Sample-level shuffle mode for DataLoader",
    )
    parser.add_argument(
        "--shuffle_buffer_size",
        type=int,
        default=1024,
        help="Buffer size when shuffle_mode=buffer",
    )
    parser.add_argument(
        "--sequences_dir",
        type=str,
        required=True,
        help="Directory with train/valid sequence .npy files",
    )
    parser.add_argument(
        "--targets_dir",
        type=str,
        required=True,
        help="Directory with train/valid target .npy files",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Log training metrics every N optimizer updates",
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=1000,
        help="Run validation every N optimizer updates",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Base directory for saving checkpoints",
    )

    parser.add_argument(
        "--add_shift",
        action="store_true",
        help="Stochastic shift augmentation up to ±3 bp",
    )
    parser.add_argument(
        "--add_rc", action="store_true", help="Random reverse-complement augmentation"
    )

    return parser.parse_args()


def sync_ddp_scalar(value: float, device: torch.device) -> float:
    """
    Ensure that `value` (e.g. avg_loss) is the same on all ranks by
    broadcasting rank 0's copy to everyone.

    If not distributed or only one process, just returns value.
    """
    if dist.is_initialized() and dist.get_world_size() > 1:
        t = torch.tensor(value, device=device)
        dist.broadcast(t, src=0)
        return t.item()
    else:
        return value


def save_best_checkpoint(model, ckpt_dir, step, args):
    """
    Save the best-so-far model at a particular optimizer step.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"best-step-{step}.pt")
    state = model.module.state_dict() if args.use_ddp else model.state_dict()
    torch.save(state, path)


def save_last_epoch_checkpoint(model, ckpt_dir, epoch, args):
    """
    Save a snapshot at the end of epoch N (for resuming).
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"last-epoch{epoch}.pt")
    state = model.module.state_dict() if args.use_ddp else model.state_dict()
    torch.save(state, path)


def run_validation(
    model: torch.nn.Module,
    valid_loader: DataLoader,
    device: torch.device,
    step: int,
    best_val_corr: float,
    ckpt_dir: str,
    args,
):
    """
    Unified validation for single-GPU or DDP.
    """
    model.eval()
    local_loss_sum = torch.zeros((), device=device)
    local_corr_sum = torch.zeros((), device=device)
    local_count = 0

    with torch.no_grad():
        for seq, tgt in tqdm(valid_loader, desc="Validation", leave=False):
            seq = seq_indices_to_one_hot(seq.long()).squeeze(-2)
            seq, tgt = (
                seq.to(device, non_blocking=True),
                tgt.to(device, non_blocking=True),
            )
            out = model(seq)

            local_loss_sum += poisson_loss(out, tgt)
            local_corr_sum += pearson_corr_coef(out, tgt).mean()
            local_count += 1

    total_loss = local_loss_sum.item()
    total_corr = local_corr_sum.item()
    stats = torch.tensor(
        [total_loss, total_corr, local_count], device=device, dtype=torch.float64
    )

    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    total_loss_sum, total_corr_sum, total_count = stats.tolist()
    avg_loss = total_loss_sum / total_count
    avg_corr = total_corr_sum / total_count

    is_master = not dist.is_initialized() or dist.get_rank() == 0
    new_best = best_val_corr

    if is_master:
        # Structured, parsable validation log
        print(
            f"LOG VAL step={step} loss={avg_loss:.6f} corr={avg_corr:.6f}",
            flush=True,
        )

        if avg_corr > best_val_corr:
            save_best_checkpoint(model, ckpt_dir, step, args)
            new_best = avg_corr

    if dist.is_initialized() and dist.get_world_size() > 1:
        best_tensor = torch.tensor(new_best, device=device)
        dist.broadcast(best_tensor, src=0)
        best_val_corr = best_tensor.item()
    else:
        best_val_corr = new_best

    model.train()
    return avg_corr, best_val_corr


def main_worker(rank: int, world_size: int, args):
    try:
        # Handle ddp and multi-GPU training
        if args.use_ddp and world_size > 1:
            dist.init_process_group(
                backend="nccl",
                init_method="tcp://127.0.0.1:12355",
                world_size=world_size,
                rank=rank,
            )
            torch.cuda.set_device(rank)
            device = torch.device(f"cuda:{rank}")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set reproducibility
        set_reproducibility(args.seed)

        # Build model
        enformer = from_pretrained("EleutherAI/enformer-official-rough")

        model = HeadAdapterWrapper(
            enformer=enformer,
            num_tracks=6,
            post_transformer_embed=False,  # by default, embeddings are taken from after the final pointwise block w/ conv -> gelu - but if you'd like the embeddings right after the transformer block with a learned layernorm, set this to True
        ).cuda()

        if args.use_ddp and world_size > 1:
            model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        elif args.use_dp and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        # set up directories
        ckpt_dir = os.path.join(args.checkpoint_dir)
        os.makedirs(ckpt_dir, exist_ok=True)
        start_epoch = None

        if args.use_ddp and world_size > 1:
            if rank == 0:
                matches = glob.glob(os.path.join(ckpt_dir, "last-epoch*.pt"))
                epochs = []
                for path in matches:
                    fname = os.path.basename(path)
                    m = re.match(r"last-epoch(\d+)\.pt$", fname)
                    if m:
                        epochs.append(int(m.group(1)))

                start_epoch = max(epochs) + 1 if epochs else 1
            else:
                start_epoch = None

            dist.barrier()

            t = torch.tensor(start_epoch or 1, device=device, dtype=torch.int64)
            dist.broadcast(t, src=0)
            start_epoch = int(t.item())

            if start_epoch > 1:
                last_ckpt = os.path.join(ckpt_dir, f"last-epoch{start_epoch - 1}.pt")
                print(
                    f"[Rank {rank}] Resuming from {last_ckpt}",
                    file=sys.stderr,
                    flush=True,
                )
                state = torch.load(last_ckpt, map_location="cpu")
                if hasattr(model, "module"):  # DDP model
                    model.module.load_state_dict(state)
                else:
                    model.load_state_dict(state)
        else:
            matches = glob.glob(os.path.join(ckpt_dir, "last-epoch*.pt"))
            epochs = []
            for path in matches:
                fname = os.path.basename(path)
                m = re.match(r"last-epoch(\d+)\.pt$", fname)
                if m:
                    epochs.append(int(m.group(1)))

            start_epoch = max(epochs) + 1 if epochs else 1
            if start_epoch > 1:
                last_ckpt = os.path.join(ckpt_dir, f"last-epoch{start_epoch - 1}.pt")
                state = torch.load(last_ckpt, map_location="cpu")
                if hasattr(model, "module"):  # DDP model
                    model.module.load_state_dict(state)
                else:
                    model.load_state_dict(state)

            # count how many update-steps per epoch (1 update = 1 batch now)
            seq_paths = sorted(Path(args.sequences_dir).glob("train_seq*.npy"))
            num_samples = sum(np.load(p, mmap_mode="r").shape[0] for p in seq_paths)

        if args.use_ddp and world_size > 1:
            local_samples = math.ceil(num_samples / world_size)
        else:
            local_samples = num_samples

        steps_per_epoch = math.ceil(local_samples / args.batch_size)
        if rank == 0:
            print(
                f"Estimated steps per epoch: {steps_per_epoch}",
                file=sys.stderr,
                flush=True,
            )

        # optimizer (fixed LR, no scheduler)
        optimizer = AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        optimizer.zero_grad()

        # global_step in terms of optimizer updates
        global_step = (start_epoch - 1) * steps_per_epoch

        print(
            f"For training, add_shift: {args.add_shift}, add_rc: {args.add_rc}\n",
            file=sys.stderr,
            flush=True,
        )

        # validation loader
        valid_loader = get_dataloader(
            args.sequences_dir,
            args.targets_dir,
            split="val",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle_files=False,
            add_shift=False,
            add_rc=False,
            shuffle_mode=None,
            shuffle_buffer_size=args.shuffle_buffer_size,
            seed=args.seed,
            rank=rank,
            world_size=world_size,
        )

        best_val_corr = -float("inf")
        avg_corr, best_val_corr = run_validation(
            model,
            valid_loader,
            device,
            global_step,
            best_val_corr,
            ckpt_dir,
            args,
        )

        if rank == 0 and start_epoch == 1:
            save_last_epoch_checkpoint(model, ckpt_dir, start_epoch - 1, args)

        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start = time.time()
            model.train()

            train_loader = get_dataloader(
                args.sequences_dir,
                args.targets_dir,
                split="train",
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle_files=True,
                add_shift=args.add_shift,
                add_rc=args.add_rc,
                shuffle_mode=args.shuffle_mode,
                shuffle_buffer_size=args.shuffle_buffer_size,
                seed=args.seed + epoch,
                rank=rank,
                world_size=world_size,
            )

            epoch_loss = 0.0
            running_loss = 0.0
            running_steps = 0

            for batch_idx, (seq, tgt) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch}", disable=(rank != 0)), start=1
            ):
                seq = seq_indices_to_one_hot(seq.long()).squeeze(-2)
                seq, tgt = seq.to(device), tgt.to(device)

                out = model(seq)

                raw_loss = poisson_loss(out, tgt)
                loss = raw_loss

                loss.backward()

                # grad clipping
                pre_clip_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )

                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                post_clip_norm = total_norm**0.5

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_loss += raw_loss.item()
                running_loss += raw_loss.item()
                running_steps += 1

                if rank == 0 and global_step % args.log_interval == 0:
                    avg_running_loss = running_loss / max(1, running_steps)
                    # Structured, parsable train log
                    print(
                        f"LOG TRAIN step={global_step} loss={avg_running_loss:.6f}",
                        flush=True,
                    )
                    running_loss = 0.0
                    running_steps = 0

                if args.val_interval > 0 and global_step % args.val_interval == 0:
                    avg_corr, best_val_corr = run_validation(
                        model,
                        valid_loader,
                        device,
                        global_step,
                        best_val_corr,
                        ckpt_dir,
                        args,
                    )

            epoch_time = time.time() - epoch_start
            avg_epoch_loss = epoch_loss / batch_idx

            if rank == 0:
                # Structured, parsable epoch summary log
                print(
                    f"LOG EPOCH epoch={epoch} step={global_step} avg_loss={avg_epoch_loss:.6f} time={epoch_time:.1f}",
                    flush=True,
                )

            # end-of-epoch validation if not just run
            if args.val_interval <= 0 or global_step % args.val_interval != 0:
                avg_corr, best_val_corr = run_validation(
                    model,
                    valid_loader,
                    device,
                    global_step,
                    best_val_corr,
                    ckpt_dir,
                    args,
                )

            if rank == 0:
                save_last_epoch_checkpoint(model, ckpt_dir, epoch, args)

        if rank == 0:
            print("Training complete.", file=sys.stderr, flush=True)

    finally:
        if args.use_ddp and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    world_size = torch.cuda.device_count()
    if args.use_ddp and world_size > 1:
        mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        main_worker(rank=0, world_size=1, args=args)
