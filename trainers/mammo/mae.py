import math
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Final, Tuple, cast

import safetensors.torch as st
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
import yaml
from einops import rearrange
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.transforms.v2 import ColorJitter, Compose, RandomHorizontalFlip, RandomResizedCrop, RandomVerticalFlip
from tqdm import tqdm
from vit import ViT, ViTConfig
from vit.tokens import apply_mask

from mjepa.data import PreprocessedTIFFDataset
from mjepa.jepa import CrossAttentionPredictor, generate_masks
from mjepa.logging import CSVLogger, SaveImage
from mjepa.mae import MAEConfig
from mjepa.optimizer import OptimizerConfig
from mjepa.trainer import (
    TrainerConfig,
    assert_all_trainable_params_have_grad,
    calculate_total_steps,
    count_parameters,
    format_large_number,
    format_pbar_description,
    ignore_warnings,
    is_rank_zero,
    rank_zero_info,
    save_checkpoint,
    setup_logdir,
    should_step_optimizer,
    seed_everything,
)


WINDOW: Final[int] = 5
DEFAULT_SIZE: Final[Tuple[int, int]] = (384, 256)
LOG_INTERVAL: Final[int] = 50


def get_train_transforms(size: Tuple[int, int]) -> Compose:
    return Compose(
        [
            RandomResizedCrop(size=size, scale=(0.1, 1.0), ratio=(0.75, 1.33)),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            ColorJitter(brightness=0.2, contrast=0.2),
        ]
    )


def get_train_dataloader(
    root: Path,
    size: Tuple[int, int],
    batch_size: int,
    num_workers: int,
    keep_volume: bool,
    shuffle: bool,
    local_rank: int,
    world_size: int,
) -> DataLoader:
    transforms = get_train_transforms(size)
    dataset = PreprocessedTIFFDataset(root=root, training=True, transform=transforms, keep_volume=keep_volume)
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=shuffle)
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler)
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=True,
        )


def train(
    modules: nn.ModuleDict,
    train_dataloader: DataLoader,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    mae_config: MAEConfig,
    trainer_config: TrainerConfig,
    log_dir: Path | None = None,
    last_epoch: int = -1,
) -> None:
    # Module setup
    rank_zero_info(f"Starting training, logging to {log_dir if log_dir else 'console only'}")
    optimizer.zero_grad()
    backbone: ViT = modules["backbone"]
    predictor: CrossAttentionPredictor = modules["predictor"]
    rank_zero_info(f"Backbone params: {format_large_number(count_parameters(backbone))}")
    rank_zero_info(f"Predictor params: {format_large_number(count_parameters(predictor))}")

    microbatch = (last_epoch + 1) * len(train_dataloader)
    step = microbatch // trainer_config.accumulate_grad_batches
    total_steps = calculate_total_steps(
        train_dataloader, trainer_config.num_epochs, trainer_config.accumulate_grad_batches
    )
    rank_zero_info(f"Training for {trainer_config.num_epochs} epochs = {total_steps} steps")
    rank_zero_info(
        f"Batch size: {trainer_config.batch_size}, Microbatch accumulation: {trainer_config.accumulate_grad_batches}"
    )

    # Metrics and logging
    train_loss = tm.RunningMean(window=WINDOW).cuda()
    train_loss_epoch = tm.MeanMetric().cuda()
    logger = CSVLogger(
        log_dir / "train.csv", interval=LOG_INTERVAL, accumulate_grad_batches=trainer_config.accumulate_grad_batches
    )
    image_saver = SaveImage(log_dir / "first_batch.png", max_save_images=8)

    img: Tensor
    for epoch in range(last_epoch + 1, trainer_config.num_epochs):
        modules.train()
        desc = format_pbar_description(step, microbatch, epoch, loss=train_loss)
        pbar = tqdm(train_dataloader, desc=desc, disable=not is_rank_zero(), leave=False)
        for img in pbar:
            img = img.cuda()
            if microbatch == 0:
                image_saver(img)

            # Fold the image to create the target
            hp, wp = backbone.config.patch_size
            target = rearrange(img, "b c (h hp) (w wp) -> b (h w) (hp wp c)", hp=hp, wp=wp)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Masked forward through student and predictor
                context_mask, target_mask = generate_masks(
                    backbone, img, mae_config.context_ratio, mae_config.target_ratio, mae_config.scale
                )
                context = cast(Tensor, backbone(img, mask=context_mask))
                tokenized_size = backbone.stem.tokenized_size(img.shape[-2:])
                pred: Tensor = predictor(tokenized_size, context, context_mask, target_mask)

                # Compute MAE loss
                target = apply_mask(target_mask, target, fill_value=None)
                loss = F.l1_loss(pred, target)
                train_loss.update(loss)
                train_loss_epoch.update(loss)

            # Backward
            assert not loss.isnan()
            loss.backward()
            assert_all_trainable_params_have_grad(modules, microbatch)
            logger.log(epoch, step, microbatch, loss=train_loss.compute().item())

            # Optimizer update and teacher update
            if should_step_optimizer(microbatch, trainer_config.accumulate_grad_batches):
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                step += 1

            microbatch += 1
            desc = format_pbar_description(step, microbatch, epoch, loss=train_loss)
            pbar.set_description(desc)

        rank_zero_info(f"Epoch: {epoch}, Loss: {train_loss_epoch.compute():.4f}")
        train_loss_epoch.reset()

        # Save checkpoint
        if log_dir:
            save_checkpoint(
                path=log_dir / f"checkpoint.pt",
                backbone=backbone,
                predictor=predictor,
                probe=None,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                epoch=epoch,
            )
            st.save_file(
                {k: v for k, v in backbone.state_dict().items() if isinstance(v, torch.Tensor)},
                str(log_dir / f"backbone.safetensors"),
            )

        pbar.close()

    # Save final checkpoint
    if log_dir:
        st.save_file(
            {k: v for k, v in backbone.state_dict().items() if isinstance(v, torch.Tensor)},
            str(log_dir / f"backbone.safetensors"),
        )


def ddp_setup() -> None:
    dist.init_process_group(backend="nccl")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("config", type=Path, help="Path to YAML configuration file")
    parser.add_argument("data", type=Path, help="Path to training data")
    parser.add_argument(
        "-n", "--name", type=str, default=None, help="Name of the run. Will be appended to the log subdirectory."
    )
    parser.add_argument("-l", "--log-dir", type=Path, default=None, help="Directory to save logs")
    parser.add_argument("--local-rank", type=int, default=1, help="Local rank / device")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to checkpoint to load")
    parser.add_argument("--size", type=int, nargs=2, default=DEFAULT_SIZE, help="Training image size")
    return parser.parse_args()


def main(args: Namespace) -> None:
    seed_everything(0)
    if not (config_path := Path(args.config)).is_file():
        raise FileNotFoundError(config_path)
    config = yaml.full_load(config_path.read_text())

    # Extract instantiated dataclasses from config
    backbone_config = config["backbone"]
    mae_config = config["mae"]
    optimizer_config = config["optimizer"]
    trainer_config = config["trainer"]
    assert isinstance(backbone_config, ViTConfig)
    assert isinstance(mae_config, MAEConfig)
    assert isinstance(optimizer_config, OptimizerConfig)
    assert isinstance(trainer_config, TrainerConfig)
    if args.log_dir and not args.log_dir.is_dir():
        raise NotADirectoryError(args.log_dir)

    # Determine distributed training parameters
    world_size = os.environ.get("WORLD_SIZE", 1)
    local_rank = os.environ.get("LOCAL_RANK", args.local_rank)
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        ddp_setup()

    # Instantiate other model elements and move to device
    backbone = backbone_config.instantiate()
    out_dim = math.prod(backbone.config.patch_size) * backbone.config.in_channels
    predictor = CrossAttentionPredictor(
        backbone, mae_config.predictor_depth, mae_config.context_pos_emb, mae_config.shared_pos_emb, out_dim
    )
    wrapper = nn.ModuleDict({"backbone": backbone, "predictor": predictor}).cuda()

    # Wrap in DDP for distributed training
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        wrapper = DDP(wrapper, device_ids=[local_rank])

    # Instantiate dataloaders
    keep_volume = False
    shuffle = True
    train_dataloader = get_train_dataloader(
        args.data,
        args.size,
        trainer_config.batch_size,
        trainer_config.num_workers,
        keep_volume,
        shuffle,
        local_rank,
        world_size,
    )

    # Instantiate optimizer and scheduler
    total_steps = trainer_config.num_epochs * len(train_dataloader) // trainer_config.accumulate_grad_batches
    optimizer, scheduler = optimizer_config.instantiate(wrapper, total_steps=total_steps)

    # Create log subdirectory with timestamp
    if args.log_dir:
        log_dir = setup_logdir(args.log_dir, config_path, name=args.name)
    else:
        log_dir = None

    ignore_warnings()
    rank_zero_info(f"Training with image size {tuple(args.size)}")
    train(
        wrapper,
        train_dataloader,
        optimizer,
        scheduler,
        mae_config,
        trainer_config,
        log_dir,
    )


def entrypoint() -> None:
    args = parse_args()
    main(args)


if __name__ == "__main__":
    entrypoint()
