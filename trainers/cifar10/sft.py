import gc
import logging
import os
from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path
from typing import Final, Sequence, cast

import safetensors.torch as st
import torch
import torch.distributed as dist
import torch.nn as nn
import torchmetrics as tm
import yaml
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, DistributedSampler
from torchmetrics.wrappers import Running
from torchvision.datasets import CIFAR10
from torchvision.transforms.v2 import (
    ColorJitter,
    Compose,
    RandomApply,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomRotation,
    RandomVerticalFlip,
    Resize,
    ToDtype,
    ToImage,
)
from tqdm import tqdm
from vit import ViT, ViTConfig

from mjepa.augmentation import (
    AugmentationConfig,
    apply_invert,
    apply_mixup,
    apply_noise,
    apply_posterize,
    cross_entropy_mixup,
    is_mixed,
)
from mjepa.logging import CSVLogger
from mjepa.optimizer import OptimizerConfig
from mjepa.trainer import (
    DataLoaderFn,
    TrainerConfig,
    assert_all_trainable_params_have_grad,
    calculate_total_steps,
    count_parameters,
    format_large_number,
    format_pbar_description,
    ignore_warnings,
    is_rank_zero,
    load_checkpoint,
    rank_zero_info,
    save_checkpoint,
    setup_logdir,
    should_step_optimizer,
    size_change,
)


NUM_CLASSES: Final[int] = 10
UNKNOWN_LABEL: Final[int] = -1
WINDOW: Final[int] = 5
LOG_INTERVAL: Final[int] = 50


def get_train_transforms(size: Sequence[int]) -> Compose:
    return Compose(
        [
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomResizedCrop(size=size, scale=(0.75, 1.0), ratio=(0.75, 1.33)),
            RandomApply([RandomRotation(degrees=15)], p=0.25),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            RandomGrayscale(p=0.1),
            ToImage(),
            ToDtype(torch.float32, scale=True),
        ]
    )


def get_val_transforms(size: Sequence[int]) -> Compose:
    return Compose(
        [
            Resize(size=size),
            ToImage(),
            ToDtype(torch.float32, scale=True),
        ]
    )


def get_train_dataloader(
    size: Sequence[int],
    batch_size: int,
    root: Path,
    num_workers: int,
    local_rank: int,
    world_size: int,
) -> DataLoader:
    transforms = get_train_transforms(size)
    dataset = CIFAR10(root=root, train=True, transform=transforms, download=True)
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
        return DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler, persistent_workers=True
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
            persistent_workers=True,
        )


def get_val_dataloader(
    size: Sequence[int],
    batch_size: int,
    root: Path,
    num_workers: int,
    local_rank: int,
    world_size: int,
) -> DataLoader:
    transforms = get_val_transforms(size)
    dataset = CIFAR10(root=root, train=False, transform=transforms, download=True)
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
        return DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler, persistent_workers=True
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False,
            persistent_workers=True,
        )


def train(
    modules: nn.ModuleDict,
    train_dataloader_fn: DataLoaderFn,
    val_dataloader_fn: DataLoaderFn,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    augmentation_config: AugmentationConfig,
    trainer_config: TrainerConfig,
    log_dir: Path | None = None,
    last_epoch: int = -1,
) -> None:
    # Module setup
    rank_zero_info(f"Starting training, logging to {log_dir if log_dir else 'console only'}")
    optimizer.zero_grad()
    backbone: ViT = modules["backbone"]
    rank_zero_info(f"Backbone params: {format_large_number(count_parameters(backbone))}")

    # DataLoader setup
    train_dataloader = train_dataloader_fn(backbone.config.img_size, trainer_config.batch_size)
    val_dataloader = val_dataloader_fn(backbone.config.img_size, trainer_config.batch_size)

    accumulate_grad_batches = trainer_config.accumulate_grad_batches
    microbatch = (last_epoch + 1) * len(train_dataloader)
    step = microbatch // accumulate_grad_batches
    total_steps = calculate_total_steps(train_dataloader, trainer_config.num_epochs, accumulate_grad_batches)
    rank_zero_info(f"Training for {trainer_config.num_epochs} epochs = {total_steps} steps")
    rank_zero_info(
        f"Batch size: {trainer_config.batch_size}, Microbatch accumulation: {trainer_config.accumulate_grad_batches}"
    )

    # Metric setup
    train_loss = tm.RunningMean(window=WINDOW).cuda()
    train_acc = Running(tm.Accuracy(task="multiclass", num_classes=NUM_CLASSES), window=WINDOW).cuda()
    val_acc = tm.Accuracy(task="multiclass", num_classes=NUM_CLASSES).cuda()
    logger = CSVLogger(
        log_dir / "run.csv", interval=LOG_INTERVAL, accumulate_grad_batches=trainer_config.accumulate_grad_batches
    )

    img: Tensor
    label: Tensor
    for epoch in range(last_epoch + 1, trainer_config.num_epochs):
        # Update training resolution / batch_size / accumulate_grad_batches if necessary
        if trainer_config.is_size_change_epoch(epoch):
            size_config = trainer_config.sizes[epoch]
            train_dataloader, val_dataloader, accumulate_grad_batches = size_change(
                size_config,
                trainer_config.batch_size,
                accumulate_grad_batches,
                train_dataloader_fn,
                val_dataloader_fn,
            )
            rank_zero_info(
                f"Changing size to {size_config.size} and batch size to {size_config.batch_size} (accumulate grad batches: {accumulate_grad_batches})"
            )

        modules.train()
        desc = format_pbar_description(step, microbatch, epoch, loss=train_loss, acc=train_acc)
        pbar = tqdm(train_dataloader, desc=desc, disable=not is_rank_zero(), leave=False)
        for img, label in pbar:
            B = img.shape[0]
            img = img.cuda()
            label = label.cuda()
            tokenized_size = backbone.stem.tokenized_size(img.shape[-2:])

            # Apply augmentations
            with torch.no_grad():
                img = apply_noise(augmentation_config, img)
                mixup_seed, (img,) = apply_mixup(augmentation_config, img)
                img = apply_invert(augmentation_config, img)[0]
                img = apply_posterize(augmentation_config, img)[0]

            # Forward pass and loss
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                features = cast(Tensor, backbone(img))
                pred = backbone.heads["cls"](features)

                loss = cross_entropy_mixup(
                    pred, label, mixup_seed, augmentation_config.mixup_prob, augmentation_config.mixup_alpha
                ).mean()
                train_loss.update(loss)

            # Metric computation
            with torch.no_grad():
                mask = is_mixed(
                    B, augmentation_config.mixup_prob, augmentation_config.mixup_alpha, mixup_seed
                ).logical_not_()
                train_acc.update(pred[mask], label[mask])

            # Backward
            assert not loss.isnan()
            loss.backward()
            assert_all_trainable_params_have_grad(modules, microbatch)
            microbatch += 1

            # Optimizer update and teacher update
            if should_step_optimizer(microbatch, accumulate_grad_batches):
                if step < total_steps:
                    scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                step += 1

            desc = format_pbar_description(step, microbatch, epoch, loss=train_loss, acc=train_acc)
            pbar.set_description(desc)

        # Validation
        pbar.close()
        if (epoch + 1) % trainer_config.check_val_every_n_epoch == 0:
            modules.eval()
            val_acc.reset()
            for img, label in tqdm(val_dataloader, desc=f"Validating: ", disable=not is_rank_zero(), leave=False):
                B = img.shape[0]
                img = img.cuda()
                label = label.cuda()
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    features = cast(Tensor, backbone(img))
                    pred = backbone.heads["cls"](features)
                    val_acc.update(pred, label)

            # Validation epoch end
            rank_zero_info(f"Epoch: {epoch}, Val Acc: {val_acc.compute():.4f}")
            logger.log(epoch, step, microbatch, acc=val_acc.compute())

        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # Save checkpoint
        if log_dir:
            save_checkpoint(
                path=log_dir / f"checkpoint.pt",
                backbone=backbone,
                predictor=None,
                teacher=None,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                epoch=epoch,
            )
            st.save_file(
                {k: v for k, v in backbone.state_dict().items() if isinstance(v, torch.Tensor)},
                str(log_dir / f"backbone.safetensors"),
            )

    # Save final checkpoint
    if log_dir:
        st.save_file(
            {k: v for k, v in backbone.state_dict().items() if isinstance(v, torch.Tensor)},
            str(log_dir / f"backbone.safetensors"),
        )


def ddp_setup() -> None:
    dist.init_process_group(backend="nccl")
    logging.info("Initialized DDP")


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
    return parser.parse_args()


def main(args: Namespace) -> None:
    if not (config_path := Path(args.config)).is_file():
        raise FileNotFoundError(config_path)
    config = yaml.full_load(config_path.read_text())

    # Extract instantiated dataclasses from config
    backbone_config = config["backbone"]
    augmentation_config = config["augmentations"]
    optimizer_config = config["optimizer"]
    trainer_config = config["trainer"]
    assert isinstance(backbone_config, ViTConfig)
    assert isinstance(augmentation_config, AugmentationConfig)
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
    wrapper = nn.ModuleDict({"backbone": backbone}).cuda()

    # Wrap in DDP for distributed training
    if world_size > 1:
        ddp_setup()
        wrapper = DDP(wrapper, device_ids=[local_rank])

    # Instantiate dataloaders
    train_dataloader_fn = partial(
        get_train_dataloader,
        root=args.data,
        num_workers=trainer_config.num_workers,
        local_rank=local_rank,
        world_size=world_size,
    )
    val_dataloader_fn = partial(
        get_val_dataloader,
        root=args.data,
        num_workers=trainer_config.num_workers,
        local_rank=local_rank,
        world_size=world_size,
    )
    train_dataloader = train_dataloader_fn(backbone.config.img_size, trainer_config.batch_size)

    # Instantiate optimizer and scheduler
    total_steps = calculate_total_steps(
        train_dataloader, trainer_config.num_epochs, trainer_config.accumulate_grad_batches
    )
    optimizer, scheduler = optimizer_config.instantiate(wrapper, total_steps=total_steps)

    if args.checkpoint:
        # NOTE: For now only support fresh checkpoint
        load_checkpoint(
            args.checkpoint,
            backbone,
            None,
            None,
            optimizer,
            scheduler,
            mode="fresh",
        )

    # Create log subdirectory with timestamp
    if args.log_dir:
        log_dir = setup_logdir(args.log_dir, config_path, name=args.name)
    else:
        log_dir = None

    ignore_warnings()
    train(
        wrapper,
        train_dataloader_fn,
        val_dataloader_fn,
        optimizer,
        scheduler,
        augmentation_config,
        trainer_config,
        log_dir,
    )


def entrypoint() -> None:
    args = parse_args()
    main(args)


if __name__ == "__main__":
    entrypoint()
