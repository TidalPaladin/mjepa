import logging
import os
from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path
from typing import Any, Final, Sequence, Tuple, cast

import safetensors.torch as st
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
import yaml
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.transforms.v2 import (
    ColorJitter,
    Compose,
    RandomApply,
    RandomChoice,
    RandomCrop,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomRotation,
    RandomVerticalFlip,
    Resize,
)
from torchvision.transforms.v2.functional import InterpolationMode
from tqdm import tqdm
from vit import ViT, ViTConfig
from vit.tokens import apply_mask

from mjepa.augmentation import AugmentationConfig, apply_invert, apply_mixup, apply_noise, apply_posterize
from mjepa.data import PreprocessedTIFFDataset
from mjepa.jepa import (
    CrossAttentionPredictor,
    JEPAConfig,
    compute_gram_loss,
    forward_gram_teacher,
    generate_masks,
    get_momentum,
    is_gram_update_epoch,
    setup_teacher,
    update_teacher,
)
from mjepa.logging import CSVLogger, SaveImage
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
    rank_zero_info,
    save_checkpoint,
    scale_change,
    setup_logdir,
    should_step_optimizer,
    size_change,
)


WINDOW: Final[int] = 5
LOG_INTERVAL: Final = 50


def get_train_transforms(size: Sequence[int]) -> Compose:
    return Compose(
        [
            RandomApply([RandomRotation(degrees=cast(Any, 15), interpolation=InterpolationMode.BILINEAR)], p=0.25),
            RandomChoice(
                [
                    Resize(size=size, interpolation=InterpolationMode.BILINEAR),
                    RandomResizedCrop(size=size, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
                    RandomCrop(size=size, pad_if_needed=True),
                ]
            ),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            ColorJitter(brightness=0.2, contrast=0.2),
        ]
    )


def get_train_dataloader(
    size: Sequence[int],
    batch_size: int,
    root: Path,
    num_workers: int,
    keep_volume: bool,
    shuffle: bool,
    local_rank: int,
    world_size: int,
    highres_data: Path | None = None,
    highres_cutoff: Tuple[int, int] = (1024, 768),
) -> DataLoader:
    if (size[0] >= highres_cutoff[0] or size[1] >= highres_cutoff[1]) and highres_data is not None:
        root = highres_data
    else:
        root = root
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
    train_dataloader_fn: DataLoaderFn,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    jepa_config: JEPAConfig,
    augmentation_config: AugmentationConfig,
    trainer_config: TrainerConfig,
    log_dir: Path | None = None,
    last_epoch: int = -1,
) -> None:
    # Module setup
    rank_zero_info(f"Starting training, logging to {log_dir if log_dir else 'console only'}")
    optimizer.zero_grad()

    # Handle DDP wrapper
    if isinstance(modules, DDP):
        backbone = cast(ViT, modules.module["backbone"])
        predictor = cast(CrossAttentionPredictor, modules.module["predictor"])
    else:
        backbone = cast(ViT, modules["backbone"])
        predictor = cast(CrossAttentionPredictor, modules["predictor"])
    rank_zero_info(f"Backbone params: {format_large_number(count_parameters(backbone))}")
    rank_zero_info(f"Predictor params: {format_large_number(count_parameters(predictor))}")
    jepa_scale = jepa_config.scale

    # DataLoader setup
    train_dataloader = train_dataloader_fn(backbone.config.img_size, trainer_config.batch_size)

    # Teacher setup
    teacher = setup_teacher(backbone)

    # Gram teacher setup (pre-allocated)
    gram_teacher: ViT | None = setup_teacher(backbone) if jepa_config.gram_start_epoch is not None else None
    computing_gram_loss = False

    accumulate_grad_batches = trainer_config.accumulate_grad_batches
    microbatch = (last_epoch + 1) * len(train_dataloader)
    step = microbatch // accumulate_grad_batches
    total_steps = calculate_total_steps(train_dataloader, trainer_config.num_epochs, accumulate_grad_batches)
    rank_zero_info(f"Training for {trainer_config.num_epochs} epochs = {total_steps} steps")
    rank_zero_info(
        f"Batch size: {trainer_config.batch_size}, Microbatch accumulation: {trainer_config.accumulate_grad_batches}"
    )

    # Metrics and logging
    train_loss = tm.RunningMean(window=WINDOW).cuda()
    train_loss_epoch = tm.MeanMetric().cuda()
    logger = (
        CSVLogger(
            log_dir / "train.csv", interval=LOG_INTERVAL, accumulate_grad_batches=trainer_config.accumulate_grad_batches
        )
        if log_dir
        else None
    )
    image_saver = SaveImage(log_dir / "first_batch.png", max_save_images=8) if log_dir else None

    img: Tensor
    for epoch in range(last_epoch + 1, trainer_config.num_epochs):
        # Update training resolution / batch_size / accumulate_grad_batches if necessary
        if trainer_config.is_size_change_epoch(epoch):
            size_config = trainer_config.sizes[epoch]
            train_dataloader, _, accumulate_grad_batches = size_change(
                size_config,
                trainer_config.batch_size,
                accumulate_grad_batches,
                train_dataloader_fn,
                None,
            )
            jepa_scale = scale_change(backbone.config.img_size, size_config, jepa_config.scale)
            rank_zero_info(
                f"Changing size to {size_config.size} and batch size to {size_config.batch_size} "
                f"(accumulate grad batches: {accumulate_grad_batches}, jepa scale: {jepa_scale})"
            )

        # Initial Gram teacher setup (if necessary)
        if epoch == jepa_config.gram_teacher_epoch and jepa_config.gram_start_epoch is not None:
            assert gram_teacher is not None
            update_teacher(teacher, gram_teacher)
            rank_zero_info(f"Gram teacher initialized on epoch {epoch}")
        # Start computing Gram loss (if necessary)
        if jepa_config.gram_start_epoch is not None and epoch == jepa_config.gram_start_epoch:
            computing_gram_loss = True
            rank_zero_info(f"Started computing Gram loss on epoch {epoch}")
        # Gram teacher update
        if is_gram_update_epoch(epoch, jepa_config.gram_start_epoch, jepa_config.gram_update_interval_epoch):
            assert gram_teacher is not None
            assert computing_gram_loss
            update_teacher(teacher, gram_teacher)
            rank_zero_info(f"Gram teacher updated on epoch {epoch}")

        modules.train()
        desc = format_pbar_description(step, microbatch, epoch, loss=train_loss)
        pbar = tqdm(train_dataloader, desc=desc, disable=not is_rank_zero(), leave=False)
        for img in pbar:
            img = img.cuda()
            tokenized_size = backbone.stem.tokenized_size(img.shape[-2:])
            rope_seed = int(torch.randint(0, 1000000, (1,)).item())
            if microbatch == 0 and image_saver is not None:
                image_saver(img)

            # Teacher forward pass (unaugmented)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16), torch.inference_mode():
                assert not teacher.training
                teacher_output = cast(Tensor, teacher(img, rope_seed=rope_seed))
                rope = (
                    teacher.rope(H=tokenized_size[0], W=tokenized_size[1], rope_seed=rope_seed)
                    if teacher.rope is not None
                    else None
                )
                teacher_output_cls = teacher.get_head("pool")(teacher_output, rope=rope)
            teacher_output = teacher_output.clone()
            teacher_output_cls = teacher_output_cls.clone()

            # Gram teacher forward pass (if necessary)
            if computing_gram_loss:
                assert gram_teacher is not None
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16), torch.inference_mode():
                    gram_teacher_output = forward_gram_teacher(
                        gram_teacher,
                        img,
                        rope_seed=rope_seed,
                        resolution_scale=jepa_config.gram_resolution_scale,
                    )
                gram_teacher_output = gram_teacher_output.clone()
                assert gram_teacher_output.shape == teacher_output.shape
            else:
                gram_teacher_output = None

            # Apply augmentations
            with torch.no_grad():
                img = apply_noise(augmentation_config, img)
                if gram_teacher_output is not None:
                    _, (img, teacher_output, gram_teacher_output) = apply_mixup(
                        augmentation_config, img, teacher_output, gram_teacher_output
                    )
                else:
                    _, (img, teacher_output) = apply_mixup(augmentation_config, img, teacher_output)
                img = apply_invert(augmentation_config, img)[0]
                img = apply_posterize(augmentation_config, img)[0]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Masked forward through student and predictor
                context_mask, target_mask = generate_masks(
                    backbone, img, jepa_config.context_ratio, jepa_config.target_ratio, jepa_scale
                )
                context = cast(Tensor, backbone(img, mask=context_mask, rope_seed=rope_seed))
                tokenized_size = backbone.stem.tokenized_size(img.shape[-2:])
                pred: Tensor = predictor(tokenized_size, context, context_mask, target_mask, rope_seed=rope_seed)

                # Second predictor forward with pooled token
                # NOTE: Stop gradient on context is very important - otherwise model cheats via over-reliance on the pooled token.
                # NOTE: Since pooled token is part of context we reuse the backbone's output norm to normalize it.
                rope = (
                    backbone.rope(H=tokenized_size[0], W=tokenized_size[1], rope_seed=rope_seed)
                    if backbone.rope is not None
                    else None
                )
                pred_cls = backbone.get_head("pool")(teacher_output, rope=rope)
                context_with_cls = torch.cat([pred_cls.unsqueeze(1), context.detach()], dim=1)
                pred_with_cls = predictor(
                    tokenized_size, context_with_cls, context_mask, target_mask, rope_seed=rope_seed
                )

                # Compute JEPA loss
                target = apply_mask(target_mask, teacher_output, fill_value=None)
                jepa_loss = F.mse_loss(pred, target) + F.mse_loss(pred_with_cls, target)
                train_loss.update(jepa_loss)
                train_loss_epoch.update(jepa_loss)

                # Compute Gram loss (if necessary)
                if computing_gram_loss:
                    assert gram_teacher_output is not None
                    gram_target = apply_mask(context_mask, gram_teacher_output, fill_value=None)
                    gram_loss = compute_gram_loss(context, gram_target, remove_neg=jepa_config.gram_remove_neg)
                else:
                    gram_loss = 0.0

            # Combine losses
            loss = jepa_loss + gram_loss

            # Backward
            assert not loss.isnan()
            loss.backward()
            assert_all_trainable_params_have_grad(modules, microbatch)
            logger.log(epoch, step, microbatch, loss=train_loss.compute().item()) if logger is not None else None

            # Optimizer update and teacher update
            if should_step_optimizer(microbatch, accumulate_grad_batches):
                if step < total_steps:
                    scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                current_momentum = get_momentum(step, total_steps, jepa_config.momentum)
                update_teacher(backbone, teacher, current_momentum)
                step += 1

            microbatch += 1
            desc = format_pbar_description(step, microbatch, epoch, loss=train_loss)
            pbar.set_description(desc)

        rank_zero_info(f"Epoch: {epoch}, Loss: {train_loss_epoch.compute():.4f}")
        train_loss_epoch.reset()

        # Save checkpoint
        if log_dir and is_rank_zero():
            save_checkpoint(
                path=log_dir / f"checkpoint.pt",
                backbone=backbone,
                predictor=predictor,
                teacher=teacher,
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
    if log_dir and is_rank_zero():
        st.save_file(
            {k: v for k, v in backbone.state_dict().items() if isinstance(v, torch.Tensor)},
            str(log_dir / f"backbone.safetensors"),
        )


def ddp_setup() -> None:
    """Initialize distributed training process group."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        logging.info("Initialized DDP")
    else:
        logging.info("DDP already initialized")


def ddp_cleanup() -> None:
    """Clean up distributed training process group."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logging.info("Cleaned up DDP")


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
    parser.add_argument("--highres-data", type=Path, default=None, help="Path to high resolution training data")
    parser.add_argument(
        "--highres-cutoff",
        type=int,
        nargs=2,
        default=(1024, 768),
        help="Resolution cutoff at which to switch to high resolution data",
    )
    return parser.parse_args()


def main(args: Namespace) -> None:
    if not (config_path := Path(args.config)).is_file():
        raise FileNotFoundError(config_path)
    config = yaml.full_load(config_path.read_text())

    # Extract instantiated dataclasses from config
    backbone_config = config["backbone"]
    jepa_config = config["jepa"]
    augmentation_config = config["augmentations"]
    optimizer_config = config["optimizer"]
    trainer_config = config["trainer"]
    assert isinstance(backbone_config, ViTConfig)
    assert isinstance(jepa_config, JEPAConfig)
    assert isinstance(augmentation_config, AugmentationConfig)
    assert isinstance(optimizer_config, OptimizerConfig)
    assert isinstance(trainer_config, TrainerConfig)
    if args.log_dir and not args.log_dir.is_dir():
        raise NotADirectoryError(args.log_dir)

    # Determine distributed training parameters
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        ddp_setup()

    # Instantiate other model elements and move to device
    backbone = backbone_config.instantiate()
    predictor = CrossAttentionPredictor(backbone, jepa_config.predictor_depth)
    wrapper = nn.ModuleDict({"backbone": backbone, "predictor": predictor}).cuda()

    # Wrap in DDP for distributed training
    if world_size > 1:
        wrapper = DDP(wrapper, device_ids=[local_rank])

    # Instantiate dataloaders
    keep_volume = False
    shuffle = True
    train_dataloader_fn = partial(
        get_train_dataloader,
        root=args.data,
        num_workers=trainer_config.num_workers,
        local_rank=local_rank,
        world_size=world_size,
        keep_volume=keep_volume,
        shuffle=shuffle,
        highres_data=args.highres_data,
        highres_cutoff=args.highres_cutoff,
    )
    train_dataloader = train_dataloader_fn(backbone.config.img_size, trainer_config.batch_size)

    # Instantiate optimizer and scheduler
    total_steps = calculate_total_steps(
        train_dataloader, trainer_config.num_epochs, trainer_config.accumulate_grad_batches
    )
    optimizer, scheduler = optimizer_config.instantiate(wrapper, total_steps=total_steps)

    # Create log subdirectory with timestamp
    if args.log_dir:
        log_dir = setup_logdir(args.log_dir, config_path, name=args.name)
    else:
        log_dir = None

    ignore_warnings()
    try:
        with tqdm.external_write_mode():
            logging.info(f"Starting training with local rank: {local_rank}, world size: {world_size}")
        train(
            cast(Any, wrapper),
            train_dataloader_fn,
            optimizer,
            scheduler,
            jepa_config,
            augmentation_config,
            trainer_config,
            log_dir,
        )
    finally:
        if world_size > 1:
            ddp_cleanup()


def entrypoint() -> None:
    args = parse_args()
    main(args)


if __name__ == "__main__":
    entrypoint()
