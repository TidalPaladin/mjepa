import math
import os
import warnings
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
from torchmetrics.wrappers import Running
from torchvision.datasets import CIFAR10
from torchvision.transforms.v2 import (
    ColorJitter,
    Compose,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomRotation,
    RandomVerticalFlip,
    ToDtype,
    ToImage,
)
from tqdm import tqdm
from vit import ViT, ViTConfig
from vit.tokens import apply_mask

from mjepa.jepa import CrossAttentionPredictor, generate_masks
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
)


NUM_CLASSES: Final[int] = 10
UNKNOWN_LABEL: Final[int] = -1
WINDOW: Final[int] = 5

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class Wrapper(nn.Module):
    def __init__(self, backbone: ViT, predictor: nn.Module, probe: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.predictor = predictor
        self.probe = probe


def get_train_transforms() -> Compose:
    return Compose(
        [
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomResizedCrop(size=(32, 32), scale=(0.75, 1.0), ratio=(0.75, 1.33)),
            RandomRotation(degrees=15),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            RandomGrayscale(p=0.1),
            ToImage(),
            ToDtype(torch.float32, scale=True),
        ]
    )


def get_val_transforms() -> Compose:
    return Compose(
        [
            ToImage(),
            ToDtype(torch.float32, scale=True),
        ]
    )


def get_train_dataloader(root: Path, batch_size: int, num_workers: int, local_rank: int, world_size: int) -> DataLoader:
    transforms = get_train_transforms()
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


def get_val_dataloader(root: Path, batch_size: int, num_workers: int, local_rank: int, world_size: int) -> DataLoader:
    transforms = get_val_transforms()
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
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
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
    probe: nn.Module = modules["probe"]
    rank_zero_info(f"Backbone params: {format_large_number(count_parameters(backbone))}")
    rank_zero_info(f"Predictor params: {format_large_number(count_parameters(predictor))}")
    rank_zero_info(f"Probe params: {format_large_number(count_parameters(probe))}")

    microbatch = (last_epoch + 1) * len(train_dataloader)
    step = microbatch // trainer_config.accumulate_grad_batches
    total_steps = calculate_total_steps(
        train_dataloader, trainer_config.num_epochs, trainer_config.accumulate_grad_batches
    )
    rank_zero_info(f"Training for {trainer_config.num_epochs} epochs = {total_steps} steps")
    rank_zero_info(
        f"Batch size: {trainer_config.batch_size}, Microbatch accumulation: {trainer_config.accumulate_grad_batches}"
    )

    # Metric setup
    train_loss = tm.RunningMean(window=WINDOW).cuda()
    train_acc = Running(tm.Accuracy(task="multiclass", num_classes=NUM_CLASSES), window=WINDOW).cuda()
    val_acc = tm.Accuracy(task="multiclass", num_classes=NUM_CLASSES).cuda()

    img: Tensor
    label: Tensor
    for epoch in range(last_epoch + 1, trainer_config.num_epochs):
        modules.train()
        desc = format_pbar_description(step, microbatch, epoch, loss=train_loss, acc=train_acc)
        pbar = tqdm(train_dataloader, desc=desc, disable=not is_rank_zero(), leave=False)
        for img, label in pbar:
            img = img.cuda()
            label = label.cuda()

            # Fold the image to create the target
            hp, wp = backbone.config.patch_size
            target = rearrange(img, "b c (h hp) (w wp) -> b (h w) (hp wp c)", hp=hp, wp=wp)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Masked forward through backbone and predictor
                context_mask, target_mask = generate_masks(
                    backbone, img, mae_config.context_ratio, mae_config.target_ratio, mae_config.scale
                )
                context, _, _ = cast(Tuple[Tensor, Tensor | None, Tensor | None], backbone(img, mask=context_mask))
                tokenized_size = backbone.stem.tokenized_size(img.shape[-2:])
                pred: Tensor = predictor(tokenized_size, context, target_mask)

                # Compute MAE loss
                target = apply_mask(target_mask, target, fill_value=None)
                mae_loss = F.l1_loss(pred, target)
                train_loss.update(mae_loss)

                # Compute linear probe loss
                probe_pred = probe(context.detach().mean(1))
                probe_loss = F.cross_entropy(probe_pred, label)

                # Combine losses
                loss = mae_loss + probe_loss

            with torch.no_grad():
                train_acc.update(probe_pred, label)

            # Backward
            assert not loss.isnan()
            loss.backward()
            assert_all_trainable_params_have_grad(modules, microbatch)
            microbatch += 1

            # Optimizer update and teacher update
            if should_step_optimizer(microbatch, trainer_config.accumulate_grad_batches):
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
                img = img.cuda()
                label = label.cuda()
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    full_output, _, _ = cast(Tuple[Tensor, Tensor | None, Tensor | None], backbone(img))
                    probe_pred = probe(full_output.mean(1))
                    val_acc.update(probe_pred, label)

            # Validation epoch end
            rank_zero_info(f"Epoch: {epoch}, Val Acc: {val_acc.compute():.4f}")

        # Save checkpoint
        if log_dir:
            save_checkpoint(
                path=log_dir / f"checkpoint.pt",
                backbone=backbone,
                predictor=predictor,
                probe=probe,
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


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("config", type=Path, help="Path to YAML configuration file")
    parser.add_argument("data", type=Path)
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
    predictor = CrossAttentionPredictor(backbone, mae_config.predictor_depth, out_dim)
    probe = backbone.create_head(NUM_CLASSES, bias=True)
    wrapper = nn.ModuleDict({"backbone": backbone, "predictor": predictor, "probe": probe}).cuda()
    nn.init.trunc_normal_(predictor.predictor_proj[0].weight, std=0.001)
    nn.init.constant_(predictor.predictor_proj[0].bias, 0.5)

    # Wrap in DDP for distributed training
    if world_size > 1:
        ddp_setup()
        wrapper = DDP(wrapper, device_ids=[local_rank])

    # Instantiate dataloaders
    train_dataloader = get_train_dataloader(
        args.data, trainer_config.batch_size, trainer_config.num_workers, local_rank, world_size
    )
    val_dataloader = get_val_dataloader(
        args.data, trainer_config.batch_size, trainer_config.num_workers, local_rank, world_size
    )

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
    train(wrapper, train_dataloader, val_dataloader, optimizer, scheduler, mae_config, trainer_config, log_dir)


def entrypoint() -> None:
    args = parse_args()
    main(args)


if __name__ == "__main__":
    entrypoint()
