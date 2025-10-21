from pathlib import Path
import argparse
from datetime import datetime
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import optim
from torch.optim import lr_scheduler
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, ToImage, ToDtype, Normalize

from models.basic_model import ExampleModel

from engine_training import train_one_epoch, validate_one_epoch
from utils.tracker import ExperimentTracker


def get_args():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    # Data
    parser.add_argument("--data_path", type=str, help="Path of the training and validation data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train_set_ratio", type=float, default=1.0)
    # Schedule
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    # Optimizer
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    args = parser.parse_args()
    
    return args


def main(args):
    # Basic reproducibility settings
    random.seed(args.seed)  # If Python random is used
    np.random.seed(args.seed)  # If NumPy random is used
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    # Sometimes using deterministic algorithms may be difficult
    #torch.use_deterministic_algorithms(True)

    # Transforms
    train_transforms = Compose(
        [
            ToImage(),
            ToDtype(dtype=torch.float32, scale=True),
            Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ]
    )
    
    val_transforms = Compose(
        [
            ToImage(),
            ToDtype(dtype=torch.float32, scale=True),
            Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ]
    )

    # Datasets
    train_set = CIFAR100(
        root=args.data_path,
        train=True,
        transform=train_transforms,
        target_transform=torch.tensor,
        download=True,
    )
    # To make faster test runs of the training scipt
    if args.train_set_ratio < 1.0:
        train_indices = torch.randperm(len(train_set))
        train_indices = train_indices[:int(args.train_set_ratio*len(train_set))]

        train_set = torch.utils.data.Subset(
            train_set,
            indices=train_indices
        )

    val_set = CIFAR100(
        root=args.data_path,
        train=False,
        transform=val_transforms,
        target_transform=torch.tensor,
        download=True,
    )

    # Dataloaders
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )

    val_loader = DataLoader(
        dataset=val_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )

    # Model
    model = ExampleModel(
        feature_size=16,
        num_stages=4,
        num_classes=100
    )

    # Optimizer, Scheduler, Loss, Tracking
    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        total_steps=args.num_epochs*len(train_loader),
        max_lr=args.lr,
        pct_start=args.warmup_epochs/args.num_epochs,
        div_factor=1e3,
        final_div_factor=1e4
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    # Main loop
    tracker = ExperimentTracker(
        log_dir=args.log_dir,
        scalar_names=[
            "train_losses",
            "val_losses",
            "val_accs"
        ],
        metric_names=["val_acc"]
    )
    tracker.log_training_start()
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch}")
        train_loss = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            scheduler,
            args.device
        )

        val_loss, val_acc = validate_one_epoch(
            model,
            val_loader,
            loss_fn,
            args.device
        )

        is_new_best = tracker.update_metric("val_acc", val_acc)
        if is_new_best:
            torch.save(
                model.state_dict(),
                args.model_weights_path
            )

        tracker.log_scalar(name="train_losses", value=train_loss)
        tracker.log_scalar(name="val_losses", value=val_loss)
        tracker.log_scalar(name="val_accs", value=val_acc)
    tracker.log_training_end()
    
    tracker.save_logs()

    tracker.plot_figure(
        scalar_names=[
            "train_losses",
            "val_losses"
        ],
        filename=args.log_dir / "train_val_losses.png",
        title="Train/Val Losses"
    )

    tracker.plot_figure(
        scalar_names=[
            "val_accs",
        ],
        filename=args.log_dir / "val_accs.png",
        title="Val Accuracy"
    )

    tracker.print_best_metrics()


if __name__ == "__main__":
    # Parse command line arguments
    args = get_args()

    args.data_path = Path(args.data_path)
    args.data_path.mkdir(exist_ok=True)
    
    project_dir = Path(__file__).parent
    # Checkpoint directory
    ckpt_dir = project_dir / "ckpt"
    ckpt_dir.mkdir(exist_ok=True)
    args.ckpt_dir = ckpt_dir
    args.model_weights_path = args.ckpt_dir / "model.pth"
    
    # Log directory
    log_dir = project_dir / "log" / datetime.strftime(datetime.now(), "%Y%m%d%H%M")
    log_dir.mkdir(parents=True)
    args.log_dir = log_dir

    main(args)
