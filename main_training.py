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
from utils.config import Config


def get_args():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument(
        "--config_path", 
        type=Path,
        help="Path to the YAML configuration file",
        required=True
    )
    # Data
    parser.add_argument(
        "--data_path", 
        type=Path, 
        help="Path to the dataset",
        required=True
    )
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    # Schedule
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--warmup_epochs", type=int)
    # Optimizer
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)

    args = parser.parse_args()
    
    return args


def main(args, config):
    # Basic reproducibility settings
    random.seed(config["seed"])  # If Python random is used
    np.random.seed(config["seed"])  # If NumPy random is used
    torch.manual_seed(config["seed"])
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
    )

    val_set = CIFAR100(
        root=args.data_path,
        train=False,
        transform=val_transforms,
        target_transform=torch.tensor,
    )

    # Dataloaders
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=True
    )

    val_loader = DataLoader(
        dataset=val_set,
        batch_size=1,
        shuffle=False,
        num_workers=config["num_workers"],
        drop_last=False
    )

    # Model
    model = ExampleModel(
        feature_size=16,
        num_stages=4,
        num_classes=100
    )
    model = model.to(config["device"])

    # Optimizer, Scheduler, Loss, Tracking
    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    scheduler = lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        total_steps=config["num_epochs"]*len(train_loader),
        max_lr=config["lr"],
        pct_start=config["warmup_epochs"]/config["num_epochs"],
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
        metric_names=["val_acc"],
        use_tensorboard=True
    )
    tracker.log_training_start()
    for epoch in range(config["num_epochs"]):
        print(f"Epoch {epoch}")
        train_loss = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            scheduler,
            config["device"]
        )

        val_loss, val_acc = validate_one_epoch(
            model,
            val_loader,
            loss_fn,
            config["device"]
        )

        is_new_best = tracker.update_metric("val_acc", val_acc)
        if is_new_best:
            torch.save(
                model.state_dict(),
                args.model_weights_path
            )

        tracker.log_scalar(name="train_losses", value=train_loss, index=epoch)
        tracker.log_scalar(name="val_losses", value=val_loss, index=epoch)
        tracker.log_scalar(name="val_accs", value=val_acc, index=epoch)
    tracker.log_training_end()
    
    tracker.log_hparams_and_metrics(
        hparams=dict(
            num_epochs=config["num_epochs"],
            lr=config["lr"],
            batch_size=config["batch_size"],
            wd=config["weight_decay"]
        )
    )
    tracker.finalize_run(
        save_logs=True,
        print_metrics=True
    )
    config.save(args.log_dir/args.config_path.name)


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
    run_start_time = datetime.strftime(datetime.now(), "%Y%m%d%H%M")
    log_dir = project_dir / "log" / run_start_time
    log_dir.mkdir(parents=True)
    args.log_dir = log_dir

    config = Config()
    config.load(args.config_path)
    config.update(args)
    print("---------- Config ----------")
    print(config)
    print("----------------------------")

    main(args, config)
