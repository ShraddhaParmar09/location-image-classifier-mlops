import argparse
import csv
import json
from pathlib import Path

import torch

from src.dataset import OSV5MDataset, COUNTRY_CLASSES
from src.model import build_resnet18
from src.preprocessing import get_transforms
from src.train import (
    create_data_loaders,
    set_seed,
    train_model
)


def run_experiment(config_path):
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(config_path)

    config = json.loads(config_path.read_text(encoding="utf-8"))

    name = config["experiment_name"]

    checkpoint_path = Path("checkpoints") / f"{name}_best.pth"
    metrics_path = Path("experiments") / f"{name}_metrics.csv"

    # Never overwrite existing experiment results.
    existing = [
        path for path in (checkpoint_path, metrics_path)
        if path.exists()
    ]

    if existing:
        raise FileExistsError(
            "Experiment outputs already exist. Nothing was overwritten: "
            + ", ".join(str(path) for path in existing)
        )

    set_seed(config["random_seed"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nExperiment: {name}")
    print(f"Device: {device}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Epochs: {config['epochs']}")

    train_transform, val_transform = get_transforms(
        preprocessing=config["preprocessing"],
        image_size=config["image_size"],
        crop_size=config["crop_size"],
        random_horizontal_flip=config["random_horizontal_flip"]
    )

    train_dataset = OSV5MDataset(
        "data/metadata/train_split.csv",
        "data/images",
        transform=train_transform
    )

    val_dataset = OSV5MDataset(
        "data/metadata/validation_split.csv",
        "data/images",
        transform=val_transform
    )

    if train_dataset.classes != COUNTRY_CLASSES:
        raise ValueError("Training class mapping does not match.")

    if val_dataset.classes != COUNTRY_CLASSES:
        raise ValueError("Validation class mapping does not match.")

    train_loader, val_loader = create_data_loaders(
        train_dataset,
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"]
    )

    model = build_resnet18(
        num_classes=config["num_classes"],
        pretrained=config["pretrained"],
        freeze_backbone=config["freeze_backbone"]
    )

    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config["learning_rate"],
        epochs=config["epochs"],
        device=device
    )

    checkpoint = {
        "model_state_dict": result["best_state_dict"],
        "architecture": config["architecture"],
        "num_classes": config["num_classes"],
        "classes": COUNTRY_CLASSES,
        "class_to_idx": {
            country: idx
            for idx, country in enumerate(COUNTRY_CLASSES)
        },
        "best_epoch": result["best_epoch"],
        "best_val_accuracy": result["best_val_accuracy"],
        "config": config
    }

    # Recheck before writing, in case files appeared during training.
    if checkpoint_path.exists() or metrics_path.exists():
        raise FileExistsError(
            "An output file appeared during training. "
            "No existing file will be overwritten."
        )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, checkpoint_path)

    with metrics_path.open(
        "x", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy"
            ]
        )
        writer.writeheader()
        writer.writerows(result["history"])

    print("\nExperiment completed.")
    print("Best epoch:", result["best_epoch"])
    print("Best validation accuracy:", result["best_val_accuracy"])
    print("Checkpoint:", checkpoint_path)
    print("Metrics:", metrics_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the experiment JSON configuration."
    )

    args = parser.parse_args()
    run_experiment(args.config)
