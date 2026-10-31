import csv
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.dataset import OSV5MDataset, COUNTRY_CLASSES
from src.model import build_resnet18
from src.preprocessing import get_transforms


def evaluate_model():
    checkpoint_path = Path(
        "checkpoints/experiment2_full_finetuning_best.pth"
    )

    classification_path = Path(
        "reports/classification_report.csv"
    )

    confusion_path = Path(
        "reports/confusion_matrix.csv"
    )

    for path in [
        classification_path,
        confusion_path
    ]:
        if path.exists():
            raise FileExistsError(
                f"{path} already exists. Nothing was overwritten."
            )

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    config = checkpoint["config"]

    _, val_transform = get_transforms(
        preprocessing=config["preprocessing"],
        image_size=config["image_size"],
        crop_size=config["crop_size"],
        random_horizontal_flip=config["random_horizontal_flip"]
    )

    val_dataset = OSV5MDataset(
        "data/metadata/validation_split.csv",
        "data/images",
        transform=val_transform
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0
    )

    model = build_resnet18(
        num_classes=config["num_classes"],
        pretrained=False,
        freeze_backbone=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []

    print("Evaluating Experiment 2...")
    print("Device:", device)
    print("Validation images:", len(val_dataset))

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(labels.tolist())

    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(COUNTRY_CLASSES))),
        target_names=COUNTRY_CLASSES,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(len(COUNTRY_CLASSES)))
    )

    classification_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with classification_path.open(
        "x", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "country",
                "precision",
                "recall",
                "f1_score",
                "support"
            ]
        )

        writer.writeheader()

        for country in COUNTRY_CLASSES:
            writer.writerow({
                "country": country,
                "precision": report[country]["precision"],
                "recall": report[country]["recall"],
                "f1_score": report[country]["f1-score"],
                "support": report[country]["support"]
            })

        for average in ["macro avg", "weighted avg"]:
            writer.writerow({
                "country": average,
                "precision": report[average]["precision"],
                "recall": report[average]["recall"],
                "f1_score": report[average]["f1-score"],
                "support": report[average]["support"]
            })

    with confusion_path.open(
        "x", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["actual/predicted"] + COUNTRY_CLASSES)

        for country, row in zip(COUNTRY_CLASSES, matrix):
            writer.writerow([country] + row.tolist())

    accuracy = (
        sum(
            actual == predicted
            for actual, predicted in zip(
                all_labels, all_predictions
            )
        )
        / len(all_labels)
    ) * 100

    print("\nEVALUATION RESULTS")
    print("-" * 50)
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Macro F1-score: {report['macro avg']['f1-score']:.4f}")
    print(
        f"Weighted F1-score: "
        f"{report['weighted avg']['f1-score']:.4f}"
    )

    print("\nCLASSIFICATION REPORT")
    for country in COUNTRY_CLASSES:
        print(
            f"{country}: "
            f"Precision={report[country]['precision']:.3f}, "
            f"Recall={report[country]['recall']:.3f}, "
            f"F1={report[country]['f1-score']:.3f}"
        )

    print("\nConfusion matrix:")
    print(matrix)

    print("\nSaved reports:")
    print(classification_path)
    print(confusion_path)


if __name__ == "__main__":
    evaluate_model()
