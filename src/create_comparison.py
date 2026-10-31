import csv
from pathlib import Path

experiments = {
    "Experiment 1 - Frozen ResNet18":
        "experiments/experiment1_frozen_resnet18_metrics.csv",

    "Experiment 2 - Full Fine-tuning":
        "experiments/experiment2_full_finetuning_metrics.csv",

    "Experiment 3 - Aspect Ratio":
        "experiments/experiment3_aspect_ratio_metrics.csv",
}

output_path = Path("reports/experiment_comparison.csv")

if output_path.exists():
    raise FileExistsError(
        f"{output_path} already exists. Nothing was overwritten."
    )

results = []

for experiment_name, metrics_path in experiments.items():
    with open(metrics_path, "r", encoding="utf-8") as file:
        records = list(csv.DictReader(file))

    if not records:
        raise ValueError(f"No metrics found in {metrics_path}")

    best = max(records, key=lambda row: float(row["val_accuracy"]))

    results.append({
        "experiment": experiment_name,
        "best_epoch": int(best["epoch"]),
        "train_loss": float(best["train_loss"]),
        "train_accuracy": float(best["train_accuracy"]),
        "val_loss": float(best["val_loss"]),
        "val_accuracy": float(best["val_accuracy"]),
    })

results.sort(
    key=lambda row: row["val_accuracy"],
    reverse=True
)

output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("x", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "experiment",
            "best_epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
        ]
    )
    writer.writeheader()
    writer.writerows(results)

print("\nEXPERIMENT COMPARISON")
print("-" * 75)

for index, result in enumerate(results, start=1):
    print(
        f"{index}. {result['experiment']} | "
        f"Best Epoch: {result['best_epoch']} | "
        f"Validation Accuracy: {result['val_accuracy']:.2f}%"
    )

print("\nBest experiment:", results[0]["experiment"])
print(f"Comparison report saved: {output_path}")
