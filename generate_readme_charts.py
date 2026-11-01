import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

BG = "#080D1B"
PANEL = "#111A2E"
TEXT = "#E5EAF5"
MUTED = "#9CAAC2"
VIOLET = "#9B8CFF"
CYAN = "#38BDF8"
PINK = "#E879F9"
GRID = "#29354D"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "savefig.facecolor": BG,
    "text.color": TEXT,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.family": "DejaVu Sans",
})

def finish(fig, filename):
    fig.tight_layout(pad=2)
    fig.savefig(REPORTS / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Created: reports/{filename}")

# 1. Confusion matrix
with open(REPORTS / "confusion_matrix.csv", newline="", encoding="utf-8") as f:
    rows = list(csv.reader(f))

labels = rows[0][1:]
matrix = np.array([[int(x) for x in row[1:]] for row in rows[1:]])

fig, ax = plt.subplots(figsize=(11, 8))
im = ax.imshow(matrix, cmap="magma")

ax.set_title("VALIDATION CONFUSION MATRIX",
             fontsize=18, fontweight="bold", pad=22)
ax.set_xlabel("PREDICTED COUNTRY", labelpad=12)
ax.set_ylabel("ACTUAL COUNTRY", labelpad=12)
ax.set_xticks(range(len(labels)), labels)
ax.set_yticks(range(len(labels)), labels)

for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        value = matrix[i, j]
        ax.text(j, i, str(value), ha="center", va="center",
                fontsize=9, fontweight="bold",
                color="white" if value >= 10 else TEXT)

cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
cbar.ax.tick_params(colors=MUTED)
finish(fig, "confusion-matrix.png")

# 2. Experiment comparison
with open(REPORTS / "experiment_comparison.csv",
          newline="", encoding="utf-8") as f:
    experiments = list(csv.DictReader(f))

experiments.sort(key=lambda row: int(row["experiment"].split()[1]))
names = [
    "EXP 1\nFrozen ResNet18",
    "EXP 2\nFull Fine-tuning",
    "EXP 3\nAspect Ratio",
]
values = [float(row["val_accuracy"]) for row in experiments]

fig, ax = plt.subplots(figsize=(10, 5.8))
bars = ax.bar(names, values, width=0.55,
              color=["#64748B", VIOLET, CYAN],
              edgecolor="none")

ax.set_title("MODEL EXPERIMENT COMPARISON",
             fontsize=18, fontweight="bold", pad=22)
ax.set_ylabel("VALIDATION ACCURACY (%)")
ax.set_ylim(0, 100)
ax.grid(axis="y", color=GRID, linestyle="--", alpha=0.7)
ax.set_axisbelow(True)
ax.spines[["top", "right", "left"]].set_visible(False)

for bar, value in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            value + 2, f"{value:.2f}%",
            ha="center", fontsize=12, fontweight="bold",
            color=TEXT)

finish(fig, "experiment-comparison.png")

# 3. Per-country classification metrics
with open(REPORTS / "classification_report.csv",
          newline="", encoding="utf-8") as f:
    report = list(csv.DictReader(f))

report = [row for row in report if row["country"] in labels]
report.sort(key=lambda row: labels.index(row["country"]))

countries = [row["country"] for row in report]
precision = [float(row["precision"]) for row in report]
recall = [float(row["recall"]) for row in report]
f1 = [float(row["f1_score"]) for row in report]

x = np.arange(len(countries))
width = 0.24

fig, ax = plt.subplots(figsize=(12, 6.5))
ax.bar(x - width, precision, width, label="Precision", color=VIOLET)
ax.bar(x, recall, width, label="Recall", color=CYAN)
ax.bar(x + width, f1, width, label="F1-score", color=PINK)

ax.set_title("PER-COUNTRY CLASSIFICATION PERFORMANCE",
             fontsize=17, fontweight="bold", pad=22)
ax.set_ylabel("METRIC SCORE")
ax.set_xticks(x, countries)
ax.set_ylim(0, 1.12)
ax.grid(axis="y", color=GRID, linestyle="--", alpha=0.7)
ax.set_axisbelow(True)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.legend(frameon=False, ncol=3, loc="upper center")

finish(fig, "classification-metrics.png")

print("\nAll three README charts generated successfully.")
