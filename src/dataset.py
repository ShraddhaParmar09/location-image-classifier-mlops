from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


# Keep the exact alphabetical class order from the original Colab notebook.
COUNTRY_CLASSES = [
    "AU", "BR", "DE", "FR", "GB",
    "IT", "JP", "RU", "SE", "US"
]

CLASS_TO_IDX = {
    country: idx
    for idx, country in enumerate(COUNTRY_CLASSES)
}

IDX_TO_CLASS = {
    idx: country
    for country, idx in CLASS_TO_IDX.items()
}


class OSV5MDataset(Dataset):
    def __init__(self, metadata_csv, image_root, transform=None):
        self.metadata_csv = Path(metadata_csv)
        self.image_root = Path(image_root)
        self.transform = transform

        self.df = pd.read_csv(self.metadata_csv)

        if "id" not in self.df.columns or "country" not in self.df.columns:
            raise ValueError(
                "Metadata CSV must contain 'id' and 'country' columns."
            )

        unknown_classes = (
            set(self.df["country"].dropna().unique())
            - set(COUNTRY_CLASSES)
        )

        if unknown_classes:
            raise ValueError(
                f"Unknown country labels found: {sorted(unknown_classes)}"
            )

        if self.df["country"].isna().any():
            raise ValueError("Country labels contain missing values.")

        self.classes = COUNTRY_CLASSES.copy()
        self.class_to_idx = CLASS_TO_IDX.copy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_id = str(row["id"])
        country = row["country"]

        image_path = (
            self.image_root / country / f"{image_id}.jpg"
        )

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        label = self.class_to_idx[country]

        if self.transform:
            image = self.transform(image)

        return image, label
