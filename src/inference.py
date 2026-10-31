import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from src.dataset import COUNTRY_CLASSES
from src.model import build_resnet18
from src.preprocessing import get_transforms


class LocationImagePredictor:
    def __init__(self, config_path):
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(config_path)

        config = json.loads(
            config_path.read_text(encoding="utf-8-sig")
        )

        checkpoint_path = Path(config["checkpoint_path"])

        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )

        self.classes = checkpoint["classes"]

        if self.classes != COUNTRY_CLASSES:
            raise ValueError("Checkpoint class mapping is incorrect.")

        if config["class_names"] != self.classes:
            raise ValueError(
                "Inference configuration and checkpoint classes differ."
            )

        if config["preprocessing"] != "fixed_resize":
            raise ValueError(
                "This predictor expects fixed_resize preprocessing."
            )

        _, self.transform = get_transforms(
            preprocessing=config["preprocessing"],
            image_size=config["image_size"],
            crop_size=None,
            random_horizontal_flip=False
        )

        self.model = build_resnet18(
            num_classes=config["num_classes"],
            pretrained=False,
            freeze_backbone=False
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(self.device)
        self.model.eval()

        print("Model loaded successfully.")
        print("Device:", self.device)
        print("Model:", config["model_name"])

    def predict(self, image_path, top_k=3):
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        image_tensor = self.transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)

        scores, indices = torch.topk(
            probabilities,
            k=min(top_k, len(self.classes)),
            dim=1
        )

        results = []

        for score, index in zip(
            scores[0].cpu().tolist(),
            indices[0].cpu().tolist()
        ):
            results.append({
                "country": self.classes[index],
                "confidence": round(score * 100, 2)
            })

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Predict the country of an image using OSV5M ResNet18."
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image."
    )

    parser.add_argument(
        "--config",
        default="experiments/inference_config.json",
        help="Path to inference configuration."
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top predictions to display."
    )

    args = parser.parse_args()

    predictor = LocationImagePredictor(args.config)
    predictions = predictor.predict(args.image, args.top_k)

    print("\nPREDICTIONS")
    print("-" * 40)

    for rank, prediction in enumerate(predictions, start=1):
        print(
            f"{rank}. {prediction['country']} "
            f"- {prediction['confidence']:.2f}%"
        )


if __name__ == "__main__":
    main()
