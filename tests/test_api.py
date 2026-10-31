from io import BytesIO

import torch
from fastapi.testclient import TestClient
from PIL import Image

from src.api import app, get_predictor
from src.dataset import COUNTRY_CLASSES


class MockModel:
    def __call__(self, image_tensor):
        return torch.tensor(
            [[5.0, 4.0, 3.0, 2.0, 1.0,
              0.5, 0.4, 0.3, 0.2, 0.1]]
        )


class MockPredictor:
    def __init__(self):
        self.device = torch.device("cpu")
        self.classes = COUNTRY_CLASSES
        self.model = MockModel()
        self.transform = lambda image: torch.zeros(3, 224, 224)


mock_predictor = MockPredictor()


def override_predictor():
    return mock_predictor


app.dependency_overrides[get_predictor] = override_predictor

client = TestClient(app)


def create_test_image():
    image = Image.new("RGB", (100, 100), color="blue")
    image_bytes = BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes.seek(0)
    return image_bytes


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model"] == "OSV5M ResNet18"
    assert data["device"] == "cpu"


def test_predict_valid_image():
    image = create_test_image()

    response = client.post(
        "/predict",
        files={
            "file": (
                "test.jpg",
                image,
                "image/jpeg"
            )
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["predicted_country"] in COUNTRY_CLASSES
    assert 0 <= data["confidence"] <= 100
    assert len(data["top_predictions"]) == 3


def test_predict_invalid_file():
    response = client.post(
        "/predict",
        files={
            "file": (
                "invalid.txt",
                b"This is not an image.",
                "text/plain"
            )
        }
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Please upload a valid image file."
    )


def test_predict_corrupted_image():
    response = client.post(
        "/predict",
        files={
            "file": (
                "corrupted.jpg",
                b"This is not a real JPEG image.",
                "image/jpeg"
            )
        }
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Invalid or corrupted image."
    )