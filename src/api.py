from io import BytesIO

import torch
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.inference import LocationImagePredictor


_predictor = None


def get_predictor():
    """Load the real model only when an endpoint needs it."""
    global _predictor

    if _predictor is None:
        _predictor = LocationImagePredictor(
            "experiments/inference_config.json"
        )

    return _predictor


app = FastAPI(
    title="OSV5M Location Image Classifier",
    description="Predict the country of an image using ResNet18.",
    version="1.0.0"
)


@app.get("/health")
def health(predictor=Depends(get_predictor)):
    return {
        "status": "healthy",
        "model": "OSV5M ResNet18",
        "device": str(predictor.device)
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    predictor=Depends(get_predictor)
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file."
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )

    try:
        with Image.open(BytesIO(contents)) as image_file:
            image_file.verify()

        image = Image.open(BytesIO(contents)).convert("RGB")

    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image."
        )

    image_tensor = predictor.transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(predictor.device)

    with torch.no_grad():
        outputs = predictor.model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

    scores, indices = torch.topk(
        probabilities,
        k=3,
        dim=1
    )

    predictions = [
        {
            "country": predictor.classes[index],
            "confidence": round(score * 100, 2)
        }
        for score, index in zip(
            scores[0].cpu().tolist(),
            indices[0].cpu().tolist()
        )
    ]

    return {
        "filename": file.filename,
        "predicted_country": predictions[0]["country"],
        "confidence": predictions[0]["confidence"],
        "top_predictions": predictions
    }