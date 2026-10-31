# OSV5M Location Image Classifier



A computer vision project that predicts the country associated with an image using a fine-tuned ResNet18 model.



## Project Overview



- Dataset: OSV5M

- Pilot dataset: 1,000 images

- Countries: 10

- Training images: 803

- Validation images: 197

- Model: ImageNet-pretrained ResNet18



## Tech Stack



Python, PyTorch, Torchvision, Pandas, NumPy, Scikit-learn, FastAPI, Uvicorn and Pytest.



## Experiments



| Experiment | Approach | Best Validation Accuracy |

|---|---|---:|

| Experiment 1 | Frozen ResNet18 | 62.94% |

| Experiment 2 | Full fine-tuning | 81.73% |

| Experiment 3 | Aspect-ratio preprocessing | 80.20% |



Best local model: Experiment 2.



## Evaluation



- Accuracy: 81.73%

- Macro F1-score: 0.7980

- Weighted F1-score: 0.8107



Detailed results are in the `reports/` directory.



## API



Start the application:



```powershell

python -m uvicorn src.api:app --host 127.0.0.1 --port 8000

```



Endpoints:

- `GET /health`

- `POST /predict`



Interactive documentation: http://127.0.0.1:8000/docs



## Automated Testing



Run:



```powershell

python -m pytest tests/test_api.py -v

```



Latest result: 4 tests passed using a mock predictor. Real-model inference was validated separately.



## Installation



```powershell

python -m venv .venv

.\.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt

```



Install a compatible CUDA-enabled PyTorch build separately for GPU acceleration.



## Dataset



Source: https://huggingface.co/datasets/osv5m/osv5m



The dataset and model checkpoints are excluded from Git because of their size.



## Limitations



- This is a 1,000-image pilot, not the complete OSV5M dataset.

- Only 10 countries are classified.

- Validation contains 197 images.

- No independent test-set evaluation has been completed.

- Confidence scores are not calibrated probabilities.



## Future Improvements



- Expand dataset and country coverage.

- Evaluate on an independent test set.

- Improve model generalization.

- Add experiment tracking.

- Deploy and monitor the API.



## Author



AI/ML Engineering | Computer Vision | Deep Learning | MLOps
