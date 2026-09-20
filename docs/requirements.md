# 4. Define requirements

Write down:

# Functional requirements

> Here are the technical requirements for your location image classifier research and deployment project, structured to satisfy both the exploratory research track and the final user-facing app pipeline.
------------------------------
> 🏛️ Functional Requirements
Functional requirements define what the system must do from a user and system perspective.

* Image Upload: The system must accept user-uploaded image files (.jpg, .jpeg, .png) via a web application interface, API endpoint, or command-line interface.
* Feature Extraction: The system must process visual features using two distinct machine learning models: a Custom Convolutional Neural Network (CNN) trained from scratch and a pre-trained Transfer Learning model (e.g., ResNet50 or EfficientNet).
* Location Prediction: The machine learning model must accurately calculate a probability distribution over the known location classes.
* Output Generation: The system must parse the probability array and return the location class name with the highest confidence percentage.
* Switchable Engine: The system backend must allow researchers to toggle between the Custom CNN model engine and the Transfer Learning engine to compare predictions on the same input image.

------------------------------
# ⚙️ Non-Functional Requirements

Non-functional requirements specify how well the system must perform its tasks.

* Inference Latency: The system must return the predicted location class within 200 milliseconds for the Transfer Learning model and within 100 milliseconds for the Custom CNN model after receiving the upload.
* Scalability: The classification API must handle up to 20 concurrent image upload requests without memory exhaustion or request timeouts.
* Maintainability & Modular Design: The code repository must cleanly separate the data preprocessing pipelines, model architectures, training configurations, and deployment code.
* Reliability: The inference engine must run within a fault-tolerant container environment (like Docker) to prevent crashes if an corrupted image file is uploaded.

------------------------------
# 📥 Input & 📤 Output Specifications

📥 [Input Image File] ──> ⚙️ [ML Processing Pipeline] ──> 📤 [JSON Prediction Response]

## Input Specifications

* Data Format: Raw image files or base64-encoded strings (image/jpeg, image/png).
* Resolution Dimensions: Variable input size, but the system pipeline will automatically resize and normalize the image to a standardized layer canvas of 224 × 224 pixels (standard for ResNet/EfficientNet models) or 256 × 256 pixels.
* Color Channels: 3-channel RGB matrix format.

## Output Specifications

* Data Format: Structured JSON payload.
* Payload Fields: Must return the predicted_location string, the confidence_score float (value between 0.00 and 1.00), and an execution metric inference_time_ms.
* Example Output:

{
  "predicted_location": "Eiffel_Tower",
  "confidence_score": 0.942,
  "inference_time_ms": 142.5
}


------------------------------
## ⚠️ Constraints
Constraints are design limits or hard boundaries placed on the project.

* Dataset Scope: The system is constrained to classifying images into a fixed set of predefined location classes (e.g., 5 to 10 specific categories configured during the dataset research stage). It cannot predict dynamic locations outside its training classes.
* Memory Constraints: The model configurations must fit within the video memory boundaries of standard consumer hardware or free-tier cloud environments (e.g., maximum 12GB to 16GB VRAM bounds).
* Network & Bandwidth: Maximized file upload limitations must be capped at 10 MB per image to prevent network bottlenecks during client-server transport.

------------------------------
## 📊 Expected Performance
These targets will validate if your research models successfully learned the features.

| Metric          | Track A: Transfer Learning               | Track B: Custom CNN (From Scratch)              |
|-----------------|------------------------------------------|-------------------------------------------------|
| Target Accuracy | ≥ 85% on the validation and test splits. | ≥ 65% to 70% on the validation and test splits. |
----------------------------------------------------------------------------------------------------------------
| Overfitting     |Gap between Train and Test accuracy must  |Gap between Train and Test accuracy must be <12|  
| Threshold       |be < 5%.                                  |% (harder to manage from scratch).               |
----------------------------------------------------------------------------------------------------------------
| Top-5 Accuracy  | ≥ 95% classification presence in top     | ≥ 80% classification presence in top            |
|                 |   probabilities.                         | probabilities.                                  |
----------------------------------------------------------------------------------------------------------------

------------------------------
## 💻 Hardware & Compute Requirements## 

## 1. Training Environment (Research Phase)

* GPU (Recommended): NVIDIA T4 (available on free cloud environments like [Google Colab](https://colab.research.google.com/)), NVIDIA RTX 3060/4060, or Apple Silicon M-series (using MPS acceleration). Minimum 8 GB VRAM is needed to handle batch processing steps without hitting Out-Of-Memory (OOM) errors.
* System RAM: Minimum 16 GB RAM to ensure smooth execution of parallel data loading tasks (num_workers > 2).
* Storage Capacity: 20 GB to 50 GB of available solid-state storage (SSD) space for storing raw scene images, tensor caches, and saved model weight weights checkpoints (.pt or .h5 files).

## 2. Deployment Environment (Production Phase)

* Compute Engine: Single-core vCPU cloud instance or microservices server container.
* System Memory: 2 GB to 4 GB RAM (enough to pull the compiled static model weights file into host memory during initialization).
* Storage Capacity: Under 1 GB to save the deployment runtime dependencies, API server script, and the optimized model weights file.


> Your current requirements say:

"known location classes"

and:

"predicted_location"

Those are fine if Part 3 changes to a location dataset.

The bigger issue is this:

"5 to 10 specific categories"

and your current examples mix locations and scene categories.

So change the requirement to something like:

The system must classify an input image into one of the predefined geographic/location classes contained in the selected dataset.

And your output can remain:

{
  "predicted_location": "...",
  "confidence_score": 0.92,
  "inference_time_ms": 142.5
}

Your functional requirements already have the right general structure: image upload → feature extraction → probability distribution → predicted class.

So don't rewrite Part 4 from zero.

----------------------------------------------------------------------------------------------------------------
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

## 5. Define project scope

This is important.

# We decide exactly what V1 will do.

> For example:

        V1
        Image
        ↓
        Model
        ↓
        Predicted location/class

        And explicitly decide what we are not building in V1.

> Defining the project scope ensures you remain focused on your core research and development objectives. By explicitly defining boundaries, you prevent "scope creep" and guarantee a functional, testable V1.
------------------------------
## 🌐 V1 Architecture Pipeline
Your core minimum viable product (MVP) pipeline operates as follows:

[ User Input Image ] ──> [ Preprocessing Pipeline ] ──> [ ML Model Inference ] ──> [ Top Predicted Location Class ]

------------------------------
## 📦 In-Scope (What V1 WILL Do)
The core functionality of V1 is strictly focused on data preparation, model training, evaluation comparison, and basic endpoint serving.
## 1. Machine Learning & Core Models

* Dual-Track Implementation: Build, train, and save two distinct models: a Custom CNN built from scratch and a pre-trained Transfer Learning model (e.g., [ResNet50](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html) or [EfficientNet-B0](https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html)).
* Fixed Classification Targets: Limit the classifier to 5 to 10 predefined locations from your chosen dataset (e.g., Eiffel Tower, Taj Mahal, Grand Canyon) to keep training times manageable.
* Single-Label Output: Analyze visual features to predict and return exactly one primary location class name per input image along with its confidence percentage.

## 2. Input Processing & Server Handling

* Standardized Image Acceptance: Accept digital image file uploads (.jpg, .jpeg, .png) up to 10 MB in file size.
* Automated Preprocessing Pipe: Automatically handle scaling (resizing to 224 × 224 pixels), tensor type casting, and standard channel normalization.
* Basic Deployment Endpoint: Expose a minimal backend API endpoint (built using a framework like Flask or FastAPI) that accepts a file payload and outputs a structured JSON response containing the prediction variables.

------------------------------
## 🚫 Out-of-Scope (What V1 WILL NOT Do)
To ensure this project remains an effective, high-quality research study, the following features are explicitly excluded from V1. They can be revisited in V2 or future development phases.

* No Multi-Label Location Detection: The system will not attempt to identify multiple locations or break down multiple items within a single frame (e.g., it will not identify "Eiffel Tower" and "Louvre Museum" simultaneously if both happen to be in the background).
* No Real-Time Streaming/Video Feed Input: The model will only process static, pre-captured images. Real-time video stream chunking or live webcam frame analysis is completely excluded.
* No Dynamic Live Map Integrations: V1 will not integrate with live map navigation services (like Google Maps or Mapbox APIs) to plot coordinates, draw path routes, or reveal street-view visuals.
* No User Management System: There will be no user accounts, profile registration, history tracking dashboards, or databases set up to store previously uploaded user photos.
* No Advanced Frontend User Interface: The system will not feature a highly polished mobile app or complex multi-page web app. Interaction will be limited to a basic web upload form or direct API calls.
* No Multi-Modal Data Fusion: The model will classify locations solely based on visual pixels. It will not use secondary signals such as EXIF camera metadata, GPS coordinates embedded in the file, or user-provided text descriptions.


> 5. Scope / V1 → 🟡 Small change

Your current V1 says:

"Limit the classifier to 5 to 10 predefined locations"

That's actually good.

But then you give examples:

Eiffel Tower, Taj Mahal, Grand Canyon

Those are landmarks, whereas earlier you were discussing geographic image geolocation.

So you need to choose what location means in V1.

Option 1 — Landmark classification
Image → Eiffel Tower
Image → Taj Mahal
Image → Grand Canyon
Option 2 — City classification
Image → Paris
Image → Mumbai
Image → Tokyo
Option 3 — Country classification
Image → India
Image → France
Image → Japan
Option 4 — Geographic region classification
Image → Western India
Image → Southern Europe
Image → Southeast Asia

For a first project, landmark/location-class classification is much more manageable than trying to predict arbitrary GPS coordinates.

But we should choose this based on the dataset we find in Part 3, rather than deciding blindly.

Your existing V1 architecture itself is fine:

User Image → Preprocessing → ML Model → Top Predicted Location Class.


