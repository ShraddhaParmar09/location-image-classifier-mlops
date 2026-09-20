## 6. Technical decision

At the end of Phase 1, we should be able to make decisions about:

# Dataset → ?

> ## 🛠️ Phase 1 Technical Decisions## 📁 Dataset → MIT Places365 (Custom 5-Class Balanced Subset)

* Decision: Download a curated subset of the [MIT Places365 dataset](http://places2.csail.mit.edu/).
* Reasoning: It contains standardized, high-quality, single-label environmental scenes. Choosing 5 specific, visually distinct classes (e.g., kitchen, playground, beach, mountain, street) with 1,000 images per class yields a perfectly balanced, manageable dataset of 5,000 total images.

# Problem type → ?

> ## 🔲 Problem Type → Single-Label, Multi-Class Image Classification

* Decision: Categorize an input image into exactly one of N predefined, mutually exclusive location categories.
* Reasoning: Your V1 scope specifies that the model maps an image to a single predicted location class, making this a textbook multi-class classification problem.

# Baseline → ?

> ## 📉 Baseline → Zero-Augmentation Custom CNN

* Decision: Build a simple 3-to-4 layer Convolutional Neural Network trained from scratch with no data augmentation.
* Reasoning: This establishes a "worst-case scenario" benchmark. It allows your research to clearly quantify how much performance improves when you introduce data augmentation, and how much further it leaps when using transfer learning.

# Model family → ?

> ## 🧬 Model Family → CNNs (Custom Architecture vs. Residual Networks)

* Decision:
* Track A (Scratch): Custom sequential CNN (Conv2D → BatchNorm → ReLU → MaxPool → Dense).
   * Track B (Transfer Learning): [PyTorch ResNet50](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html) or [EfficientNet-B0](https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html) pre-trained on ImageNet.
* Reasoning: ResNet50 and EfficientNet are industry standards for image feature extraction. They offer an ideal benchmark against a custom scratch network to demonstrate the power of pre-trained spatial weights.

# Evaluation metrics → ?

> ## 📊 Evaluation Metrics → Accuracy, F1-Score, Confusion Matrix, & Latency

* Decision:
* Classification Quality: Overall Accuracy, per-class Precision, Recall, and Macro F1-Score.
   * Error Analysis: A localized Confusion Matrix to identify which locations are easily confused (e.g., beach vs. mountain).
   * Computational Cost: Inference Latency (measured in milliseconds per image).
* Reasoning: Accuracy alone can hide flaws. F1-Score ensures the model performs well across all classes, while inference speed validates the 200ms deployment requirement.

# Tech stack → ?

> ## 💻 Tech Stack → Python, PyTorch, & FastAPI

* Decision:
* Core Language: Python 3.10+
   * Deep Learning Framework: PyTorch (leveraging torchvision for pre-trained models).
   * Data Pipeline: NumPy, Pandas, Pillow (PIL).
   * Deployment API Backend: FastAPI.
* Reasoning: PyTorch offers a highly Pythonic interface that is ideal for tracking custom layer weights during research. FastAPI is exceptionally lightweight and delivers the low-latency routing needed to meet your 200ms processing threshold.

# Deployment target → ?

> ## 🚀 Deployment Target → Dockerized Local Containerized Host or AWS EC2 Micro-Instance

* Decision: Wrap the FastAPI application and trained model weights into a standardized Docker Container. For testing, run it on a local machine or host it on a basic cloud instance (like an AWS EC2 t3.medium or Google Cloud Compute Engine).
* Reasoning: Packaging the application into a Docker container isolates your Python environment. This ensures that your local development testing mirrors cloud behavior exactly, making your deployment reproducible and scalable.

> 6. Technical Decisions → 🟡 Depends on Part 3

Your model strategy does NOT need to be thrown away.

I actually want to keep:

Track A

Custom CNN from scratch

Track B

Pretrained ResNet50 / EfficientNet

That is a useful experiment.

Your current technical decision already establishes those two tracks.

What changes is the dataset they train on.

So:

BEFORE

Places365
   ↓
5 scene classes
   ↓
Custom CNN
      VS
ResNet/EfficientNet

becomes:

AFTER

Location/landmark dataset
   ↓
5–10 location classes
   ↓
Custom CNN
      VS
ResNet/EfficientNet

The model comparison stays.

