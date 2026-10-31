import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes=10, pretrained=True, freeze_backbone=False):
    """
    Build an ImageNet-pretrained ResNet18 for country classification.

    Args:
        num_classes: Number of output country classes.
        pretrained: Whether to load ImageNet pretrained weights.
        freeze_backbone: Freeze all layers except the final classifier.

    Returns:
        Configured ResNet18 model.
    """

    weights = (
        models.ResNet18_Weights.DEFAULT
        if pretrained
        else None
    )

    model = models.resnet18(weights=weights)

    # Replace the original 1000-class ImageNet classifier.
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

        # Keep the new classification layer trainable.
        for parameter in model.fc.parameters():
            parameter.requires_grad = True

    return model


def count_parameters(model):
    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return {
        "total_parameters": total,
        "trainable_parameters": trainable
    }
