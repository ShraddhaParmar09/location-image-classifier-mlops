from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(preprocessing, image_size=224, crop_size=None,
                   random_horizontal_flip=False):
    """
    Return training and validation transforms for an experiment.
    """

    normalize = transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )

    if preprocessing == "fixed_resize":
        train_transforms = [
            transforms.Resize((224, 224))
        ]

        if random_horizontal_flip:
            train_transforms.append(
                transforms.RandomHorizontalFlip()
            )

        train_transforms.extend([
            transforms.ToTensor(),
            normalize
        ])

        val_transforms = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ]

    elif preprocessing == "aspect_ratio_resize_crop":
        if crop_size is None:
            raise ValueError(
                "crop_size is required for aspect-ratio preprocessing."
            )

        train_transforms = [
            transforms.Resize(image_size),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            normalize
        ]

        val_transforms = [
            transforms.Resize(image_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize
        ]

    else:
        raise ValueError(
            f"Unsupported preprocessing method: {preprocessing}"
        )

    return (
        transforms.Compose(train_transforms),
        transforms.Compose(val_transforms)
    )
