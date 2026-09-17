"""Keep Member 02 preprocessing; add mild geometry for E3/E4 training only."""
from torchvision import transforms
from torchvision.transforms import InterpolationMode

AUGMENTATION = {
    "degrees": 5.0, "translate": (0.02, 0.02),
    "interpolation": "bilinear", "fill": 0,
    "horizontal_flip": False, "vertical_flip": False,
}


def build_transform(split, augment=False, image_size=224):
    """Even augment=True never enables random transforms for validation/test."""
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    steps = [transforms.Resize((image_size, image_size))]
    if split == "train" and augment:
        steps.append(transforms.RandomAffine(
            degrees=AUGMENTATION["degrees"],
            translate=AUGMENTATION["translate"],
            interpolation=InterpolationMode.BILINEAR,
            fill=AUGMENTATION["fill"],
        ))
    steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    return transforms.Compose(steps)
