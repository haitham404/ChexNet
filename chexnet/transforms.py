"""Image transforms for CheX-ray14 training and evaluation."""

import torchvision.transforms as transforms

_normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])


def get_transform_train(img_crop: int = 224):
    """Training transform with augmentation."""
    return transforms.Compose([
        transforms.RandomResizedCrop(img_crop),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        _normalize,
    ])


def get_transform_val(img_resize: int = 256, img_crop: int = 224):
    """Validation transform (deterministic)."""
    return transforms.Compose([
        transforms.Resize(img_resize),
        transforms.CenterCrop(img_crop),
        transforms.ToTensor(),
        _normalize,
    ])


def get_transform_test_tencrop(img_resize: int = 256, img_crop: int = 224):
    """Test-time augmentation with TenCrop."""
    return transforms.Compose([
        transforms.Resize(img_resize),
        transforms.TenCrop(img_crop),
        transforms.Lambda(lambda crops: torch.stack([
            transforms.Compose([transforms.ToTensor(), _normalize])(c)
            for c in crops
        ])),
    ])