from torchvision import transforms
import torch
from torchvision.transforms import InterpolationMode,Compose, ColorJitter, RandomHorizontalFlip, RandomResizedCrop, ToTensor, Normalize

def mask_to_tensor(mask):
    mask_tensor = transforms.functional.pil_to_tensor(mask).long()  # Keep integer labels
    return mask_tensor

def get_transform(is_train):
    if is_train:
        # Separate transformations for image and mask
        transforms_image = Compose([
            # transforms.Resize((594, 800)),
            RandomResizedCrop(size=(594, 800), scale=(0.7, 1.0), ratio=(0.72,1.0), interpolation=InterpolationMode.BILINEAR),
            RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(torch.nn.ModuleList([ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),]), p=0.4),
            transforms.RandomEqualize(p=0.35),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        transforms_mask = transforms.Compose([
            RandomResizedCrop(size=(594, 800), scale=(0.7, 1.0), ratio=(0.72,1.0), interpolation=InterpolationMode.NEAREST),
            RandomHorizontalFlip(p=0.5),
            mask_to_tensor
        ])
    else:
        # Transformations for validation (no augmentation, only resizing and normalization)
        transforms_image = transforms.Compose([
            transforms.Resize((594, 800)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        transforms_mask = transforms.Compose([
            transforms.Resize((594, 800),interpolation=InterpolationMode.NEAREST),
            mask_to_tensor
        ])
    
    return transforms_image,transforms_mask