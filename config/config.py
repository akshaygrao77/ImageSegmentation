from torchvision import transforms
import torch
from torchvision.transforms import InterpolationMode,Compose, ColorJitter, RandomHorizontalFlip, RandomResizedCrop, ToTensor, Normalize

def mask_to_tensor(mask):
    mask_tensor = transforms.functional.pil_to_tensor(mask).long()  # Keep integer labels
    return mask_tensor


def get_mean_stds(dataset=None):
    if(dataset is not None and dataset == "Car_damages_dataset"):
        # Car damages trainset
        return [0.4693, 0.4597, 0.4667],[0.2515, 0.2506, 0.2500]
    # Imagenet standard mean,std
    return [0.485, 0.456, 0.406],[0.229, 0.224, 0.225]


# RandomAutocontrast , RandomEqualize
def get_transform(is_train,dataset=None):
    mean,std=get_mean_stds(dataset)
    final_size = (640, 640)
    if is_train:
        # Separate transformations for image and mask
        transforms_image = Compose([
            RandomResizedCrop(size=final_size, scale=(0.7, 1.0), ratio=(0.72,1.0), interpolation=InterpolationMode.BILINEAR),
            RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation((-10,10),interpolation=InterpolationMode.BILINEAR),
            transforms.RandomAutocontrast(p=0.4),
            transforms.RandomApply(torch.nn.ModuleList([ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),]), p=0.4),
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])
        transforms_mask = transforms.Compose([
            RandomResizedCrop(size=final_size, scale=(0.7, 1.0), ratio=(0.72,1.0), interpolation=InterpolationMode.NEAREST),
            RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation((-10,10),interpolation=InterpolationMode.NEAREST),
            mask_to_tensor
        ])
    else:
        # Transformations for validation (no augmentation, only resizing and normalization)
        transforms_image = transforms.Compose([
            transforms.Resize(final_size,interpolation=InterpolationMode.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])
        transforms_mask = transforms.Compose([
            transforms.Resize(final_size,interpolation=InterpolationMode.NEAREST),
            mask_to_tensor
        ])
    
    return transforms_image,transforms_mask