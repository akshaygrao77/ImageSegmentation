import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO
import numpy as np

class COCOSegmentationDataset(Dataset):
    def __init__(self, root, ann_file, transforms_image=None, transforms_mask=None):
        """
        Args:
            root (str): Path to the root directory containing images.
            ann_file (str): Path to the COCO JSON file.
            transforms_image (callable, optional): Data transformations to apply to the image.
            transforms_mask (callable, optional): Data transformations to apply to the mask.
        """
        print("root ", root)
        print("ann_file ", ann_file)
        self.root = root
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.transforms_image = transforms_image
        self.transforms_mask = transforms_mask

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        # Get image ID
        img_id = self.ids[index]
        
        # Load image
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        self.img_path = img_path
        
        # print("img_path ",img_path)
        # Load annotations (segmentation masks)
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Create a mask
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        for ann in anns:
            mask = np.maximum(mask, self.coco.annToMask(ann) * ann['category_id'])
        mask = Image.fromarray(mask)  # Convert the mask to a PIL Image for augmentation

        # Apply transformations if specified
        if self.transforms_image and self.transforms_mask:
            seed = np.random.randint(2147483647)  # Generate a random seed for consistent transformations
            torch.manual_seed(seed)  # Set the seed for image transformations
            image = self.transforms_image(image)
            
            torch.manual_seed(seed)  # Set the same seed for mask transformations
            mask = self.transforms_mask(mask)

        return image, mask
