import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from .utils import rgb_mask_to_label

class CustomSegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        
        all_pairs = []
        for category in os.listdir(root_dir):
            category_path = os.path.join(root_dir, category)
            if not os.path.isdir(category_path):
                continue
                
            image_dir = os.path.join(category_path, category)
            mask_dir = os.path.join(category_path, f"{category}_target")
            
            if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
                continue
                
            for filename in os.listdir(image_dir):
                if not filename.endswith('.png'):
                    continue
                image_path = os.path.join(image_dir, filename)
                mask_name = filename.replace(f"{category}_", f"{category}_target_")
                mask_path = os.path.join(mask_dir, mask_name)
                
                if os.path.exists(mask_path):
                    all_pairs.append((image_path, mask_path))
    
    def __len__(self):
        return len(self.image_mask_pairs)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.image_mask_pairs[idx]
        image = Image.open(img_path)
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        image_array = np.array(image)
        mask_array = np.array(mask)

        if len(image_array.shape) == 3:
            image_array = image_array[:, :, 0]
        mask_array = rgb_mask_to_label(mask_array)

        return image_array, mask_array