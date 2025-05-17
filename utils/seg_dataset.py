import os
from PIL import Image
from torch.utils.data import Dataset

class CustomSegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        """
        :param root_dir: 根目录，如 'dataset/'
        :param transform: 用于图像的 transform
        :param target_transform: 用于 mask 的 transform
        """
        self.image_mask_pairs = []
        self.transform = transform
        self.target_transform = target_transform

        # 遍历每个类别子文件夹（如 madian, yuyan...）
        for category in os.listdir(root_dir):
            category_path = os.path.join(root_dir, category)
            if not os.path.isdir(category_path):
                continue

            image_dir = os.path.join(category_path, category)
            mask_dir = os.path.join(category_path, f"{category}_target")

            if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
                continue  # 如果该类没有图像或 mask 子文件夹则跳过

            # 遍历图像文件
            for filename in os.listdir(image_dir):
                if not filename.endswith('.png'):
                    continue
                image_path = os.path.join(image_dir, filename)
                mask_name = filename.replace(f"{category}_", f"{category}_target_")
                mask_path = os.path.join(mask_dir, mask_name)

                if os.path.exists(mask_path):
                    self.image_mask_pairs.append((image_path, mask_path))

    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.image_mask_pairs[idx]
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask
