import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

color_to_label = {
    (0, 0, 0): 0,  # 黑色：  background
    (0, 255, 0): 1,  # 绿色：  liugua
    (255, 170, 0): 2   # 棕色：  huahen
}

label_to_color = {
    0: (0, 0, 0),  # 黑色：  background
    1: (0, 255, 0),  # 绿色：  liugua
    2: (255, 170, 0)   # 棕色：  huahen
}

def rgb_mask_to_label(mask_rgb):
    h, w, _ = mask_rgb.shape
    label_mask = np.zeros((h, w), dtype=np.uint8)   # 黑色：  背景
    for color, label in color_to_label.items():
        matches = np.all(mask_rgb == color, axis=-1)
        label_mask[matches] = label
    return label_mask

def rgb_to_grayscale(image_path):
    img = Image.open(image_path)
    img_array = np.array(img)
    
    if len(img_array.shape) < 3:   # 图像已经是单通道
        return img_array
    else:
        return img_array[:, :, 0]

if __name__ == "__main__":
    image = rgb_to_grayscale("dataset/liugua/liugua/liugua_4.png")

    print(image.shape)

    save_path = "output_grayscale.png"

    # 保存为 PNG 以保持无损
    Image.fromarray(image).save(save_path)
    print(f"已保存灰度图像到: {save_path}")

    img = Image.open("output_grayscale.png")
    img_array = np.array(img)
    print(img_array.shape)