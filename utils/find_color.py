import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def analyze_color_distribution(image_path):
    """
    分析RGB图像中的像素颜色分布
    
    参数:
        image_path (str): 图像文件的路径
        
    返回:
        dict: 键为RGB颜色元组 (r,g,b)，值为该颜色的像素数量
    """
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"文件不存在: {image_path}")
        return {}
    
    # 读取图像
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # 检查图像是否为RGB
    if len(img_array.shape) < 3 or img_array.shape[2] < 3:
        print(f"警告: 图像不是RGB格式: {img_array.shape}")
        return {}
    
    # 重塑图像数组为二维数组，每行表示一个像素
    pixels = img_array.reshape(-1, img_array.shape[2])
    
    # 计算颜色分布
    color_counts = {}
    for pixel in pixels:
        # 使用RGB值的元组作为字典键
        color = tuple(pixel[:3])
        color_counts[color] = color_counts.get(color, 0) + 1
    
    return color_counts

def display_color_distribution(color_counts, top_n=10):
    """
    显示颜色分布信息
    
    参数:
        color_counts (dict): 颜色分布字典
        top_n (int): 显示前N个最常见的颜色
    """
    if not color_counts:
        print("没有颜色数据可显示")
        return
    
    # 按像素数量排序
    sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
    
    # 计算总像素数
    total_pixels = sum(color_counts.values())
    
    print(f"\n颜色分布分析 (显示前{top_n}种颜色):")
    print("-" * 50)
    print(f"{'RGB值':<15} | {'像素数量':<10} | {'百分比':<8} | 颜色样本")
    print("-" * 50)
    
    # 显示前N种最常见的颜色
    for i, (color, count) in enumerate(sorted_colors[:top_n]):
        percentage = (count / total_pixels) * 100
        # 创建颜色样本字符串
        color_sample = f"\033[48;2;{color[0]};{color[1]};{color[2]}m    \033[0m"
        print(f"{str(color):<15} | {count:<10} | {percentage:6.2f}% | {color_sample}")
    
    # 如果颜色种类超过top_n，显示其他颜色的汇总
    if len(sorted_colors) > top_n:
        other_count = sum(count for _, count in sorted_colors[top_n:])
        other_percentage = (other_count / total_pixels) * 100
        print(f"{'其他颜色':<15} | {other_count:<10} | {other_percentage:6.2f}% | (多种)")
    
    print("-" * 50)
    print(f"总计: {total_pixels} 像素, {len(color_counts)} 种不同颜色")

def plot_color_distribution(color_counts, top_n=10):
    """
    绘制颜色分布图表
    
    参数:
        color_counts (dict): 颜色分布字典
        top_n (int): 显示前N个最常见的颜色
    """
    if not color_counts:
        print("没有颜色数据可显示")
        return
    
    # 按像素数量排序
    sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
    top_colors = sorted_colors[:top_n]
    
    # 准备图表数据
    labels = [str(color) for color, _ in top_colors]
    counts = [count for _, count in top_colors]
    colors = [f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}" for color, _ in top_colors]
    
    # 创建图表
    plt.figure(figsize=(12, 6))
    
    # 条形图
    ax1 = plt.subplot(1, 2, 1)
    bars = ax1.bar(range(len(top_colors)), counts, color=colors)
    ax1.set_xticks(range(len(top_colors)))
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_ylabel('像素数量')
    ax1.set_title(f'前{top_n}种最常见颜色的分布')
    
    # 饼图
    ax2 = plt.subplot(1, 2, 2)
    total_count = sum(counts)
    percentages = [count/total_count*100 for count in counts]
    wedges, texts, autotexts = ax2.pie(
        counts, 
        labels=[f"{p:.1f}%" for p in percentages], 
        autopct='', 
        colors=colors, 
        wedgeprops=dict(width=0.5)
    )
    ax2.set_title('颜色分布占比')
    
    # 添加图例
    ax2.legend(
        wedges, 
        [f"{color} ({p:.1f}%)" for color, p in zip(labels, percentages)], 
        loc="center left", 
        bbox_to_anchor=(1, 0, 0.5, 1)
    )
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 使用示例
    image_path = 'dataset/huahen/huahen_target/huahen_target_7.png'
    # image_path = 'dataset/liugua/liugua_target/liugua_target_3.png'
    color_counts = analyze_color_distribution(image_path)
    
    # 显示颜色分布文本信息
    display_color_distribution(color_counts, top_n=10)
    
    # 绘制颜色分布图表
    plot_color_distribution(color_counts, top_n=8)
    
    # 直接输出字典形式的颜色分布
    print("\n颜色分布字典:")
    print(color_counts)