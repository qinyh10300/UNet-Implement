import argparse
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import datetime

from models import *
from utils import *

@torch.inference_mode()   # 禁用反向传播，只进行前向计算
def evaluate(model, val_loader, device, amp):
    model.eval()

    dice_score = 0
    progress = tqdm.tqdm(enumerate(val_loader), total=len(val_loader), desc='Evaluation')
    for i, batch in progress:
        images, true_masks = batch

        if len(images.shape) == 3:  # [B,H,W]对于单通道（灰度图）的情况
            images = images.unsqueeze(1)  # [B,1,H,W]

        assert images.shape[1] == model.n_channels, \
            f'Network has been defined with {model.n_channels} input channels, ' \
            f'but loaded images have {images.shape[1]} channels.'
        
        images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        # memory_format=torch.channels_last不改变张量的维度布局，只改变内存中的存储顺序用于gpu加速
        true_masks = true_masks.to(device=device, dtype=torch.long)

        # 使用混合精度
        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
            masks_pred = model(images)

            # convert to one-hot format
            true_masks_one_hot = F.one_hot(true_masks, num_classes=model.n_classes)   # shape=(b, h, w, c)
            true_masks_one_hot = true_masks_one_hot.permute(0, 3, 1, 2).float()   # shape=(b, c, h, w)
            masks_pred = F.one_hot(masks_pred.argmax(dim=1), num_classes=model.n_classes)  # shape=(b, h, w, c)
            masks_pred = masks_pred.permute(0, 3, 1, 2).float()  # shape=(b, c, h, w)
            
            # compute the Dice score, `[:, 1:]` ignoring background
            dice_score += multiclass_dice_coeff(masks_pred[:, 1:], true_masks_one_hot[:, 1:], reduce_batch_first=False)
            # reduce_batch_first=False 决定是逐样本计算还是将整个批次作为一个整体计算

    return dice_score / len(val_loader)   # dice_score分数越高越好

def train_one_epoch(model, train_loader, optimizer, criterion, device, amp, args, gradient_clipping=1.0):
    model.train()
    epoch_loss = 0
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    
    progress = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), desc='Training')
    for i, batch in progress:
        images, true_masks = batch
        
        if len(images.shape) == 3:  # [B,H,W]
            images = images.unsqueeze(1)  # [B,1,H,W]
            
        assert images.shape[1] == model.n_channels, \
            f'Network has been defined with {model.n_channels} input channels, ' \
            f'but loaded images have {images.shape[1]} channels.'
            
        images = images.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=torch.long)
        
        # 使用混合精度
        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
            masks_pred = model(images)
            # print(masks_pred.shape, true_masks.shape)
            # mask_pred.shape = (b, c, h, w)
            # true_masks.shape = (b, h, w)
            loss1 = criterion(masks_pred, true_masks)    # CE损失函数会自动对pred进行softmax计算

            # 打印非零数值，检测mask有没有导入正确
            # non_zero_mask = true_masks != 0
            # non_zero_values = true_masks[non_zero_mask]
            # print("非零值:", non_zero_values)

            masks_pred_softmax = F.softmax(masks_pred, dim=1).float()
            true_masks_one_hot = F.one_hot(true_masks, num_classes=model.n_classes)   # shape=(b, h, w, c)
            true_masks_one_hot = true_masks_one_hot.permute(0, 3, 1, 2).float()   # shape=(b, c, h, w)
            loss2 = dice_loss(masks_pred_softmax, true_masks_one_hot, multiclass=True)

            loss = args.alpha1 * loss1 + args.alpha2 * loss2
            
        optimizer.zero_grad(set_to_none=True)    # 清空梯度
        grad_scaler.scale(loss).backward()    # 反向传播
        grad_scaler.unscale_(optimizer)    # 梯度裁剪（可选）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping)
        grad_scaler.step(optimizer)    # 参数更新
        grad_scaler.update()
        
        epoch_loss += loss.item()
        progress.set_postfix(loss=f'{epoch_loss / (i+1):.4f}')   # 在进度条的右侧添加当前的平均损失值信息
    
    return epoch_loss / len(train_loader)

def record_prediction_examples(model, val_loader, epoch, writer, device, amp):
    """记录预测结果的可视化示例到 TensorBoard"""
    model.eval()
    
    images, true_masks = next(iter(val_loader))
    
    if len(images.shape) == 3:  # [B,H,W]
        images = images.unsqueeze(1)  # [B,1,H,W]
    
    images = images.to(device=device, dtype=torch.float32)
    true_masks = true_masks.to(device=device, dtype=torch.long)
    
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        with torch.no_grad():
            mask_pred = model(images)
            mask_pred = torch.argmax(mask_pred, dim=1)
    
    images = images.cpu().numpy()
    true_masks = true_masks.cpu().numpy()
    mask_pred = mask_pred.cpu().numpy()
    
    for i in range(min(4, len(images))):  # 最多记录 4 个示例
        # 原图
        writer.add_image(f'images/epoch_{epoch}/sample_{i}/input', images[i], epoch, dataformats='CHW')
        
        # 真实掩码
        colored_true_mask = mask_to_rgb(true_masks[i])
        writer.add_image(f'masks/epoch_{epoch}/sample_{i}/true', colored_true_mask, epoch, dataformats='HWC')
        
        # 预测掩码
        colored_pred_mask = mask_to_rgb(mask_pred[i])
        writer.add_image(f'masks/epoch_{epoch}/sample_{i}/pred', colored_pred_mask, epoch, dataformats='HWC')

def mask_to_rgb(mask):
    """将标签掩码转换为 RGB 彩色图像"""
    label_to_color = {
        0: (0, 0, 0),      # 黑色： background
        1: (0, 255, 0),    # 绿色： liugua
        2: (255, 170, 0),  # 棕色： huahen
    }
    
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for label, color in label_to_color.items():
        rgb[mask == label] = color
    
    return rgb

def get_args():
    parser = argparse.ArgumentParser(description="Train the UNet on images and target masks")
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size")
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--n_channels', type=int, default=1, help="Number of channels of your photos")
    parser.add_argument('--classes', type=int, default=6, help="Number of classes")
    parser.add_argument('--base_channels', type=int, default=4, help="Number of basic channels used in UNet")
    parser.add_argument('--random_seed', type=int, default=42, help="Random seed to spilce dataset")
    parser.add_argument('--val_percent', type=float, default=0.2, help="evaluation percent of total dataset")
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--save_ckpt_frequency', type=int, default=10, help='How many epoches to save a checkpoint')
    parser.add_argument('--dir_checkpoint', type=str, default="./checkpoints", help='Where to save checkpoints')
    parser.add_argument('--alpha1', type=float, default=0.4, help='Hyperparameter Alpha 1')
    parser.add_argument('--alpha2', type=float, default=0.6, help='Hyperparameter Alpha 2')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    # 创建 TensorBoard 日志目录
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'runs/unet_{timestamp}'
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard 日志目录: {log_dir}")
    
    # 记录超参数
    writer.add_text('Parameters', str(vars(args)))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    full_dataset = CustomSegmentationDataset(root_dir='./dataset/')
    n_val = int(len(full_dataset) * args.val_percent)
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.random_seed))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = UNet(n_channels=args.n_channels, n_classes=args.classes, base_channels=args.base_channels)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95))
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    # 如果连续5个epochs所监听的指标没有提升（‘max’）,则自适应调整优化器optimizer（降低学习率）

    for epoch in range(args.epochs):
        # ******************Train******************
        epoch_train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            amp=args.amp,
            args=args
        )
        print(f"Training Loss: {epoch_train_loss} in Epoch-{epoch}")
        writer.add_scalar('Loss/train', epoch_train_loss, epoch)

        # ******************Eval******************
        val_score = evaluate(model, val_loader, device, args.amp)
        print(f"Evaluation Dice score: {val_score} in Epoch-{epoch}")
        writer.add_scalar('Dice/validation', val_score, epoch)
        
        # 获取当前学习率并记录
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning Rate', current_lr, epoch)
        scheduler.step(val_score)   # 更新学习率调度器

        # 记录预测示例
        if epoch % 5 == 0:  # 每 5 个 epoch 记录一次示例图
            record_prediction_examples(model, val_loader, epoch, writer, device, args.amp)

        if (epoch + 1) % args.save_ckpt_frequency == 0 or epoch == args.epochs - 1:
            Path(args.dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, str(args.dir_checkpoint + '/checkpoint_epoch{}.pth'.format(epoch)))
            print(f'Checkpoint {epoch} saved!')

    # 关闭 TensorBoard writer
    writer.close()
    print('Training complete!')