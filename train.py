import argparse
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from models import *
from utils import *

def train_one_epoch(model, train_loader, optimizer, criterion, device, amp,):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()

    model.train()
    for batch in train_loader:
        images, true_masks = batch

        images = images.unsqueeze(1)

        print(images.shape)
        assert images.shape[1] == model.n_channels, \
            f'Network has been defined with {model.n_channels} input channels, ' \
            f'but loaded images have {images.shape[1]} channels. Please check that ' \
            'the images are loaded correctly.'

        images = images.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=torch.long)

        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
            masks_pred = model(images)
            if model.n_classes == 1:
                loss = criterion(masks_pred.squeeze(1), true_masks.float())
                loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
            else:
                loss = criterion(masks_pred, true_masks)
                loss += dice_loss(
                    F.softmax(masks_pred, dim=1).float(),
                    F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                    multiclass=True
                )

        optimizer.zero_grad(set_to_none=True)
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        grad_scaler.step(optimizer)
        grad_scaler.update()


def get_args():
    parser = argparse.ArgumentParser(description="Train the UNet on images and target masks")
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-5, help="Learning rate")
    parser.add_argument('--n_channels', type=int, default=1, help="Number of channels of your photos")
    parser.add_argument('--classes', type=int, default=5, help="Number of classes")
    parser.add_argument('--base_channels', type=int, default=32, help="Number of basic channels used in UNet")
    parser.add_argument('--random_seed', type=int, default=42, help="Random seed to spilce dataset")
    parser.add_argument('--val_percent', type=float, default=0.2, help="evaluation percent of total dataset")
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

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

    for epoch in range(args.epochs):
        # Train
        train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            amp=args.amp
        )

        # # Eval
        # model.eval()
        # test_loss = 0
        # test_acc = 0
        
        # with torch.no_grad():
        #     for images, masks in val_loader:
        #         images, masks = images.to(device), masks.to(device)
        #         outputs = model(images)
        #         loss = combined_loss(outputs, masks)
        #         test_loss += loss.item()
        #         test_acc += (outputs.round() == masks).float().mean().item()
        
        # test_loss /= len(val_loader)
        # test_acc /= len(val_loader)
        # print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

