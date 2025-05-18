import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from UNet import *
from utils import *

def train_one_epoch():
    pass

def get_args():
    parser = argparse.ArgumentParser(description="Train the UNet on images and target masks")
    parser.add_argument('--epochs', type=int, required=True, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, required=True, help="Batch size")
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-5, help="Learning rate")
    parser.add_argument('--n_channels', type=int, default=1, help="Number of channels of your photos")
    parser.add_argument('--classes', type=int, default=5, help="Number of classes")
    parser.add_argument('--base_channels', type=int, default=32, help="Number of basic channels used in UNet")
    parser.add_argument('--random_seed', type=int, default=42, help="Random seed to spilce dataset")
    parser.add_argument('--val_percent', type=float, default=0.2, help="evaluation percent of total dataset")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    full_dataset = CustomSegmentationDataset(root_dir='./dataset/', split='train')
    n_val = int(len(full_dataset) * args.val_percent)
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.random_seed))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = UNet(n_channels=args.n_channels, n_classes=args.classes, base_channels=args.base_channels)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        # Train
        train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        # Eval
        model.eval()
        test_loss = 0
        test_acc = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = combined_loss(outputs, masks)
                test_loss += loss.item()
                test_acc += (outputs.round() == masks).float().mean().item()
        
        test_loss /= len(val_loader)
        test_acc /= len(val_loader)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

