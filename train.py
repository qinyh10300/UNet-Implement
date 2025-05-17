import argparse
import torch
import torch.optim as optim
from UNet import UNet

def train_model():
    pass

def get_args():
    parser = argparse.ArgumentParser(description="Train the UNet on images and target masks")
    parser.add_argument('--epochs', type=int, required=True, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, required=True, help="Batch size")
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-5, help="Learning rate")
    parser.add_argument('--n_channels', type=int, default=1, help="Number of channels of your photos")
    parser.add_argument('--classes', type=int, default=5, help="Number of classes")
    parser.add_argument('--base_channels', type=int, default=32, help="Number of basic channels used in UNet")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet(n_channels=args.n_channels, n_classes=args.classes, base_channels=args.base_channels)
    model.to(device)

    # 设置优化器和学习率
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_model(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
    )

