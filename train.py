import os
import json
import time
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import albumentations as A
from architecture import UNet
from data_loading import ISICDataset, get_augmentations
from test import DiceBCELoss, test_model

def main():
    # Parse arguments
    home = os.path.expanduser("~")
    parser = argparse.ArgumentParser(description='Train a UNet model for skin lesion segmentation.')
    parser.add_argument('--images_path', type=str, default=home+'/Skin-Lesion-Segmentation/data/images', help='Path to the training images')
    parser.add_argument('--masks_path', type=str, default=home+'/Skin-Lesion-Segmentation/data/labels', help='Path to the training masks')
    parser.add_argument('--img_size', type=int, default=256, help='Size of the input images')
    parser.add_argument('--batch_size', type=int, default=80, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs to train the model')
    parser.add_argument('--checkpoint_interval', type=int, default=5, help='Interval of epochs to save checkpoints')
    parser.add_argument('--augmentation', type=str, default='geometric', help='How to augment the data')
    parser.add_argument('--output_dir', type=str, default=home+'/Skin-Lesion-Segmentation/model_checkpoints/output', help='Directory to save the model checkpoints')
    parser.add_argument('--loss', type=str, default='diceBCE', help='Loss function to use')
    parser.add_argument('--lambda_dice', type=float, default=1.0, help='Weight for the dice loss')
    parser.add_argument('--lambda_bce', type=float, default=1.0, help='Weight for the BCE loss')

    args = parser.parse_args()

    images_path = args.images_path
    masks_path = args.masks_path
    img_size = args.img_size
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    checkpoint_interval = args.checkpoint_interval
    augmentation = args.augmentation
    output_dir = args.output_dir
    loss = args.loss
    lambda_dice = args.lambda_dice
    lambda_bce = args.lambda_bce

    os.makedirs(output_dir, exist_ok=True)
    # Define the augmentation pipeline
    geometric_transform, color_transform = get_augmentations(augmentation, img_size)

    val_test_transform = A.Resize(height=img_size, width=img_size)

    # Load the data
    ids = [image_file[:-4] for image_file in os.listdir(images_path) if image_file.endswith('.jpg')]
    train_size = int(0.8 * len(ids))
    val_size = int(0.1 * len(ids))

    train_ids = ids[:train_size]
    val_ids = ids[train_size:train_size+val_size]
    test_ids = ids[train_size+val_size:]

    train_dataset = ISICDataset(images_path=images_path,
                                masks_path=masks_path,
                                ids=train_ids,
                                size=img_size, 
                                geometric_transform=geometric_transform,
                                color_transform=color_transform)

    val_dataset = ISICDataset(images_path=images_path,
                                masks_path=masks_path,
                                ids=val_ids,
                                size=img_size, 
                                geometric_transform=val_test_transform)

    test_dataset = ISICDataset(images_path=images_path,
                                masks_path=masks_path,
                                ids=test_ids,
                                size=img_size, 
                                geometric_transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=3, n_classes=1).to(device)

    # Define the loss function and optimizer
    if loss == 'diceBCE':
        criterion = DiceBCELoss(lambda_dice=lambda_dice, lambda_bce=lambda_bce)
    elif loss == 'BCE':
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError("Invalid value for loss")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    train_losses = []
    val_losses = []
    # Print training parameters
    print(f"Training with {len(train_dataset)} samples")
    print(f"Validation with {len(val_dataset)} samples")
    print(f"Testing with {len(test_dataset)} samples")
    print(f"Using device {device}")
    print(f"Using augmentation: {augmentation}")
    print(f"Using loss function {loss}")
    print(f"Using learning rate {learning_rate}")
    print(f"Using batch size {batch_size}")
    print(f"Training for {num_epochs} epochs")
    print(f"Saving checkpoints every {checkpoint_interval} epochs")
    print(f"Output directory: {output_dir}")

    # Train the model
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        val_loss = 0

        if epoch % checkpoint_interval == 0 and epoch != 0:
            checkpoint = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch}
            torch.save(checkpoint, f"{output_dir}/" + "checkpointN"+str(epoch)+".pth.tar")  

        train_loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        for batch_idx, (images, masks) in train_loop:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, masks)
            loss.backward()
            optimizer.step()
            train_loop.set_description(f"Epoch[{epoch}/{num_epochs}]")
            train_loop.set_postfix(loss = loss.item())
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        val_loop = tqdm(enumerate(val_loader), total=len(val_loader), leave=False)
        for batch_idx, (images, masks) in val_loop:
            images, masks = images.to(device), masks.to(device)
            with torch.no_grad():
                output = model(images)
                loss = criterion(output, masks)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print(f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}")
        
    end_time = time.time()
    train_time = end_time - start_time

    # Save the model
    checkpoint = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
    torch.save(checkpoint, f"{output_dir}/" + "checkpoint_last.pth.tar")


    model.eval()
    test_results = test_model(test_loader, model, device)

    # Save the summary
    losses = {'train_losses': train_losses, 'val_losses': val_losses, 'test_results': test_results}
    summary = {'train_time': train_time, 'num_epochs': num_epochs, 'learning_rate': learning_rate, 'batch_size': batch_size, 'augmentation': augmentation, 'losses': losses, 'lambda_dice': lambda_dice, 'lambda_bce': lambda_bce, 'losses': losses, 'train_time': train_time}


    with open(f"{output_dir}/" + "summary.json", 'w') as f:
        json.dump(summary, f)


if __name__ == '__main__':
    main()