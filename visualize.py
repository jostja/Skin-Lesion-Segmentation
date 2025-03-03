import argparse
import torch
from architecture import UNet
import albumentations as A
from data_loading import ISICDataset
from torch.utils.data import Subset
import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to the checkpoint file')
    parser.add_argument('--images_path', type=str, default='/hpi/fs00/home/jannis.jost/Skin-Lesion-Segmentation/data/images', help='Path to the training images')
    parser.add_argument('--masks_path', type=str, default='/hpi/fs00/home/jannis.jost/Skin-Lesion-Segmentation/data/labels', help='Path to the training masks')
    parser.add_argument('--img_size', type=int, default=256, help='Size of the input images')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of test samples to visualize')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=3, n_classes=1)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    transform = A.Resize(height=args.img_size, width=args.img_size)
    dataset = ISICDataset(images_path=args.images_path,
                          masks_path=args.masks_path,
                          size=args.img_size,
                          transform=transform)

    train_size = int(0.8 * len(dataset))
    test_dataset = Subset(dataset, range(train_size, len(dataset)))

    num_samples = min(args.num_samples, len(test_dataset))  # Ensure we don't exceed dataset size

    fig, axes = plt.subplots(num_samples, 3, figsize=(10, 4 * num_samples))  # Grid for visualization

    for i in range(num_samples):
        img, true_mask = test_dataset[i]
        img = img.permute(1, 2, 0).numpy()
        true_mask = true_mask.permute(1, 2, 0).squeeze().numpy()

        with torch.no_grad():
            img_tensor = torch.Tensor(img).unsqueeze(0).permute(0, 3, 1, 2).to(device)
            generated_mask = model(img_tensor).squeeze().cpu().numpy()
            generated_mask = (generated_mask > 0.5).astype(np.float32)  # Threshold the mask for better visualization

        true_mask_resized = cv2.resize(true_mask, (img.shape[1], img.shape[0]))
        true_mask_stacked = np.stack((true_mask_resized,) * 3, axis=-1)

        generated_mask_resized = cv2.resize(generated_mask, (img.shape[1], img.shape[0]))
        generated_mask_stacked = np.stack((generated_mask_resized,) * 3, axis=-1)

        # Plot each row with: Original Image, True Mask, Generated Mask
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(true_mask_stacked)
        axes[i, 1].set_title('True Mask')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(generated_mask_stacked)
        axes[i, 2].set_title('Generated Mask')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig('multiple_image_masks.png')
    plt.show()

if __name__ == '__main__':
    main()