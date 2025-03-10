import os
import json
import torch
import torch.nn as nn
from architecture import UNet
from data_loading import ISICDataset
from torch.utils.data import DataLoader, Subset
import argparse
import albumentations as A

class JaccardIndex(torch.nn.Module):
    def __init__(self):
        super(JaccardIndex, self).__init__()
    def forward(self, pred, target):
        smooth = 1e-8
        pred = torch.sigmoid(pred)
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        union = (pred + target).sum() - intersection
        return (intersection + smooth) / (union + smooth)

class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
    def forward(self, pred, target):
        smooth = 1e-8
        pred = torch.sigmoid(pred)
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

class DiceBCELoss(torch.nn.Module):
    def __init__(self, lambda_dice=1.0, lambda_bce=1.0):
        super(DiceBCELoss, self).__init__()
        self.lambda_dice = lambda_dice
        self.lambda_bce = lambda_bce
        self.dice = DiceLoss()
        self.ce = nn.BCEWithLogitsLoss()
    def forward(self, pred, target):
        return self.lambda_dice * self.dice(pred, target) + self.lambda_bce * self.ce(pred, target)



def calculate_loss(test_loader, model, criterion, device):
    test_loss = 0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for (images, masks) in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            test_loss += loss.item()
    model.train()
    return test_loss / len(test_loader)

def calculate_jaccard_indices(test_loader, model, device, threshold=0.65):
    jaccard_metric = 0
    threshold_jaccard_metric = 0
    model.to(device)
    criterion = JaccardIndex()
    with torch.no_grad():
        for (images, masks) in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            
            jaccard_index = criterion(outputs, masks).item()
            jaccard_metric += jaccard_index
            threshold_jaccard_metric += jaccard_index if jaccard_index >= threshold else 0
    return jaccard_metric / len(test_loader), threshold_jaccard_metric / len(test_loader)
        

def test_model(test_loader, model, device):
    # Test the model
    Dice_loss = calculate_loss(test_loader, model, DiceLoss(), device)
    print(f'Dice loss: {Dice_loss}')
    BCE_loss = calculate_loss(test_loader, model, nn.BCEWithLogitsLoss(), device)
    print(f'BCE loss: {BCE_loss}')
    # DiceBCE_loss = calculate_loss(test_loader, model, DiceBCELoss(), device)
    # print(f'DiceBCE loss: {DiceBCE_loss}')
    jaccard_index, threshold_jaccard_index = calculate_jaccard_indices(test_loader, model, device)
    print(f'Jaccard index: {jaccard_index}')
    print(f'Threshold Jaccard index: {threshold_jaccard_index}')
    # Return the results
    results = {
        'Dice_loss': Dice_loss,
        'BCE_loss': BCE_loss,
        #'DiceBCE_loss': DiceBCE_loss,
        'jaccard_index': jaccard_index,
        'threshold_jaccard_index': threshold_jaccard_index
    }
    return results
    

def main():
    home = os.path.expanduser('~')
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', type=str, default=home+'/Skin-Lesion-Segmentation/data/images', help='Path to the training images')
    parser.add_argument('--masks_path', type=str, default=home+'/Skin-Lesion-Segmentation/data/labels', help='Path to the training masks')
    parser.add_argument('--img_size', type=int, default=256, help='Size of the input images')
    parser.add_argument('--model_checkpoint', type=str, default=home+'/Skin-Lesion-Segmentation/model_checkpoints/test', help='Path to the model checkpoint')
    args = parser.parse_args()
    
    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet(n_channels=3, n_classes=1)
    checkpoint = torch.load(args.model_checkpoint, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    # Load the test data
    ids = [image_file[:-4] for image_file in os.listdir(args.images_path) if image_file.endswith('.jpg')]
    train_size = int(0.8 * len(ids))
    val_size = int(0.1 * len(ids))
    test_ids = ids[train_size + val_size:]
    test_transform = A.Compose([
        A.Resize(height=args.img_size, width=args.img_size),
    ])
    test_dataset = ISICDataset(images_path=args.images_path,
                                masks_path=args.masks_path,
                                ids=test_ids,
                                size=args.img_size,
                                geometric_transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Test the model
    results = test_model(test_loader, model, device)
    
    # Save the results
    directory = os.path.dirname(args.model_checkpoint)
    file = os.path.splitext(os.path.basename(args.model_checkpoint))[0]
    with open(os.path.join(directory, file + 'results.json'), 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':
    main()