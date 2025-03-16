# Skin Lesion Segmentation with UNet

This repository provides a PyTorch implementation for training a UNet model for skin lesion segmentation. The model is trained using the ISIC dataset and our training supports data augmentation, multiple loss functions, and model checkpoints.

## Prerequisites

Ensure you have the following dependencies installed:

- Python 3.7+
- PyTorch
- Torchvision
- Albumentations
- Segmentation Models PyTorch (smp)
- tqdm
- OpenCV (`cv2`)

You can install the required dependencies using:

```bash
pip install torch torchvision albumentations tqdm segmentation-models-pytorch opencv-python
```

## Dataset

Download the ISIC dataset and place the images and labels in the appropriate directories:

- **Images:** `Skin-Lesion-Segmentation/data/images`
- **Masks:** `Skin-Lesion-Segmentation/data/labels`

[ISIC Dataset Link](https://www.kaggle.com/datasets/tschandl/isic2018-challenge-task1-data-segmentation/data)

## Model Checkpoints

Pre-trained model checkpoints are available for download:

[Download Model Checkpoints](https://drive.google.com/drive/folders/1GG3RhCCeAK8jxDWmllAXjVvhIAxKnpci?usp=sharing)

To execute the demo notebook, ensure the model checkpoint `model.pth.tar` is in the `model_checkpoints` directory.
You also find the model checkpoints for the ensemble model in the `ensemble` directory and the model with the Mobilenet backbone at `mobilenet.pth.tar`.

## How to Run

To train the model, use the following command:

```bash
python train.py --output_dir "model_checkpoints/segmentation_model" --lambda_dice 1 --lambda_bce 0
```
The above command trains the model with a dice loss only and saves the model checkpoints in the `model_checkpoints/segmentation_model` directory.

There are more arguments available to modify the training.

### Available Arguments

| Argument                | Default                                  | Description                        |
| ----------------------- | ---------------------------------------- | ---------------------------------- |
| `--images_path`         | `/data/images`                           | Path to training images            |
| `--masks_path`          | `data/labels`                            | Path to training masks             |
| `--img_size`            | `256`                                    | Size of input images               |
| `--output_dir `         | `/model_checkpoints/output`              | Directory to save the checkpoints  |
| `--batch_size`          | `80`                                     | Batch size for training            |
| `--learning_rate`       | `1e-3`                                   | Learning rate for optimizer        |
| `--num_epochs`          | `30`                                     | Number of epochs to train          |
| `--checkpoint_interval` | `5`                                      | Interval to save model checkpoints |
| `--augmentation`        | `geometric`                              | Type of augmentation               |
| `--model_type`          | `UNet`                                   | Model type (UNet or Mobilenet)     |
| `--loss`                | `diceBCE`                                | Loss function (diceBCE or BCE)     |
| `--lambda_dice`         | `1.0`                                    | Weight for the dice loss           |
| `--lambda_bce`          | `1.0`                                    | Weight for the BCE loss            |



## Testing the Model

After training the model is evaluated automatically on the test set and the results are saved as part of a training summary.
If you still want to test the model, you can use the following command:

```bash
python test.py --model_path <path_to_checkpoint> --model_type <model_type>
```
The results will be saved in a `<checkpoint_name>_results.json` file in the models directory.


## GitHub Repository

Find the full source code on GitHub: [GitHub Repo](https://github.com/jostja/Skin-Lesion-Segmentation)

## Results & Summary

The training process generates a summary file for the training, including training parameters, loss values and model performance:

- Training loss
- Validation loss
- Testing results

These details are saved in `summary.json` in the output directory.

## EnsembleUNet

There also exists functionality for an ensemble model that combines multiple UNet models to improve segmentation performance. The ensemble model uses the output of multiple UNet models to generate the final segmentation mask.

You can find the ensemble model checkpoints at [Download Model Checkpoints](https://drive.google.com/drive/folders/1csiIvs3EUVGpjIgoQOrTkPDm5GlTYvd1?usp=sharing).

To test an ensemble model, put the model checkpoints of the UNets in the `model_checkpoints/ensemble` directory and run:

```bash
python test.py --model_path "model_checkpoints/ensemble" --model_type "EnsembleUNet"
```

The test results will be saved in ensemble_results.json in `model_checkpoints`.

