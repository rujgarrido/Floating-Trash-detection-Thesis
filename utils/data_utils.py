"""
Data Utilities for Seawaste Detection
Handles data loading, preprocessing, and augmentation for marine debris datasets.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import yaml
import json
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging


logger = logging.getLogger(__name__)


class MarineDataPreprocessor:
    """Preprocessing utilities for marine/underwater imagery"""

    @staticmethod
    def enhance_underwater_image(image: np.ndarray) -> np.ndarray:
        """
        Enhance underwater images using color correction and contrast adjustment

        Args:
            image: Input image (BGR format)

        Returns:
            Enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Merge channels
        enhanced_lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # White balance correction
        enhanced = MarineDataPreprocessor.white_balance(enhanced)

        return enhanced

    @staticmethod
    def white_balance(image: np.ndarray) -> np.ndarray:
        """
        Apply white balance correction to compensate for underwater color cast

        Args:
            image: Input image (BGR format)

        Returns:
            White-balanced image
        """
        result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        return result

    @staticmethod
    def remove_water_haze(image: np.ndarray, omega: float = 0.95) -> np.ndarray:
        """
        Reduce water haze/turbidity effect

        Args:
            image: Input image (BGR format)
            omega: Transmission retention parameter

        Returns:
            Dehazed image
        """
        # Simple dark channel prior method
        def get_dark_channel(img, size=15):
            min_channel = np.min(img, axis=2)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
            dark = cv2.erode(min_channel, kernel)
            return dark

        # Normalize
        norm_img = image.astype(np.float64) / 255.0

        # Get dark channel
        dark_channel = get_dark_channel(norm_img, size=15)

        # Estimate atmospheric light
        flat_dark = dark_channel.flatten()
        num_pixels = flat_dark.size
        num_brightest = int(max(num_pixels * 0.001, 1))
        indices = np.argpartition(flat_dark, -num_brightest)[-num_brightest:]

        atmospheric_light = np.zeros(3)
        for idx in indices:
            y, x = np.unravel_index(idx, dark_channel.shape)
            atmospheric_light += norm_img[y, x, :]
        atmospheric_light /= num_brightest

        # Estimate transmission
        transmission = 1 - omega * get_dark_channel(norm_img / atmospheric_light, size=15)
        transmission = np.clip(transmission, 0.1, 1.0)

        # Recover image
        recovered = np.zeros_like(norm_img)
        for i in range(3):
            recovered[:, :, i] = (norm_img[:, :, i] - atmospheric_light[i]) / transmission + atmospheric_light[i]

        recovered = np.clip(recovered * 255, 0, 255).astype(np.uint8)

        return recovered


class SeawasteDataset:
    """Custom dataset class for seawaste detection"""

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        transform=None,
        preprocess_marine: bool = True
    ):
        """
        Initialize dataset

        Args:
            images_dir: Directory containing images
            labels_dir: Directory containing YOLO format labels
            transform: Augmentation transforms
            preprocess_marine: Apply marine-specific preprocessing
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform
        self.preprocess_marine = preprocess_marine
        self.preprocessor = MarineDataPreprocessor()

        # Get image files
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_files.extend(list(self.images_dir.glob(ext)))

        logger.info(f"Found {len(self.image_files)} images in {images_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply marine preprocessing
        if self.preprocess_marine:
            image = self.preprocessor.enhance_underwater_image(image)

        # Load labels
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        boxes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    boxes.append([float(x) for x in line.strip().split()])

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes)
            image = transformed['image']
            boxes = transformed['bboxes']

        return image, boxes


def get_augmentation_transforms(config: dict, is_train: bool = True):
    """
    Get augmentation transforms based on configuration

    Args:
        config: Configuration dictionary
        is_train: Whether training or validation

    Returns:
        Albumentations compose transform
    """
    if not is_train:
        return A.Compose([
            A.Resize(640, 640),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    aug_config = config.get('augmentation', {})

    transforms = [
        A.Resize(640, 640),
        A.HorizontalFlip(p=aug_config.get('fliplr', 0.5)),
        A.VerticalFlip(p=aug_config.get('flipud', 0.0)),
        A.Rotate(limit=aug_config.get('degrees', 0), p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=int(aug_config.get('hsv_h', 0.015) * 180),
            sat_shift_limit=int(aug_config.get('hsv_s', 0.7) * 255),
            val_shift_limit=int(aug_config.get('hsv_v', 0.4) * 255),
            p=0.5
        ),
    ]

    # Marine-specific augmentations
    marine_aug = config.get('marine_augmentation', {})
    if marine_aug.get('water_color_shift', False):
        transforms.append(
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5)
        )

    if marine_aug.get('turbidity_simulation', False):
        transforms.append(A.GaussianBlur(blur_limit=(3, 7), p=0.3))
        transforms.append(A.GaussNoise(var_limit=(10, 50), p=0.3))

    transforms.append(
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
    )


def verify_dataset_structure(data_yaml: str) -> Dict:
    """
    Verify dataset structure and return statistics

    Args:
        data_yaml: Path to dataset YAML configuration

    Returns:
        Dictionary with dataset statistics
    """
    with open(data_yaml, 'r') as f:
        config = yaml.safe_load(f)

    dataset_path = Path(config['path'])
    stats = {
        'valid': True,
        'train_images': 0,
        'val_images': 0,
        'test_images': 0,
        'train_labels': 0,
        'val_labels': 0,
        'test_labels': 0,
        'classes': config['nc'],
        'class_names': list(config['names'].values())
    }

    # Check train set
    train_path = dataset_path / config['train']
    if train_path.exists():
        stats['train_images'] = len(list(train_path.glob('*.jpg'))) + len(list(train_path.glob('*.png')))

        labels_path = dataset_path / 'labels' / 'train'
        if labels_path.exists():
            stats['train_labels'] = len(list(labels_path.glob('*.txt')))

    # Check val set
    val_path = dataset_path / config['val']
    if val_path.exists():
        stats['val_images'] = len(list(val_path.glob('*.jpg'))) + len(list(val_path.glob('*.png')))

        labels_path = dataset_path / 'labels' / 'val'
        if labels_path.exists():
            stats['val_labels'] = len(list(labels_path.glob('*.txt')))

    # Check test set
    if 'test' in config:
        test_path = dataset_path / config['test']
        if test_path.exists():
            stats['test_images'] = len(list(test_path.glob('*.jpg'))) + len(list(test_path.glob('*.png')))

            labels_path = dataset_path / 'labels' / 'test'
            if labels_path.exists():
                stats['test_labels'] = len(list(labels_path.glob('*.txt')))

    stats['total_images'] = stats['train_images'] + stats['val_images'] + stats['test_images']
    stats['total_labels'] = stats['train_labels'] + stats['val_labels'] + stats['test_labels']

    return stats


def split_dataset(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """
    Split dataset into train/val/test sets

    Args:
        source_dir: Source directory with images and labels
        output_dir: Output directory for split dataset
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed for reproducibility
    """
    import shutil
    import random

    random.seed(seed)
    np.random.seed(seed)

    source_path = Path(source_dir)
    output_path = Path(output_dir)

    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_files = list(source_path.glob('images/*.jpg')) + list(source_path.glob('images/*.png'))
    random.shuffle(image_files)

    # Calculate split indices
    n_total = len(image_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]

    # Copy files
    for split_name, split_files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        for img_file in split_files:
            # Copy image
            shutil.copy(
                img_file,
                output_path / 'images' / split_name / img_file.name
            )

            # Copy label if exists
            label_file = source_path / 'labels' / f"{img_file.stem}.txt"
            if label_file.exists():
                shutil.copy(
                    label_file,
                    output_path / 'labels' / split_name / f"{img_file.stem}.txt"
                )

    logger.info(f"Dataset split complete:")
    logger.info(f"  Train: {len(train_files)} images")
    logger.info(f"  Val:   {len(val_files)} images")
    logger.info(f"  Test:  {len(test_files)} images")


if __name__ == "__main__":
    # Example usage
    print("Seawaste Data Utilities")
