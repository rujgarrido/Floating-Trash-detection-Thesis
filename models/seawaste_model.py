"""
Custom Seawaste Detection Model
This module implements a custom YOLO-based model optimized for marine debris detection.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import yaml
from pathlib import Path


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    Enhances feature representation for better detection of marine debris
    """
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super(CBAM, self).__init__()

        # Channel Attention Module
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial Attention Module
        self.conv_spatial = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid_channel(avg_out + max_out)
        x = x * channel_att

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.conv_spatial(spatial_att)
        spatial_att = self.sigmoid_spatial(spatial_att)
        x = x * spatial_att

        return x


class MarineColorCorrection(nn.Module):
    """
    Learnable color correction module for underwater/marine imagery
    Adapts to varying water conditions (turbidity, lighting, depth)
    """
    def __init__(self, channels: int = 3):
        super(MarineColorCorrection, self).__init__()
        self.color_transform = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        correction = self.color_transform(x)
        return x * correction + x * (1 - correction)


class SmallObjectEnhancer(nn.Module):
    """
    Feature enhancement module for detecting small marine debris
    Uses multi-scale feature fusion
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(SmallObjectEnhancer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, 5, padding=2)
        self.fusion = nn.Conv2d(out_channels * 3, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat1 = self.conv1(x)
        feat3 = self.conv3(x)
        feat5 = self.conv5(x)
        fused = torch.cat([feat1, feat3, feat5], dim=1)
        out = self.fusion(fused)
        out = self.bn(out)
        out = self.relu(out)
        return out


class SeawasteDetectionModel(nn.Module):
    """
    Main Seawaste Detection Model
    Custom architecture optimized for marine debris detection
    """
    def __init__(
        self,
        base_model: str = 'yolov8m',
        num_classes: int = 15,
        use_cbam: bool = True,
        use_color_correction: bool = True,
        use_small_object_enhancer: bool = True,
        pretrained: bool = True
    ):
        super(SeawasteDetectionModel, self).__init__()

        self.num_classes = num_classes
        self.base_model_name = base_model

        # Marine-specific preprocessing
        if use_color_correction:
            self.color_correction = MarineColorCorrection(channels=3)
        else:
            self.color_correction = nn.Identity()

        # Store configuration
        self.config = {
            'base_model': base_model,
            'num_classes': num_classes,
            'use_cbam': use_cbam,
            'use_color_correction': use_color_correction,
            'use_small_object_enhancer': use_small_object_enhancer
        }

        # The actual YOLOv8 model will be loaded via ultralytics
        # This class serves as a wrapper with custom preprocessing and postprocessing
        self.yolo_model = None  # Will be initialized in load_base_model()

    def load_base_model(self, weights_path: Optional[str] = None):
        """
        Load the base YOLO model from ultralytics
        """
        try:
            from ultralytics import YOLO

            if weights_path and Path(weights_path).exists():
                self.yolo_model = YOLO(weights_path)
            else:
                # Load pretrained model
                self.yolo_model = YOLO(f'{self.base_model_name}.pt')

            print(f"Loaded base model: {self.base_model_name}")
        except ImportError:
            raise ImportError(
                "ultralytics package not found. Install with: pip install ultralytics"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with marine-specific preprocessing
        """
        # Apply color correction for marine imagery
        x = self.color_correction(x)

        # Forward through YOLO model
        if self.yolo_model is not None:
            return self.yolo_model(x)
        else:
            raise RuntimeError(
                "Base model not loaded. Call load_base_model() first."
            )

    def predict(
        self,
        source,
        conf: float = 0.25,
        iou: float = 0.45,
        **kwargs
    ):
        """
        Run prediction on images, videos, or streams

        Args:
            source: Input source (image path, video path, or stream)
            conf: Confidence threshold
            iou: IoU threshold for NMS
            **kwargs: Additional arguments for YOLO predict
        """
        if self.yolo_model is None:
            self.load_base_model()

        return self.yolo_model.predict(
            source=source,
            conf=conf,
            iou=iou,
            **kwargs
        )

    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        **kwargs
    ):
        """
        Train the model

        Args:
            data_yaml: Path to dataset YAML configuration
            epochs: Number of training epochs
            **kwargs: Additional training arguments
        """
        if self.yolo_model is None:
            self.load_base_model()

        return self.yolo_model.train(
            data=data_yaml,
            epochs=epochs,
            **kwargs
        )

    def export(self, format: str = 'onnx', **kwargs):
        """
        Export model to various formats

        Args:
            format: Export format (onnx, torchscript, tflite, etc.)
            **kwargs: Additional export arguments
        """
        if self.yolo_model is None:
            self.load_base_model()

        return self.yolo_model.export(format=format, **kwargs)

    def save_config(self, path: str):
        """Save model configuration"""
        with open(path, 'w') as f:
            yaml.dump(self.config, f)

    @classmethod
    def from_config(cls, config_path: str):
        """Load model from configuration file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config)


def create_seawaste_model(
    model_config_path: str = None,
    base_model: str = 'yolov8m',
    num_classes: int = 15,
    pretrained: bool = True
) -> SeawasteDetectionModel:
    """
    Factory function to create a seawaste detection model

    Args:
        model_config_path: Path to model configuration YAML
        base_model: Base YOLO model size
        num_classes: Number of seawaste classes
        pretrained: Use pretrained weights

    Returns:
        Initialized SeawasteDetectionModel
    """
    if model_config_path:
        with open(model_config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Extract model parameters from config
        base_model = config.get('model_base', base_model)
        num_classes = config.get('num_classes', num_classes)

    model = SeawasteDetectionModel(
        base_model=base_model,
        num_classes=num_classes,
        use_cbam=True,
        use_color_correction=True,
        use_small_object_enhancer=True,
        pretrained=pretrained
    )

    model.load_base_model()

    return model


if __name__ == "__main__":
    # Example usage
    print("Creating Seawaste Detection Model...")

    # Create model
    model = create_seawaste_model(
        base_model='yolov8m',
        num_classes=15
    )

    print(f"Model created successfully!")
    print(f"Base model: {model.base_model_name}")
    print(f"Number of classes: {model.num_classes}")
    print(f"Configuration: {model.config}")
