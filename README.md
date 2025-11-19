# Seawaste Detection Model

A custom deep learning model for detecting marine debris and waste in underwater/ocean imagery. This project implements a state-of-the-art object detection system optimized for identifying various types of marine pollution.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
  - [Evaluation](#evaluation)
- [Model Architecture](#model-architecture)
- [Configuration](#configuration)
- [Results](#results)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Overview

This research project focuses on automated detection of marine debris using computer vision and deep learning techniques. The model is specifically designed to identify 15 different categories of seawaste, including plastic bottles, fishing nets, styrofoam, and other common marine pollutants.

### Key Objectives

- Develop an accurate and efficient object detection model for marine debris
- Handle challenging underwater/marine imaging conditions (turbidity, lighting variations, color distortion)
- Support real-time or near-real-time detection for practical applications
- Provide comprehensive evaluation metrics for research purposes

## Features

- **Custom YOLO-based Architecture**: Optimized YOLOv8 model with marine-specific enhancements
- **Marine Image Preprocessing**: Specialized preprocessing for underwater imagery
  - Underwater color correction
  - Haze/turbidity removal
  - Contrast enhancement (CLAHE)
  - White balance adjustment
- **15 Seawaste Categories**: Comprehensive classification of common marine debris
- **Advanced Augmentation**: Marine-specific data augmentation techniques
- **Attention Mechanisms**: CBAM (Convolutional Block Attention Module) for improved feature extraction
- **Small Object Detection**: Enhanced detection capabilities for distant/small debris
- **Comprehensive Evaluation**: Detailed metrics and visualization tools
- **Multiple Export Formats**: Support for PyTorch, ONNX, TensorFlow Lite, and more

## Project Structure

```
Object-detection-Thesis/
├── configs/
│   ├── seawaste_dataset.yaml      # Dataset configuration
│   └── seawaste_model.yaml        # Model hyperparameters
├── data/
│   └── datasets/
│       └── seawaste/              # Dataset directory
│           ├── images/
│           │   ├── train/
│           │   ├── val/
│           │   └── test/
│           └── labels/
│               ├── train/
│               ├── val/
│               └── test/
├── models/
│   ├── __init__.py
│   └── seawaste_model.py          # Custom model architecture
├── scripts/
│   ├── train.py                   # Training script
│   ├── detect.py                  # Inference script
│   └── evaluate.py                # Evaluation script
├── utils/
│   ├── __init__.py
│   └── data_utils.py              # Data processing utilities
├── notebooks/                     # Jupyter notebooks for analysis
├── checkpoints/                   # Model checkpoints
├── results/                       # Training/evaluation results
│   ├── images/                    # Detection visualizations
│   └── metrics/                   # Metric plots and reports
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.7+ (for GPU training)
- 8GB+ RAM (16GB+ recommended)
- GPU with 6GB+ VRAM (for training)

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/Aztigma-Thesis/Object-detection-Thesis.git
cd Object-detection-Thesis
```

2. **Create virtual environment** (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Verify installation**

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Dataset Preparation

### Dataset Format

The model uses YOLO format for annotations. Each image should have a corresponding `.txt` file with annotations in the format:

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalized to [0, 1].

### Seawaste Classes

The model detects 15 categories of marine debris:

1. `plastic_bottle` - PET and HDPE bottles
2. `plastic_bag` - Single-use plastic bags
3. `fishing_net` - Abandoned fishing nets (ghost nets)
4. `fishing_line` - Monofilament/multifilament lines
5. `buoy` - Floating markers and buoys
6. `styrofoam` - Expanded polystyrene foam
7. `metal_can` - Aluminum/steel cans
8. `glass_bottle` - Glass containers
9. `food_container` - Plastic food packaging
10. `cigarette_butt` - Cigarette filters
11. `straw` - Plastic drinking straws
12. `rope` - Synthetic/natural fiber ropes
13. `cloth_fabric` - Textile materials
14. `wood_debris` - Processed wood waste
15. `other_plastic` - Miscellaneous plastic waste

### Preparing Your Dataset

1. **Organize your data**:

```bash
data/datasets/seawaste/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

2. **Split dataset** (if you have unsplit data):

```python
from utils.data_utils import split_dataset

split_dataset(
    source_dir='path/to/your/data',
    output_dir='data/datasets/seawaste',
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1
)
```

3. **Verify dataset structure**:

```python
from utils.data_utils import verify_dataset_structure

stats = verify_dataset_structure('configs/seawaste_dataset.yaml')
print(stats)
```

## Usage

### Training

Train the model using the training script:

```bash
# Basic training
python scripts/train.py \
    --data configs/seawaste_dataset.yaml \
    --model-config configs/seawaste_model.yaml \
    --epochs 300 \
    --batch-size 16

# Training with custom parameters
python scripts/train.py \
    --data configs/seawaste_dataset.yaml \
    --model-config configs/seawaste_model.yaml \
    --epochs 300 \
    --batch-size 16 \
    --imgsz 640 \
    --device cuda \
    --workers 8 \
    --project seawaste_detection \
    --name experiment_v1

# Resume training from checkpoint
python scripts/train.py \
    --data configs/seawaste_dataset.yaml \
    --model-config configs/seawaste_model.yaml \
    --weights runs/train/experiment_v1/weights/last.pt \
    --resume
```

#### Training Parameters

- `--data`: Path to dataset configuration YAML
- `--model-config`: Path to model configuration YAML
- `--epochs`: Number of training epochs (default: 300)
- `--batch-size`: Training batch size (default: 16)
- `--imgsz`: Input image size (default: 640)
- `--device`: Device to use (cuda/cpu/mps)
- `--workers`: Number of dataloader workers
- `--project`: Project name for logging
- `--name`: Experiment name
- `--resume`: Resume training from checkpoint
- `--weights`: Path to initial weights

### Inference

Run detection on images, videos, or streams:

```bash
# Detect in single image
python scripts/detect.py \
    --weights runs/train/experiment_v1/weights/best.pt \
    --source path/to/image.jpg \
    --conf 0.25

# Detect in video
python scripts/detect.py \
    --weights runs/train/experiment_v1/weights/best.pt \
    --source path/to/video.mp4 \
    --conf 0.25 \
    --show

# Detect in directory of images
python scripts/detect.py \
    --weights runs/train/experiment_v1/weights/best.pt \
    --source path/to/images/ \
    --conf 0.25 \
    --report

# Real-time webcam detection
python scripts/detect.py \
    --weights runs/train/experiment_v1/weights/best.pt \
    --source 0 \
    --conf 0.25

# RTSP stream
python scripts/detect.py \
    --weights runs/train/experiment_v1/weights/best.pt \
    --source rtsp://camera_ip:port/stream \
    --conf 0.25
```

#### Inference Parameters

- `--weights`: Path to trained model weights (.pt file)
- `--source`: Input source (image, video, directory, webcam index, or stream URL)
- `--conf`: Confidence threshold (default: 0.25)
- `--iou`: IoU threshold for NMS (default: 0.45)
- `--show`: Display results in real-time
- `--report`: Generate detection report with statistics

### Evaluation

Evaluate model performance on test set:

```bash
python scripts/evaluate.py \
    --weights runs/train/experiment_v1/weights/best.pt \
    --data configs/seawaste_dataset.yaml \
    --plots
```

The evaluation script generates:
- Per-class metrics (Precision, Recall, mAP50, mAP50-95, F1-Score)
- Confusion matrix
- Precision-Recall curves
- Overall performance metrics
- Detailed evaluation reports (JSON, YAML, CSV)

## Model Architecture

### Base Architecture

The model is built on YOLOv8 with the following enhancements:

- **Backbone**: CSPDarknet with depth and width multipliers
- **Neck**: PANet (Path Aggregation Network)
- **Head**: Custom detection head with anchor-free approach

### Custom Enhancements

1. **Marine Color Correction Module**: Learnable color correction for underwater imagery
2. **CBAM Attention**: Convolutional Block Attention Module for better feature representation
3. **Small Object Enhancer**: Multi-scale feature fusion for detecting distant debris
4. **Marine-Specific Preprocessing**:
   - Underwater color correction
   - Haze removal
   - Contrast enhancement

### Model Variants

- `yolov8n` - Nano (fastest, smallest)
- `yolov8s` - Small
- `yolov8m` - Medium (default, balanced)
- `yolov8l` - Large
- `yolov8x` - Extra Large (most accurate)

## Configuration

### Dataset Configuration (`configs/seawaste_dataset.yaml`)

Defines dataset paths, classes, and augmentation parameters.

### Model Configuration (`configs/seawaste_model.yaml`)

Defines model architecture, hyperparameters, and training settings:

- Learning rate schedule
- Optimizer settings
- Loss function weights
- Augmentation parameters
- Inference settings

## Results

Results from training and evaluation are saved in the `runs/` directory:

```
runs/
├── train/
│   └── experiment_v1/
│       ├── weights/
│       │   ├── best.pt
│       │   └── last.pt
│       ├── results.png
│       └── ...
├── detect/
│   └── detect_YYYYMMDD_HHMMSS/
│       ├── image1.jpg
│       └── detection_report.yaml
└── eval/
    └── eval_YYYYMMDD_HHMMSS/
        ├── per_class_metrics.csv
        ├── evaluation_report.json
        └── plots/
```

## Performance Benchmarks

(To be updated after training on your dataset)

Expected performance on well-annotated marine debris datasets:
- mAP50: 70-85%
- mAP50-95: 45-60%
- Inference speed: 30-60 FPS (on RTX 3080)

## Research Applications

This model can be used for:

- Autonomous marine debris collection systems
- Ocean cleanup monitoring
- Environmental impact assessment
- Marine pollution research
- Underwater robotics and ROV systems
- Coastal monitoring systems

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{seawaste_detection_2025,
  title={Custom Seawaste Detection Model for Marine Debris Identification},
  author={Your Name},
  year={2025},
  publisher={GitHub},
  howpublished={\\url{https://github.com/Aztigma-Thesis/Object-detection-Thesis}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv8 by Ultralytics
- Marine debris datasets from [list your sources]
- Research guidance from [your institution/advisors]

## Contact

For questions or collaboration opportunities:

- GitHub Issues: [Create an issue](https://github.com/Aztigma-Thesis/Object-detection-Thesis/issues)
- Email: [your-email@example.com]

## Future Work

- [ ] Implement additional detection architectures (Faster R-CNN, EfficientDet)
- [ ] Expand dataset with more diverse marine environments
- [ ] Add 3D bounding box detection for depth estimation
- [ ] Integrate with underwater drone control systems
- [ ] Deploy as web service API
- [ ] Mobile app for field deployment
- [ ] Real-time video analytics dashboard

---

**Note**: This is a research project. Model performance depends heavily on the quality and diversity of your training dataset. Ensure proper data collection and annotation for best results.
