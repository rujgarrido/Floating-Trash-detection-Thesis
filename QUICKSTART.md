# Quick Start Guide

Get started with the Seawaste Detection Model in 5 minutes!

## Prerequisites

- Python 3.8+
- GPU with CUDA support (recommended)

## Installation

```bash
# Clone repository
git clone https://github.com/Aztigma-Thesis/Object-detection-Thesis.git
cd Object-detection-Thesis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Test (No Dataset Required)

Test the model structure without a dataset:

```python
from models.seawaste_model import create_seawaste_model

# Create model
model = create_seawaste_model(
    base_model='yolov8n',  # Use nano model for quick test
    num_classes=15
)

print("Model created successfully!")
```

## Prepare Your Dataset

1. **Organize your images and labels**:

```
data/datasets/seawaste/
├── images/
│   └── (your images here)
└── labels/
    └── (your YOLO format labels here)
```

2. **Split dataset**:

```python
from utils.data_utils import split_dataset

split_dataset(
    source_dir='data/datasets/seawaste',
    output_dir='data/datasets/seawaste',
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1
)
```

## Train Your First Model

```bash
python scripts/train.py \
    --data configs/seawaste_dataset.yaml \
    --model-config configs/seawaste_model.yaml \
    --epochs 50 \
    --batch-size 8 \
    --imgsz 640
```

Training tips:
- Start with fewer epochs (50-100) to test
- Reduce batch size if you get out-of-memory errors
- Use smaller model (`yolov8n` or `yolov8s`) for faster training

## Run Inference

After training, run detection on new images:

```bash
python scripts/detect.py \
    --weights runs/train/custom_model_v1/weights/best.pt \
    --source path/to/test/image.jpg \
    --conf 0.25 \
    --show
```

## Evaluate Model

```bash
python scripts/evaluate.py \
    --weights runs/train/custom_model_v1/weights/best.pt \
    --data configs/seawaste_dataset.yaml \
    --plots
```

## Common Issues

### Out of Memory (OOM)

Reduce batch size:
```bash
--batch-size 4  # or even 2
```

### Slow Training

Use smaller model:
```bash
--model-config configs/seawaste_model.yaml
# Edit the file and change model_base to 'yolov8n'
```

### No GPU Available

Force CPU training:
```bash
--device cpu
```

Note: CPU training will be significantly slower.

## Next Steps

1. **Collect more data**: The model improves with more diverse training examples
2. **Tune hyperparameters**: Adjust learning rate, batch size, augmentation
3. **Experiment with architectures**: Try different model sizes (n, s, m, l, x)
4. **Monitor training**: Use TensorBoard to track metrics
5. **Evaluate thoroughly**: Test on real-world data

## Getting Help

- Check [README.md](README.md) for detailed documentation
- Review example notebooks in `notebooks/`
- Open an issue on GitHub
- Review YOLO documentation: https://docs.ultralytics.com/

## Example Workflow

```bash
# 1. Verify installation
python -c "from models import create_seawaste_model; print('OK')"

# 2. Prepare dataset (if needed)
# Add your images to data/datasets/seawaste/

# 3. Train small model (quick test)
python scripts/train.py \
    --epochs 10 \
    --batch-size 8

# 4. Check results
ls runs/train/custom_model_v1/

# 5. Run detection
python scripts/detect.py \
    --weights runs/train/custom_model_v1/weights/best.pt \
    --source data/datasets/seawaste/images/val/ \
    --report

# 6. Evaluate
python scripts/evaluate.py \
    --weights runs/train/custom_model_v1/weights/best.pt
```

Happy detecting! 🌊🔍
