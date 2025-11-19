"""
Training Script for Seawaste Detection Model
This script handles the complete training pipeline for marine debris detection.
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import torch
from datetime import datetime
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.seawaste_model import create_seawaste_model
from ultralytics import YOLO


def setup_logging(log_dir: Path):
    """Setup logging configuration"""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def validate_dataset(data_yaml: str, logger) -> bool:
    """Validate dataset structure and configuration"""
    if not Path(data_yaml).exists():
        logger.error(f"Dataset configuration not found: {data_yaml}")
        return False

    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)

    # Check required fields
    required_fields = ['path', 'train', 'val', 'nc', 'names']
    for field in required_fields:
        if field not in data_config:
            logger.error(f"Missing required field in dataset config: {field}")
            return False

    # Check if paths exist
    dataset_path = Path(data_config['path'])
    if not dataset_path.exists():
        logger.warning(f"Dataset path does not exist: {dataset_path}")
        logger.warning("Please ensure dataset is available before training")

    logger.info(f"Dataset configuration validated successfully")
    logger.info(f"Number of classes: {data_config['nc']}")
    logger.info(f"Classes: {list(data_config['names'].values())}")

    return True


def train_model(args):
    """Main training function"""

    # Setup logging
    log_dir = Path(args.save_dir) / 'logs'
    logger = setup_logging(log_dir)

    logger.info("="*60)
    logger.info("SEAWASTE DETECTION MODEL TRAINING")
    logger.info("="*60)

    # Load model configuration
    logger.info(f"Loading model configuration from: {args.model_config}")
    model_config = load_config(args.model_config)

    # Load dataset configuration
    logger.info(f"Loading dataset configuration from: {args.data}")
    if not validate_dataset(args.data, logger):
        logger.error("Dataset validation failed. Exiting...")
        return

    # Create results directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to: {save_dir}")

    # Check device
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Initialize model
    logger.info(f"Initializing model: {model_config.get('model_base', 'yolov8m')}")

    try:
        # Load base YOLO model
        base_model = model_config.get('model_base', 'yolov8m')

        if args.resume and args.weights:
            logger.info(f"Resuming training from: {args.weights}")
            model = YOLO(args.weights)
        else:
            logger.info(f"Starting fresh training with {base_model}")
            model = YOLO(f'{base_model}.pt')

    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        raise

    # Prepare training arguments
    hyperparams = model_config.get('hyperparameters', {})

    train_args = {
        'data': args.data,
        'epochs': args.epochs or hyperparams.get('epochs', 300),
        'batch': args.batch_size or hyperparams.get('batch_size', 16),
        'imgsz': args.imgsz or hyperparams.get('imgsz', 640),
        'device': device,
        'workers': args.workers or model_config.get('workers', 8),
        'project': args.project or model_config.get('logging', {}).get('project', 'seawaste_detection'),
        'name': args.name or model_config.get('logging', {}).get('name', 'custom_model'),
        'exist_ok': True,
        'pretrained': not args.no_pretrained,
        'optimizer': hyperparams.get('optimizer', 'Adam'),
        'lr0': hyperparams.get('lr0', 0.01),
        'lrf': hyperparams.get('lrf', 0.01),
        'momentum': hyperparams.get('momentum', 0.937),
        'weight_decay': hyperparams.get('weight_decay', 0.0005),
        'warmup_epochs': hyperparams.get('warmup_epochs', 3.0),
        'box': hyperparams.get('box', 7.5),
        'cls': hyperparams.get('cls', 0.5),
        'dfl': hyperparams.get('dfl', 1.5),
        'patience': args.patience or hyperparams.get('patience', 50),
        'save': True,
        'save_period': args.save_period or model_config.get('logging', {}).get('save_period', 10),
        'cache': args.cache,
        'val': True,
        'plots': True,
        'verbose': args.verbose,
    }

    # Add augmentation parameters
    augmentation = model_config.get('augmentation', {})
    if augmentation.get('enable', True):
        train_args.update({
            'hsv_h': augmentation.get('hsv_h', 0.015),
            'hsv_s': augmentation.get('hsv_s', 0.7),
            'hsv_v': augmentation.get('hsv_v', 0.4),
            'degrees': augmentation.get('degrees', 0.0),
            'translate': augmentation.get('translate', 0.1),
            'scale': augmentation.get('scale', 0.5),
            'shear': augmentation.get('shear', 0.0),
            'perspective': augmentation.get('perspective', 0.0),
            'flipud': augmentation.get('flipud', 0.0),
            'fliplr': augmentation.get('fliplr', 0.5),
            'mosaic': augmentation.get('mosaic', 1.0),
            'mixup': augmentation.get('mixup', 0.0),
        })

    logger.info("\nTraining Configuration:")
    logger.info("-" * 60)
    for key, value in train_args.items():
        logger.info(f"{key:20s}: {value}")
    logger.info("-" * 60)

    # Start training
    logger.info("\nStarting training...")
    try:
        results = model.train(**train_args)

        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*60)

        # Log final metrics
        if hasattr(results, 'results_dict'):
            logger.info("\nFinal Metrics:")
            for metric, value in results.results_dict.items():
                logger.info(f"{metric}: {value}")

        # Save final model
        final_model_path = save_dir / 'final_model.pt'
        logger.info(f"\nSaving final model to: {final_model_path}")

    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"\nError during training: {e}")
        raise

    logger.info("\nTraining script finished")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train Seawaste Detection Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '--data',
        type=str,
        default='configs/seawaste_dataset.yaml',
        help='Path to dataset YAML configuration'
    )

    parser.add_argument(
        '--model-config',
        type=str,
        default='configs/seawaste_model.yaml',
        help='Path to model configuration YAML'
    )

    # Training parameters
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--imgsz', type=int, help='Input image size')
    parser.add_argument('--device', type=str, help='Device to use (cuda/cpu/mps)')
    parser.add_argument('--workers', type=int, help='Number of dataloader workers')

    # Model parameters
    parser.add_argument('--weights', type=str, help='Path to initial weights')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    parser.add_argument('--no-pretrained', action='store_true', help='Start without pretrained weights')

    # Logging and saving
    parser.add_argument('--project', type=str, help='Project name')
    parser.add_argument('--name', type=str, help='Experiment name')
    parser.add_argument('--save-dir', type=str, default='./runs/train', help='Directory to save results')
    parser.add_argument('--save-period', type=int, help='Save checkpoint every N epochs')

    # Other parameters
    parser.add_argument('--patience', type=int, help='Early stopping patience')
    parser.add_argument('--cache', action='store_true', help='Cache images for faster training')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_model(args)
