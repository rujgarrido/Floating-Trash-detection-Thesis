"""
Evaluation Script for Seawaste Detection Model
Evaluate model performance on test dataset and generate comprehensive metrics.
"""

import argparse
import sys
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import logging
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from ultralytics import YOLO
import json


def setup_logging(log_dir: Path):
    """Setup logging configuration"""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_model(weights_path: str, logger) -> YOLO:
    """Load trained model"""
    if not Path(weights_path).exists():
        logger.error(f"Model weights not found: {weights_path}")
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    logger.info(f"Loading model from: {weights_path}")
    model = YOLO(weights_path)
    logger.info("Model loaded successfully")

    return model


def evaluate_model(model: YOLO, data_yaml: str, logger):
    """
    Evaluate model on validation/test set

    Args:
        model: Loaded YOLO model
        data_yaml: Path to dataset configuration
        logger: Logger instance

    Returns:
        Evaluation results
    """
    logger.info("Starting model evaluation...")

    results = model.val(
        data=data_yaml,
        split='val',  # Can be 'val' or 'test'
        save_json=True,
        plots=True
    )

    return results


def calculate_per_class_metrics(results, class_names: List[str], logger) -> pd.DataFrame:
    """
    Calculate detailed metrics for each class

    Args:
        results: Evaluation results from YOLO
        class_names: List of class names
        logger: Logger instance

    Returns:
        DataFrame with per-class metrics
    """
    logger.info("Calculating per-class metrics...")

    metrics_data = []

    if hasattr(results, 'box'):
        box_metrics = results.box

        # Get per-class metrics
        if hasattr(box_metrics, 'class_result'):
            for i, class_name in enumerate(class_names):
                metrics_data.append({
                    'Class': class_name,
                    'Precision': float(box_metrics.p[i]) if hasattr(box_metrics, 'p') else 0.0,
                    'Recall': float(box_metrics.r[i]) if hasattr(box_metrics, 'r') else 0.0,
                    'mAP50': float(box_metrics.ap50[i]) if hasattr(box_metrics, 'ap50') else 0.0,
                    'mAP50-95': float(box_metrics.ap[i]) if hasattr(box_metrics, 'ap') else 0.0,
                })

    df = pd.DataFrame(metrics_data)

    # Calculate F1 score
    if not df.empty:
        df['F1-Score'] = 2 * (df['Precision'] * df['Recall']) / (df['Precision'] + df['Recall'] + 1e-6)

    return df


def plot_metrics(metrics_df: pd.DataFrame, save_dir: Path, logger):
    """
    Generate visualization plots for metrics

    Args:
        metrics_df: DataFrame with metrics
        save_dir: Directory to save plots
        logger: Logger instance
    """
    logger.info("Generating metric visualizations...")

    save_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)

    # 1. Per-class performance comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    metrics_to_plot = ['Precision', 'Recall', 'mAP50', 'F1-Score']

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]

        if metric in metrics_df.columns:
            metrics_df.plot(
                x='Class',
                y=metric,
                kind='bar',
                ax=ax,
                legend=False,
                color='steelblue'
            )
            ax.set_title(f'{metric} per Class', fontsize=14, fontweight='bold')
            ax.set_xlabel('Class', fontsize=12)
            ax.set_ylabel(metric, fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'per_class_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved per-class metrics plot: {save_dir / 'per_class_metrics.png'}")

    # 2. Precision-Recall comparison
    fig, ax = plt.subplots(figsize=(12, 8))

    x = np.arange(len(metrics_df))
    width = 0.35

    ax.bar(x - width/2, metrics_df['Precision'], width, label='Precision', alpha=0.8)
    ax.bar(x + width/2, metrics_df['Recall'], width, label='Recall', alpha=0.8)

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Precision vs Recall by Class', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df['Class'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'precision_recall_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved precision-recall comparison: {save_dir / 'precision_recall_comparison.png'}")

    # 3. Overall metrics summary
    fig, ax = plt.subplots(figsize=(10, 6))

    summary_metrics = {
        'Mean Precision': metrics_df['Precision'].mean(),
        'Mean Recall': metrics_df['Recall'].mean(),
        'Mean mAP50': metrics_df['mAP50'].mean(),
        'Mean mAP50-95': metrics_df['mAP50-95'].mean(),
        'Mean F1-Score': metrics_df['F1-Score'].mean()
    }

    colors = plt.cm.Set3(range(len(summary_metrics)))
    bars = ax.bar(summary_metrics.keys(), summary_metrics.values(), color=colors, alpha=0.8)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Overall Model Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=15, ha='right')

    plt.tight_layout()
    plt.savefig(save_dir / 'overall_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved overall metrics plot: {save_dir / 'overall_metrics.png'}")


def save_evaluation_report(
    results,
    metrics_df: pd.DataFrame,
    save_dir: Path,
    logger
):
    """
    Save comprehensive evaluation report

    Args:
        results: Evaluation results
        metrics_df: Per-class metrics DataFrame
        save_dir: Directory to save report
        logger: Logger instance
    """
    logger.info("Generating evaluation report...")

    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'overall_metrics': {},
        'per_class_metrics': metrics_df.to_dict('records')
    }

    # Extract overall metrics
    if hasattr(results, 'box'):
        box = results.box
        report['overall_metrics'] = {
            'mAP50': float(box.map50) if hasattr(box, 'map50') else 0.0,
            'mAP50-95': float(box.map) if hasattr(box, 'map') else 0.0,
            'mean_precision': float(metrics_df['Precision'].mean()),
            'mean_recall': float(metrics_df['Recall'].mean()),
            'mean_f1_score': float(metrics_df['F1-Score'].mean())
        }

    # Save as JSON
    json_path = save_dir / 'evaluation_report.json'
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=4)

    logger.info(f"Saved evaluation report (JSON): {json_path}")

    # Save as YAML
    yaml_path = save_dir / 'evaluation_report.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(report, f, default_flow_style=False)

    logger.info(f"Saved evaluation report (YAML): {yaml_path}")

    # Save metrics as CSV
    csv_path = save_dir / 'per_class_metrics.csv'
    metrics_df.to_csv(csv_path, index=False)

    logger.info(f"Saved per-class metrics (CSV): {csv_path}")

    # Print summary to console
    logger.info("\n" + "="*60)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*60)
    logger.info(f"\nOverall Metrics:")
    for metric, value in report['overall_metrics'].items():
        logger.info(f"  {metric:20s}: {value:.4f}")

    logger.info(f"\nPer-Class Metrics:")
    logger.info(metrics_df.to_string(index=False))
    logger.info("="*60)


def run_evaluation(args):
    """Main evaluation function"""

    # Setup logging
    save_dir = Path(args.save_dir) / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(save_dir)

    logger.info("="*60)
    logger.info("SEAWASTE DETECTION MODEL EVALUATION")
    logger.info("="*60)

    # Load model
    model = load_model(args.weights, logger)

    # Load dataset config to get class names
    logger.info(f"Loading dataset configuration: {args.data}")
    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)

    class_names = list(data_config['names'].values())
    logger.info(f"Number of classes: {len(class_names)}")

    # Run evaluation
    try:
        results = evaluate_model(model, args.data, logger)

        # Calculate per-class metrics
        metrics_df = calculate_per_class_metrics(results, class_names, logger)

        # Generate plots
        if args.plots:
            plot_metrics(metrics_df, save_dir, logger)

        # Save evaluation report
        save_evaluation_report(results, metrics_df, save_dir, logger)

        logger.info("\n" + "="*60)
        logger.info("EVALUATION COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"\nResults saved to: {save_dir}")

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Evaluate Seawaste Detection Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='Path to trained model weights'
    )

    parser.add_argument(
        '--data',
        type=str,
        default='configs/seawaste_dataset.yaml',
        help='Path to dataset YAML configuration'
    )

    parser.add_argument(
        '--save-dir',
        type=str,
        default='./runs/eval',
        help='Directory to save evaluation results'
    )

    parser.add_argument(
        '--plots',
        action='store_true',
        default=True,
        help='Generate visualization plots'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_evaluation(args)
