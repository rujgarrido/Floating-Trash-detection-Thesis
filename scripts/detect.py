"""
Detection/Inference Script for Seawaste Detection Model
Run inference on images, videos, or camera streams to detect marine debris.
"""

import argparse
import sys
from pathlib import Path
import yaml
import cv2
import numpy as np
from typing import Union, List
import logging
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from ultralytics import YOLO


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def load_model(weights_path: str, logger) -> YOLO:
    """Load trained model weights"""
    if not Path(weights_path).exists():
        logger.error(f"Model weights not found: {weights_path}")
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    logger.info(f"Loading model from: {weights_path}")
    model = YOLO(weights_path)
    logger.info("Model loaded successfully")

    return model


def process_image(
    model: YOLO,
    image_path: str,
    conf_threshold: float,
    iou_threshold: float,
    save_dir: Path,
    show: bool = False
):
    """Process a single image"""
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        iou=iou_threshold,
        save=True,
        project=str(save_dir.parent),
        name=save_dir.name,
        show=show
    )

    return results


def process_video(
    model: YOLO,
    video_path: str,
    conf_threshold: float,
    iou_threshold: float,
    save_dir: Path,
    show: bool = False
):
    """Process a video file"""
    results = model.predict(
        source=video_path,
        conf=conf_threshold,
        iou=iou_threshold,
        save=True,
        project=str(save_dir.parent),
        name=save_dir.name,
        show=show,
        stream=True
    )

    return results


def process_stream(
    model: YOLO,
    source: Union[int, str],
    conf_threshold: float,
    iou_threshold: float,
    save_dir: Path
):
    """Process live stream (webcam or RTSP)"""
    results = model.predict(
        source=source,
        conf=conf_threshold,
        iou=iou_threshold,
        save=True,
        project=str(save_dir.parent),
        name=save_dir.name,
        show=True,
        stream=True
    )

    return results


def detect_in_directory(
    model: YOLO,
    directory: str,
    conf_threshold: float,
    iou_threshold: float,
    save_dir: Path,
    extensions: List[str] = None
):
    """Process all images in a directory"""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

    directory_path = Path(directory)
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    image_files = []
    for ext in extensions:
        image_files.extend(directory_path.glob(f'*{ext}'))
        image_files.extend(directory_path.glob(f'*{ext.upper()}'))

    if not image_files:
        print(f"No images found in {directory}")
        return

    print(f"Found {len(image_files)} images")

    results = model.predict(
        source=str(directory),
        conf=conf_threshold,
        iou=iou_threshold,
        save=True,
        project=str(save_dir.parent),
        name=save_dir.name
    )

    return results


def generate_detection_report(results, save_path: Path, logger):
    """Generate a detection report with statistics"""
    logger.info("Generating detection report...")

    report = {
        'total_images': 0,
        'total_detections': 0,
        'detections_per_class': {},
        'confidence_stats': {
            'mean': 0.0,
            'min': 1.0,
            'max': 0.0
        }
    }

    all_confidences = []

    for result in results:
        report['total_images'] += 1

        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes

            for box in boxes:
                report['total_detections'] += 1

                # Get class and confidence
                cls = int(box.cls)
                conf = float(box.conf)

                # Update class counts
                class_name = result.names[cls]
                if class_name not in report['detections_per_class']:
                    report['detections_per_class'][class_name] = 0
                report['detections_per_class'][class_name] += 1

                # Track confidences
                all_confidences.append(conf)

    # Calculate confidence statistics
    if all_confidences:
        report['confidence_stats']['mean'] = np.mean(all_confidences)
        report['confidence_stats']['min'] = np.min(all_confidences)
        report['confidence_stats']['max'] = np.max(all_confidences)

    # Save report
    report_path = save_path / 'detection_report.yaml'
    with open(report_path, 'w') as f:
        yaml.dump(report, f, default_flow_style=False)

    logger.info(f"Detection report saved to: {report_path}")

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("DETECTION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total images processed: {report['total_images']}")
    logger.info(f"Total detections: {report['total_detections']}")
    logger.info(f"\nDetections per class:")
    for class_name, count in sorted(report['detections_per_class'].items()):
        logger.info(f"  {class_name:25s}: {count}")
    logger.info(f"\nConfidence statistics:")
    logger.info(f"  Mean: {report['confidence_stats']['mean']:.3f}")
    logger.info(f"  Min:  {report['confidence_stats']['min']:.3f}")
    logger.info(f"  Max:  {report['confidence_stats']['max']:.3f}")
    logger.info("="*60)

    return report


def run_detection(args):
    """Main detection function"""
    logger = setup_logging()

    logger.info("="*60)
    logger.info("SEAWASTE DETECTION - INFERENCE")
    logger.info("="*60)

    # Load model
    model = load_model(args.weights, logger)

    # Create save directory
    save_dir = Path(args.save_dir) / f"detect_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to: {save_dir}")

    # Determine source type and run detection
    source = args.source

    logger.info(f"Source: {source}")
    logger.info(f"Confidence threshold: {args.conf}")
    logger.info(f"IoU threshold: {args.iou}")

    try:
        if source.isdigit():
            # Webcam
            logger.info("Running detection on webcam...")
            results = process_stream(
                model, int(source), args.conf, args.iou, save_dir
            )
        elif source.startswith(('rtsp://', 'http://', 'https://')):
            # Stream
            logger.info("Running detection on stream...")
            results = process_stream(
                model, source, args.conf, args.iou, save_dir
            )
        elif Path(source).is_file():
            # Check if video or image
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
            if Path(source).suffix.lower() in video_extensions:
                logger.info("Running detection on video...")
                results = process_video(
                    model, source, args.conf, args.iou, save_dir, args.show
                )
            else:
                logger.info("Running detection on image...")
                results = process_image(
                    model, source, args.conf, args.iou, save_dir, args.show
                )
        elif Path(source).is_dir():
            # Directory of images
            logger.info("Running detection on directory...")
            results = detect_in_directory(
                model, source, args.conf, args.iou, save_dir
            )
        else:
            logger.error(f"Invalid source: {source}")
            return

        # Generate report if requested
        if args.report and not isinstance(results, type(None)):
            generate_detection_report(results, save_dir, logger)

        logger.info("\nDetection completed successfully!")
        logger.info(f"Results saved to: {save_dir}")

    except KeyboardInterrupt:
        logger.warning("\nDetection interrupted by user")
    except Exception as e:
        logger.error(f"Error during detection: {e}")
        raise


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run inference with Seawaste Detection Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='Path to trained model weights (.pt file)'
    )

    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Source for detection (image path, video path, directory, webcam index, or stream URL)'
    )

    # Detection parameters
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold for detections'
    )

    parser.add_argument(
        '--iou',
        type=float,
        default=0.45,
        help='IoU threshold for NMS'
    )

    # Output parameters
    parser.add_argument(
        '--save-dir',
        type=str,
        default='./runs/detect',
        help='Directory to save detection results'
    )

    parser.add_argument(
        '--show',
        action='store_true',
        help='Show detection results in real-time'
    )

    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate detection report with statistics'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_detection(args)
