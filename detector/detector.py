import os
import sys
import io
import time
import numpy as np
from PIL import Image
from typing import List, Optional, Dict, Any
import onnxruntime as ort
import yaml

# Import configuration and CLO value processing functions
from .config import config
from .clo_processor import map_detections_to_clo, get_base_clo_value
from .postprocessor import get_postprocessor


# Setup logging for detector service
import logging
import os

detector_logger = logging.getLogger(__name__)


def setup_detector_logging():
    """Setup logging for the detector service with relative paths."""
    # Get detector directory for relative path resolution
    detector_dir = os.path.dirname(os.path.abspath(__file__))

    # Configure logging
    log_level = getattr(logging, config.logging.level.upper(), logging.INFO)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    if config.logging.enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler - only create if file logging is explicitly enabled
    if config.logging.enable_file and config.logging.file and config.logging.file is not None:
        # Resolve log file path relative to detector directory
        log_file_path = os.path.join(detector_dir, config.logging.file)

        # Only create log directory if file logging is enabled
        log_dir = os.path.dirname(log_file_path)
        os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Setup logging
# detector_logger = setup_detector_logging()
detector_logger = logging.getLogger(__name__)


# Detection result as dictionary
def create_detection_result(
    class_id: int,
    class_name: str,
    confidence: float,
    bbox: List[float],
    clo_value: Optional[float] = None,
    zone: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a detection result dictionary."""
    return {
        "class_id": class_id,
        "class_name": class_name,
        "confidence": confidence,
        "bbox": bbox,  # [x1, y1, x2, y2]
        "clo_value": clo_value,
        "zone": zone,  # "left" or "right"
    }


# Zone data as dictionary
def create_zone_data(detections: List[Dict[str, Any]], total_clo_value: Optional[float] = None) -> Dict[str, Any]:
    """Create a zone data dictionary."""
    return {"detections": detections, "total_clo_value": total_clo_value}


# Response model as dictionary
def create_inference_response(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Create an inference response dictionary."""
    return {"left": left, "right": right}


# Global model variables
model_session = None
class_names = []


# Load class names from YAML file
def load_class_names(yaml_file):
    try:
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)

        # Extract category names and create a list indexed by ID
        categories = data.get("categories", {})

        # Find the maximum ID to determine list size
        max_id = max(categories.keys()) if categories else 0

        # Create a list where index corresponds to class ID
        labels = [""] * (max_id + 1)  # Initialize with empty strings

        for class_id, class_name in categories.items():
            labels[class_id] = class_name

        return labels
    except Exception as e:
        detector_logger.error(f"Error loading labels from {yaml_file}: {e}")
        return []


# Load ONNX model
def load_model(onnx_file: str = None, device: str = "cpu", categories_file: str = None):
    global model_session, class_names

    if model_session is None:
        try:
            # Use config defaults if not provided
            if onnx_file is None:
                onnx_file = config.model.onnx_file
            if categories_file is None:
                categories_file = config.paths.categories_file

            # Check if the model file exists
            model_path = onnx_file

            # Try different paths for the model file
            if not os.path.exists(model_path):
                # Try relative to detector directory
                detector_dir = os.path.dirname(os.path.abspath(__file__))
                alt_model_path = os.path.join(detector_dir, "models/model.onnx")
                if os.path.exists(alt_model_path):
                    model_path = alt_model_path
                    detector_logger.info(f"Using model from detector directory: {model_path}")
                else:
                    # Try to download the model using the original S3ModelDownloader
                    detector_logger.warning(f"Model file not found: {model_path} or {alt_model_path}")
                    detector_logger.info("Attempting to download model...")

                    try:
                        from .download_model import S3ModelDownloader

                        downloader = S3ModelDownloader()
                        downloader.download_all()

                        # Check if model was downloaded
                        if os.path.exists(alt_model_path):
                            model_path = alt_model_path
                            detector_logger.info(f"Successfully downloaded model to: {model_path}")
                        else:
                            raise FileNotFoundError(
                                f"Could not download model. Please manually download the RT-DETR v2 ONNX model and place it at: {alt_model_path}"
                            )
                    except Exception as e:
                        detector_logger.warning(f"Model download failed: {e}")
                        raise FileNotFoundError(
                            f"Model file not found: {model_path} or {alt_model_path}. Please download the RT-DETR v2 ONNX model and place it at: {alt_model_path}"
                        )

            # Set execution providers
            if device.lower() == "gpu":
                try:
                    providers = [("CUDAExecutionProvider", {}), ("CPUExecutionProvider", {})]
                    detector_logger.info("Attempting GPU acceleration with CUDA")
                except:
                    providers = [("CPUExecutionProvider", {})]
                    detector_logger.info("GPU not available, falling back to CPU")
            else:
                providers = [("CPUExecutionProvider", {})]
                detector_logger.info("Using CPU execution (stable and reliable)")

            # Create ONNX Runtime session
            model_session = ort.InferenceSession(model_path, providers=providers)

            # Log available providers and current device
            detector_logger.info(f"Available providers: {ort.get_available_providers()}")
            detector_logger.info(f"Current provider: {model_session.get_providers()}")

            # Load class names
            if os.path.exists(categories_file):
                class_names = load_class_names(categories_file)
                detector_logger.info(f"Loaded {len(class_names)} class names from {categories_file}")
            else:
                # Try relative to detector directory
                detector_dir = os.path.dirname(os.path.abspath(__file__))
                alt_categories_file = os.path.join(detector_dir, "configs/categories.yaml")
                if os.path.exists(alt_categories_file):
                    class_names = load_class_names(alt_categories_file)
                    detector_logger.info(
                        f"Loaded {len(class_names)} class names from detector directory: {alt_categories_file}"
                    )
                else:
                    detector_logger.warning(f"Categories file not found: {categories_file} or {alt_categories_file}")
                    class_names = []

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")


# Preprocess image
def preprocess_image(image_data, target_size=(640, 640)):
    """Preprocess image data"""
    # Convert bytes to PIL Image
    img = Image.open(io.BytesIO(image_data)).convert("RGB")
    orig_w, orig_h = img.size

    # Resize image
    img_resized = img.resize(target_size, Image.Resampling.LANCZOS)

    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(img_resized, dtype=np.float32) / 255.0

    # Transpose from HWC (Height, Width, Channel) to CHW (Channel, Height, Width)
    img_chw = np.transpose(img_array, (2, 0, 1))

    # Add batch dimension: (3, 640, 640) -> (1, 3, 640, 640)
    img_tensor = np.expand_dims(img_chw, axis=0)

    return img_tensor, (orig_w, orig_h)


# Process image for inference
def process_image(
    image_data: bytes,
    img_size: int = 640,
    conf_thres: float = 0.6,
    device: str = "cpu",
    onnx_file: str = None,
    categories_file: str = None,
    source_id: str = "default",
) -> Dict[str, Any]:
    try:
        # Load model if not already loaded
        load_model(onnx_file, device, categories_file)

        # Preprocess image
        im_data, (orig_w, orig_h) = preprocess_image(image_data, (img_size, img_size))
        orig_size = np.array([[orig_w, orig_h]], dtype=np.int64)

        # Run inference
        output = model_session.run(output_names=None, input_feed={"images": im_data, "orig_target_sizes": orig_size})

        labels, boxes, scores = output

        # Process detections
        detections = []
        for i in range(len(labels[0])):
            label = int(labels[0][i])
            score = float(scores[0][i])
            box = boxes[0][i]

            # Apply confidence threshold
            if score >= conf_thres:
                # Get class name
                class_name = class_names[label] if label < len(class_names) and class_names[label] else f"Class_{label}"

                detections.append(
                    create_detection_result(
                        class_id=label,
                        class_name=class_name,
                        confidence=score,
                        bbox=[float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                    )
                )

        # Apply temporal postprocessing if enabled
        if config.postprocessing.enabled:
            postprocessor = get_postprocessor()
            detections = postprocessor.postprocess_with_cache(source_id, detections)
            detector_logger.debug(f"Applied postprocessing for source {source_id}")

        # Apply CLO value mapping post-processing
        if config.clo.enabled:
            detections, total_clo_value = map_detections_to_clo(detections, config.clo.values_file)
        else:
            total_clo_value = None

        # Separate detections by zone
        left_detections = [d for d in detections if d["zone"] == "left"]
        right_detections = [d for d in detections if d["zone"] == "right"]

        # Calculate CLO values for each zone
        left_clo_value = sum(d["clo_value"] for d in left_detections if d["clo_value"] is not None)
        right_clo_value = sum(d["clo_value"] for d in right_detections if d["clo_value"] is not None)

        # Note: Base CLO value is already included in the total_clo_value from map_detections_to_clo
        # We need to distribute it to the appropriate zone
        if config.clo.enabled and total_clo_value is not None:
            # Since all detections are currently assigned to left zone,
            # assign the full total (including base) to left zone
            left_clo_value = total_clo_value

        return create_inference_response(
            left=create_zone_data(
                detections=left_detections, total_clo_value=left_clo_value if left_clo_value > 0 else None
            ),
            right=create_zone_data(
                detections=right_detections, total_clo_value=right_clo_value if right_clo_value > 0 else None
            ),
        )

    except Exception as e:
        raise RuntimeError(f"Inference failed: {str(e)}")


def detect(
    img_input=None,
    conf_thres: float = 0.6,
    img_size: int = 640,
    device: str = "cpu",
    onnx_file: str = None,
    categories_file: str = None,
    source_id: str = "default",
) -> Dict[str, Any]:
    """
    Detect clothing items in an image.

    Args:
        img_input: Can be one of:
            - str: Path to the image file
            - bytes: Raw image data
            - None: Use the default image (default)
        conf_thres: Confidence threshold for detections (default: 0.6)
        img_size: Input image size (default: 640)
        device: Device to use for inference - 'cpu' or 'gpu' (default: 'cpu')
        onnx_file: Path to ONNX model file (uses config default if None)
        categories_file: Path to categories YAML file (uses config default if None)
        source_id: Unique identifier for the source (e.g., camera ID) for postprocessing (default: "default")

        Returns:
        Dictionary containing detection results in JSON format

    Raises:
        FileNotFoundError: If image file is not found
        RuntimeError: If model loading or inference fails
    """

    # Handle different input types
    if img_input is None:
        # Use default image
        default_image_path = config.paths.default_image

        # Make path relative to detector script location
        if not os.path.isabs(default_image_path):
            detector_dir = os.path.dirname(os.path.abspath(__file__))
            default_image_path = os.path.join(detector_dir, default_image_path)

        detector_logger.info(f"Using default image: {default_image_path}")

        # Check if default image exists
        if not os.path.exists(default_image_path):
            raise FileNotFoundError(f"Default image file not found: {default_image_path}")

        # Read default image file
        try:
            with open(default_image_path, "rb") as f:
                image_data = f.read()
            detector_logger.debug(f"Successfully read default image file, size: {len(image_data)} bytes")
        except Exception as e:
            raise RuntimeError(f"Error reading default image file: {e}")

    elif isinstance(img_input, str):
        # Handle image path
        detector_logger.info(f"Processing image: {img_input}")

        # Check if image exists
        if not os.path.exists(img_input):
            raise FileNotFoundError(f"Image file not found: {img_input}")

        # Read image file
        try:
            with open(img_input, "rb") as f:
                image_data = f.read()
            detector_logger.debug(f"Successfully read image file, size: {len(image_data)} bytes")
        except Exception as e:
            raise RuntimeError(f"Error reading image file: {e}")

    elif isinstance(img_input, bytes):
        # Handle raw image data
        detector_logger.info("Processing raw image data")
        image_data = img_input
        detector_logger.debug(f"Using provided image data, size: {len(image_data)} bytes")

    else:
        raise ValueError(
            f"Invalid img_input type: {type(img_input)}. Must be str (path), bytes (image data), or None (default image)"
        )

    # Process image
    detector_logger.info("Processing image for detection...")
    result = process_image(image_data, img_size, conf_thres, device, onnx_file, categories_file, source_id)
    total_detections = len(result["left"]["detections"]) + len(result["right"]["detections"])
    detector_logger.info(f"Image processed, found {total_detections} detections")

    return result
