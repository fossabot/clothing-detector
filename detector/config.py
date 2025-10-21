import os
from typing import Dict, Any, Optional, List


class ModelConfig:
    """Model configuration settings"""

    def __init__(
        self, onnx_file: str = "model.onnx", img_size: int = 640, conf_thres: float = 0.6, device: str = "cpu"
    ):
        self.onnx_file = onnx_file
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.device = device


class PathsConfig:
    """File path configuration settings"""

    def __init__(
        self,
        categories_file: str = "configs/categories.yaml",
        default_image: str = "./default/default_img.jpg",
        output_directory: str = "output",
    ):
        self.categories_file = categories_file
        self.default_image = default_image
        self.output_directory = output_directory


class InferenceConfig:
    """Inference settings"""

    def __init__(self, batch_size: int = 1, max_detections: int = 100):
        self.batch_size = batch_size
        self.max_detections = max_detections


class CLOConfig:
    """CLO processing configuration"""

    def __init__(self, enabled: bool = True, values_file: str = "configs/clo_values.yaml"):
        self.enabled = enabled
        self.values_file = values_file


class LoggingConfig:
    """Logging configuration"""

    def __init__(
        self,
        level: str = "INFO",
        enable_console: bool = True,
        enable_file: bool = False,
        file: Optional[str] = "./logs/detector.log",
    ):
        self.level = level
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.file = file if file is not None else None


class PostProcessingConfig:
    """Postprocessing configuration for temporal stabilization"""

    def __init__(
        self,
        enabled: bool = False,
        history_size: int = 5,
        expiry_seconds: float = 5.0,
        keep_alive_frames: int = 3,
        min_presence_ratio: float = 0.5,
        conf_thresholds: Optional[Dict[str, float]] = None,
        valid_layers: Optional[Dict[str, List[str]]] = None,
        rules_file: str = "configs/postprocessing.yaml",
    ):
        self.enabled = enabled
        self.history_size = history_size
        self.expiry_seconds = expiry_seconds
        self.keep_alive_frames = keep_alive_frames
        self.min_presence_ratio = min_presence_ratio
        self.conf_thresholds = conf_thresholds or {}
        self.valid_layers = valid_layers or {}
        self.rules_file = rules_file
    
    def _load_rules_from_file(self):
        """Load confidence thresholds and layer rules from YAML file."""
        try:
            import yaml
            
            # Get detector directory for relative path resolution
            detector_dir = os.path.dirname(os.path.abspath(__file__))
            rules_path = os.path.join(detector_dir, self.rules_file)
            
            if os.path.exists(rules_path):
                with open(rules_path, 'r') as f:
                    rules_data = yaml.safe_load(f)
                
                # Load confidence thresholds
                if 'conf_thresholds' in rules_data:
                    self.conf_thresholds.update(rules_data['conf_thresholds'])
                
                # Load valid layers
                if 'valid_layers' in rules_data:
                    self.valid_layers.update(rules_data['valid_layers'])
                
                print(f"Loaded postprocessing rules from {rules_path}")
            else:
                print(f"Postprocessing rules file not found: {rules_path}")
        except Exception as e:
            print(f"Error loading postprocessing rules: {e}")


class DetectorConfig:
    """Main detector configuration"""

    def __init__(self):
        self.model = ModelConfig()
        self.paths = PathsConfig()
        self.inference = InferenceConfig()
        self.clo = CLOConfig()
        self.logging = LoggingConfig()
        self.postprocessing = PostProcessingConfig()

    @classmethod
    def from_yaml(cls, yaml_file: str = "configs/config.yaml") -> "DetectorConfig":
        """Load configuration from YAML file"""
        try:
            import yaml

            # Get detector directory for relative path resolution
            detector_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(detector_dir, yaml_file)

            with open(config_path, "r") as f:
                data = yaml.safe_load(f)

            # Create config instance
            config = cls()

            # Update model config if present
            if "model" in data:
                model_data = data["model"]
                config.model = ModelConfig(
                    onnx_file=model_data.get("onnx_file", config.model.onnx_file),
                    img_size=model_data.get("img_size", config.model.img_size),
                    conf_thres=model_data.get("conf_thres", config.model.conf_thres),
                    device=model_data.get("device", config.model.device),
                )

            # Update paths config if present
            if "paths" in data:
                paths_data = data["paths"]
                config.paths = PathsConfig(
                    categories_file=paths_data.get("categories_file", config.paths.categories_file),
                    default_image=paths_data.get("default_image", config.paths.default_image),
                    output_directory=paths_data.get("output_directory", config.paths.output_directory),
                )

            # Update inference config if present
            if "inference" in data:
                inference_data = data["inference"]
                config.inference = InferenceConfig(
                    batch_size=inference_data.get("batch_size", config.inference.batch_size),
                    max_detections=inference_data.get("max_detections", config.inference.max_detections),
                )

            # Update CLO config if present
            if "clo" in data:
                clo_data = data["clo"]
                config.clo = CLOConfig(
                    enabled=clo_data.get("enabled", config.clo.enabled),
                    values_file=clo_data.get("values_file", config.clo.values_file),
                )

            # Update logging config if present
            if "logging" in data:
                logging_data = data["logging"]
                config.logging = LoggingConfig(
                    level=logging_data.get("level", config.logging.level),
                    enable_console=logging_data.get("enable_console", config.logging.enable_console),
                    enable_file=logging_data.get("enable_file", config.logging.enable_file),
                    file=logging_data.get("file", config.logging.file),
                )

            # Update postprocessing config if present
            if "postprocessing" in data:
                postprocessing_data = data["postprocessing"]
                config.postprocessing = PostProcessingConfig(
                    enabled=postprocessing_data.get("enabled", config.postprocessing.enabled),
                    history_size=postprocessing_data.get("history_size", config.postprocessing.history_size),
                    expiry_seconds=postprocessing_data.get("expiry_seconds", config.postprocessing.expiry_seconds),
                    keep_alive_frames=postprocessing_data.get("keep_alive_frames", config.postprocessing.keep_alive_frames),
                    min_presence_ratio=postprocessing_data.get("min_presence_ratio", config.postprocessing.min_presence_ratio),
                    conf_thresholds=postprocessing_data.get("conf_thresholds", config.postprocessing.conf_thresholds),
                    valid_layers=postprocessing_data.get("valid_layers", config.postprocessing.valid_layers),
                    rules_file=postprocessing_data.get("rules_file", config.postprocessing.rules_file),
                )
                
                # Load rules from file if specified
                if postprocessing_data.get("load_rules_from_file", True):
                    config.postprocessing._load_rules_from_file()

            return config
        except Exception as e:
            print(f"Error loading config from {yaml_file}: {e}")
            print("Using default configuration")
            return cls()

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "model": {
                "onnx_file": self.model.onnx_file,
                "img_size": self.model.img_size,
                "conf_thres": self.model.conf_thres,
                "device": self.model.device,
            },
            "paths": {
                "categories_file": self.paths.categories_file,
                "default_image": self.paths.default_image,
                "output_directory": self.paths.output_directory,
            },
            "inference": {"batch_size": self.inference.batch_size, "max_detections": self.inference.max_detections},
            "clo": {"enabled": self.clo.enabled, "values_file": self.clo.values_file},
            "logging": {
                "level": self.logging.level,
                "enable_console": self.logging.enable_console,
                "enable_file": self.logging.enable_file,
                "file": self.logging.file,
            },
            "postprocessing": {
                "enabled": self.postprocessing.enabled,
                "history_size": self.postprocessing.history_size,
                "expiry_seconds": self.postprocessing.expiry_seconds,
                "keep_alive_frames": self.postprocessing.keep_alive_frames,
                "min_presence_ratio": self.postprocessing.min_presence_ratio,
                "conf_thresholds": self.postprocessing.conf_thresholds,
                "valid_layers": self.postprocessing.valid_layers,
                "rules_file": self.postprocessing.rules_file,
            },
        }


# Global configuration instance
config = DetectorConfig.from_yaml()


# Export for easy access
__all__ = ["DetectorConfig", "config"]
