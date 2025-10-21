# RT-DETR v2 Clothing Detection Package

A lightweight Python package for clothing detection using RT-DETR v2 ONNX model with CLO (Clothing Insulation) value calculation. Simple one-function API that returns JSON results.

## 🚀 Features

- **Simple API**: One function `detect()` for all detection needs
- **Flexible Input**: Accepts image paths, raw image data, or uses default image
- **JSON Output**: Returns structured JSON data ready for any application
- **CLO Calculation**: Automatic clothing insulation value computation
- **Fast Inference**: Optimized ONNX runtime with CPU/GPU support
- **Auto Model Download**: Automatically downloads required models

## 🛠️ Installation

```bash
git clone <repository-url>
cd clothing-detector
pip install -e .
```

## ⚡ Quick Start

```python
from detector import detect

# Detect clothing in any image
result = detect("path/to/your/image.jpg")
print(result)
```

That's it! The model downloads automatically on first use.

## 📝 Usage Examples

### Basic Detection
```python
from detector import detect
import json

# Detect objects in your image
result = detect("my_photo.jpg")

# Pretty print the JSON result
print(json.dumps(result, indent=2))
```

### Different Input Types
```python
# Use default test image
result = detect()

# Use raw image data
with open("image.jpg", "rb") as f:
    result = detect(f.read())

# Custom settings
result = detect("image.jpg", conf_thres=0.8, device="gpu")
```

## 📊 Response Format

```json
{
  "left": {
    "detections": [
      {
        "class_id": 3,
        "class_name": "sweater",
        "confidence": 0.93,
        "bbox": [84.35, 189.04, 499.44, 400.10],
        "clo_value": 0.36,
        "zone": "left"
      }
    ],
    "total_clo_value": 0.67
  },
  "right": {
    "detections": [],
    "total_clo_value": null
  }
}
```

## 🔧 Function Parameters

```python
detect(img_input=None, conf_thres=0.6, img_size=640, device="cpu")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `img_input` | str/bytes/None | None | Image path, raw bytes, or None for default image |
| `conf_thres` | float | 0.6 | Confidence threshold (0.0-1.0) |
| `img_size` | int | 640 | Input image size for processing |
| `device` | str | "cpu" | Processing device ("cpu" or "gpu") |