from setuptools import setup, find_packages

setup(
    name="detector",
    version="0.1.0",
    description="RT-DETR v2 ONNX Inference Package",
    packages=find_packages(exclude=("tests", "docs")),
    package_data={
        "detector": [
            "configs/*.yaml",
            "default/*.jpg",
        ],
    },
    include_package_data=True,
    install_requires=[
        # ML and data processing
        "numpy==1.24.3",
        "onnxruntime==1.19.2",
        "Pillow==10.4.0",

        # Utilities
        "PyYAML==5.3.1",
        "boto3==1.26.165",
    ],
    python_requires=">=3.8",
)
