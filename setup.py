from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cnn_framework",
    version="0.0.8",
    author="Thomas Bonte",
    author_email="thomas.bonte@mines-paristech.fr",
    description="CNN framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://https://github.com/15bonte/cnn_framework",
    project_urls={
        "Bug Tracker": "https://https://github.com/15bonte/cnn_framework/issues"
    },
    classifiers=[
        "Programming Language :: Python :: 3.9",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch>=1.12.1",
        "scikit-image>=0.19.3",
        "matplotlib",
        "scikit-learn>=1.2.2",
        "pint>=0.19.2",
        "torchvision>=0.13.1",
        "albumentations==1.3.0",
        "torchmetrics>=0.11.4",
        "big-fish>=0.6.2",
        "pillow>=9.2.0",
        "segmentation-models-pytorch>=0.3.0",
        "protobuf==3.20.*",
        "tensorboard==2.8.0",
        "aicsimageio>=4.12.1",
    ],
    python_requires=">=3.9",
)
