from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cnn_framework",
    version="0.1",
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
    ],
    extras_require={':python_version == "3.7.*"': ["pickle5"]},
    python_requires=">=3.7",
)
