"""Setup script for SoFlow package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="soflow",
    version="0.1.0",
    author="SoFlow Team",
    description="Solution Flow Models for One-Step Generative Modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Gaurav14cs17/GenAI",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=23.0",
            "isort>=5.0",
            "flake8>=6.0",
        ],
        "fid": [
            "clean-fid>=0.1.35",
            "torch-fidelity>=0.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "soflow-train=scripts.train:main",
            "soflow-sample=scripts.sample:main",
            "soflow-evaluate=scripts.evaluate:main",
        ],
    },
)

