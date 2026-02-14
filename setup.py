from setuptools import setup, find_packages

setup(
    name="arc-nodsl",
    version="0.1.0",
    description="SPARC: Slot Programs via Active Radiation for ARC",

    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "einops>=0.7.0",
        "hydra-core>=1.3.0",
        "tensorboard>=2.15.0",
        "tqdm>=4.66.0",
        "scipy>=1.11.0",
        "matplotlib>=3.8.0",
        "Pillow>=10.0.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "isort>=5.12.0",
        ],
        "logging": [
            "wandb>=0.16.0",
        ],
    },
)
