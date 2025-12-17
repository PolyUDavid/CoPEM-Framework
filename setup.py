#!/usr/bin/env python3
"""
CoPEM Framework Setup
Consensus-Driven Predictive Energy Management for Autonomous Emergency Braking

Installation:
    pip install -e .

For development:
    pip install -e ".[dev]"

Date: December 15, 2025
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as f:
        return f.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="copem-framework",
    version="1.0.0",
    author="[Your Names]",
    author_email="[your.email@institution.edu]",
    description="Consensus-Driven Predictive Energy Management for Energy-Positive AEB",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/PolyUDavid/CoPEM-Framework",
    project_urls={
        "Bug Reports": "https://github.com/PolyUDavid/CoPEM-Framework/issues",
        "Source": "https://github.com/PolyUDavid/CoPEM-Framework",
        "Paper": "[Link to paper]",
    },
    packages=find_packages(exclude=["tests", "experiments", "scripts"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.3.1",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
        "viz": [
            "plotly>=5.14.1",
            "dash>=2.11.0",
        ],
        "distributed": [
            "ray[default]>=2.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "copem-validate=scripts.validate_installation:main",
            "copem-train=scripts.train_copem:main",
            "copem-test=scripts.run_tests:main",
        ],
    },
    include_package_data=True,
    package_data={
        "copem": [
            "data/paper_data/*.json",
            "configs/*.yaml",
        ],
    },
    zip_safe=False,
    keywords=[
        "autonomous driving",
        "emergency braking",
        "energy recovery",
        "reinforcement learning",
        "consensus algorithm",
        "electric vehicles",
        "byzantine fault tolerance",
        "control barrier function",
    ],
)

