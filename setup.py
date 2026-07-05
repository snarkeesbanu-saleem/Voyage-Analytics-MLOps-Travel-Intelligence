"""Setup script for the Voyage Analytics MLOps package."""

from pathlib import Path

from setuptools import find_packages, setup

# ---------------------------------------------------------------------------
# Read requirements from requirements.txt
# ---------------------------------------------------------------------------
_requirements_path = Path(__file__).resolve().parent / "requirements.txt"

install_requires: list[str] = []
if _requirements_path.exists():
    install_requires = [
        line.strip()
        for line in _requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

# ---------------------------------------------------------------------------
# Package metadata
# ---------------------------------------------------------------------------
setup(
    name="voyage-analytics",
    version="0.1.0",
    description="MLOps pipeline for travel analytics — flight pricing, "
                "gender classification, and churn prediction.",
    author="Voyage Analytics Team",
    packages=find_packages(),
    install_requires=install_requires,
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
