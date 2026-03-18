<!-- markdownlint-disable MD033 MD041 -->

<div align="center">

# AA-SI Visualization

**Visualization tools for NOAA Fisheries Active Acoustics Strategic Initiative (AA-SI) echogram and sonar data**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Overview](#overview) •
[Getting Started](#getting-started) •
[Development](#development) •
[Project Structure](#project-structure)

</div>

---

## Overview

`aa-si-visualization` provides echogram plotting and visualization utilities for
active acoustics data processed through the AA-SI pipeline. Key capabilities include:

- **Sv echograms** — plot volume backscattering strength with configurable depth, ping range, and color scales
- **Cluster echograms** — visualize ML clustering results with categorical coloring
- **ML feature echograms** — display regridded/normalized ML feature data
- **Calibration difference echograms** — side-by-side comparison of baseline vs. calibrated Sv with difference panels
- **Flexible axes** — x-axis in seconds, pings, bins, or meters; y-axis in meters or range samples
- **Overlay lines** — add annotation lines on echogram plots

## Getting Started

### Requirements

- Python 3.10 or higher
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/nmfs-ost/AA-SI_Visualization.git
cd AA-SI_Visualization

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
```

## Usage

```python
from aa_si_visualization import plot_sv_echogram, plot_cluster_echogram

# Plot a standard Sv echogram
plot_sv_echogram(ds_Sv, frequency_nominal=frequencies)

# Plot ML cluster results
plot_cluster_echogram(ds_ml_ready, dataset_name="my_model", specific_data_name="clusters")
```

## Development

### Running Tests

```bash
pytest
pytest --cov=aa_si_visualization
```

### Code Quality

```bash
black src/ tests/
pylint src/aa_si_visualization
pre-commit run --all-files
```

### Building

```bash
pip install build
python -m build
```

---

## Project Structure

```
├── .gitignore
├── .pre-commit-config.yaml
├── .pylintrc
├── CHANGELOG.md
├── LICENSE
├── NOTICE
├── pyproject.toml
├── README.md
├── src/
│   └── aa_si_visualization/
│       ├── __init__.py
│       ├── assorted.py
│       ├── echogram.py
│       └── echogram_handlers.py
└── tests/
    ├── conftest.py
    └── test_package.py
```

---

## License

This project uses the Apache License 2.0. See [LICENSE](LICENSE) for details.

---

## Disclaimer

This repository is a scientific product and is not official communication of the National Oceanic and Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project code is provided on an ‘as is’ basis and the user assumes responsibility for its use. Any claims against the Department of Commerce or Department of Commerce bureaus stemming from the use of this GitHub project will be governed by all applicable Federal law. Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by the Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to imply endorsement of any commercial product or activity by DOC or the United States Government.
