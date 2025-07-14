# S2GOS Scene Generator

Synthetic scene generator for the DTE-S2GOS service. This package generates realistic Earth observation scenes by combining Digital Elevation Models (DEM), land cover data, and spectral material properties into comprehensive 3D scene descriptions.

## Overview

The S2GOS Scene Generator creates synthetic scenes for Earth observation simulation by:

## Installation

### Prerequisites

- Python 3.9+
- [pixi](https://pixi.sh/) (recommended) or conda/mamba

### Development Installation

```bash
# Clone the repository
git clone <repository-url>
cd s2gos/packages/s2gos-generator

# Install development environment with pixi
pixi install -e dev

# Or install with pip in development mode
pip install -e .
```

## Data Requirements

The S2GOS Scene Generator requires several external data sources. Follow these steps to set up your data environment:

### 1. Digital Elevation Model (DEM) Data

**Source**: [Copernicus GLO-30 DEM](https://registry.opendata.aws/copernicus-dem/)

### 2. Land Cover Data

**Source**: [ESA WorldCover 2021](https://worldcover2021.esa.int/)

### 3. Material Configuration

**Included**: Pre-configured material library with spectral data

**Location**: `src/s2gos_generator/data/materials.json`

### Included Examples

- **`examples/simple_scene_generation.py`**: Basic scene generation workflow

### Running Examples

```bash
# Simple scene generation
pixi run python examples/simple_scene_generation.py
```

## Output Structure

Generated scenes follow this directory structure:

```
output/
└── scene_name/
    ├── scene_config.json          # Scene configuration
    ├── scene_description.yaml     # Eradiate scene description
    ├── data/                      # Processed geospatial data
    │   ├── dem.zarr              # Digital elevation model
    │   ├── landcover.zarr        # Land cover classification
    │   └── materials.json        # Material assignments
    ├── meshes/                   # 3D surface meshes
    │   ├── surface.ply           # Main surface mesh
    │   ├── buffer.ply            # Buffer zone mesh (if enabled)
    │   └── background.ply        # Background mesh (if enabled)
    └── textures/                 # Material textures
        ├── landcover_map.png     # Land cover visualization
        └── material_preview.png  # Material assignment preview
```
