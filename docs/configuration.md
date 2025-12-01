# Configuration

The S2GOS generator is configured through a `s2gos_settings.toml` file. The generator automatically searches for this file by climbing up the directory tree from your script's location.

## Installation Modes

The configuration structure depends on your installation:

- **Standalone** (`s2gos-generator` only): Requires `[common]` and `[generator]` sections
- **Monorepo** (with `s2gos-simulator`): Includes `[common]`, `[generator]`, and `[simulator]` sections

## Configuration Example

```toml
# s2gos_settings.toml
# -----------------------------------------
[common]
# List of directories for resolving relative file paths (searched in order)
search_paths = [
    "/home/user/s2gos/packages/s2gos-generator/resources/data",
    "/home/user/s2gos/data",
]

[generator.data]
# Required: Root directories for DEM and land cover data
dem_root_dir = "/path/to/DEM"
landcover_root_dir = "/path/to/Landcover/"

# Optional: Override defaults (relative paths resolved via search_paths)
# dem_index_path = "dem_index.feather"              
# landcover_index_path = "landcover_index.feather"  
# material_config_path = "materials.json"           

# Optional: Only needed in monorepo with s2gos-simulator
# [simulator]
# See s2gos-simulator documentation for available options
```

## Configuration Sections

### `[common]` - Shared Settings

Settings used by both generator and simulator packages.

#### `search_paths`
*List of paths, optional (default: empty list)*

Prioritized list of directories for resolving relative file paths. The file resolver searches these paths in order and returns the first match. Useful for locating resource files (materials.json, ephemeris data) and index files across different environments.

Paths can be absolute or relative. Local paths are automatically resolved, and remote paths (s3://, etc.) are supported. Environment override available via `S2GOS_SEARCH_PATHS`.

### `[generator.data]` - Data Sources

Specifies locations of required geospatial datasets.

##### `dem_root_dir`
*Path string, required*

Root directory containing Copernicus DEM tiles. Can be absolute or relative (resolved via search_paths).

##### `landcover_root_dir`
*Path string, required*

Root directory containing ESA WorldCover tiles. Can be absolute or relative (resolved via search_paths).

##### `dem_index_path`
*Path string, optional (default: "dem_index.feather")*

Path to DEM tile catalog in Feather format. Relative paths resolved via search_paths.

##### `landcover_index_path`
*Path string, optional (default: "landcover_index.feather")*

Path to land cover tile catalog in Feather format. Relative paths resolved via search_paths.

##### `material_config_path`
*Path string, optional (default: "materials.json")*

Path to materials configuration defining optical properties for land cover classes. Relative paths resolved via search_paths.

### `[simulator]` - Simulator Settings (Optional)

When installed in a monorepo with s2gos-simulator, simulator-specific settings can be included here. See s2gos-simulator documentation for available options.