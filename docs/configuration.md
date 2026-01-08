# Configuration

The S2GOS generator is configured through a `s2gos_settings.yaml` file. The generator automatically searches for this file by climbing up the directory tree from your script's location.

## Installation Modes

The configuration structure depends on your installation:

- **Standalone** (`s2gos-generator` only): Requires `common` and `generator` sections.
- **Monorepo** (with `s2gos-simulator`): Includes `common`, `generator`, and `simulator` sections.

## Configuration Example

```yaml
# s2gos_settings.yaml
## ========================================================================== ##
common:
    ## List of data paths to always add to the file resolver
    search_paths : [
        "/home/martonn/Projects/s2gos/s2gos/packages/s2gos-generator/resources/data",
        "/home/martonn/Projects/s2gos/s2gos/data",
    ]


## ========================================================================== ##
generator:
    # Datasets source paths.
    datasets:
        dem:
            type : "indexed-geotiff"
            root_directory : <local directory>
            index_path : <local path>

        landcover:
            type : "zarr"
            path : 
                value: "s3://path/to/worldcover.zarr"
                protocol : "s3"
                endpoint_url : <endpoint_url>
                key : <key>
                secret : <secret>

    config:
        material : "./some/local/path/to_json.json"
        material_2 : 
            value : "some/other/path"
            protocol : "https"



# Optional: Only needed in monorepo with s2gos-simulator
# simulator:
# See s2gos-simulator documentation for available options
```

## Configuration Sections

### `common` - Shared Settings

Settings used by both generator and simulator packages.

#### `search_paths`
*List of paths, optional (default: empty list)*

Prioritized list of directories for resolving relative file paths. The file resolver searches these paths in order and returns the first match. Useful for locating resource files (materials.json, ephemeris data) and index files across different environments.

Paths can be absolute or relative. Local paths are automatically resolved, and remote paths (s3://, etc.) are supported. Environment override available via `S2GOS_SEARCH_PATHS`.

### `generator.datasets` - Data Sources

Specifies locations of geospatial datasets and data files. The following keywords are currently accepted:
- `dem`: the location to the Copernicus DEM.
- `landcover`: the location the ESA WorldCover tiles.
- `material`: the location to the material configuration JSON file.

A dataset/file is a nested object that requires the following keyword: 
- `type`: specifies the type of dataset or file being used.

[specify the alias method]

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