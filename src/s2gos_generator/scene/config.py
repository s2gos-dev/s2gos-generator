import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from s2gos_utils.scene import SceneDescription
from s2gos_utils.scene.materials import Material, get_landcover_mapping, load_materials
from upath import UPath


def _convert_atmosphere_config_to_dict(atmosphere_config) -> dict:
    """Convert to scene description dictionary format.

    Args:
        atmosphere_config: Atmosphere config object from scene generation configuration

    Returns:
        Dictionary format suitable for scene description
    """

    base_dict = {
        "boa": atmosphere_config.boa,
        "toa": atmosphere_config.toa,
        "type": atmosphere_config.details.type,
    }

    if atmosphere_config.details.type == "molecular":
        mol_config = atmosphere_config.details
        mol_atm_dict = {
            "altitude_min": mol_config.thermoprops.altitude_min,
            "altitude_max": mol_config.thermoprops.altitude_max,
            "altitude_step": mol_config.thermoprops.altitude_step,
            "constituent_scaling": mol_config.thermoprops.constituent_scaling,
            "absorption_database": mol_config.absorption_database.value
            if mol_config.absorption_database
            else None,
            "has_absorption": mol_config.has_absorption,
            "has_scattering": mol_config.has_scattering,
        }

        if mol_config.thermoprops.thermoprops_file:
            mol_atm_dict["thermoprops_file"] = str(
                mol_config.thermoprops.thermoprops_file
            )
        else:
            mol_atm_dict["thermoprops_identifier"] = mol_config.thermoprops.identifier

        base_dict["molecular_atmosphere"] = mol_atm_dict

    elif atmosphere_config.details.type == "homogeneous":
        homogeneous_config = atmosphere_config.details
        base_dict.update(
            {
                "aerosol_ot": homogeneous_config.optical_thickness,
                "aerosol_scale": homogeneous_config.scale_height,
                "aerosol_ds": homogeneous_config.aerosol_dataset.value,
                "reference_wavelength": homogeneous_config.reference_wavelength,
                "has_absorption": homogeneous_config.has_absorption,
            }
        )

    elif atmosphere_config.details.type == "heterogeneous":
        heterogeneous_config = atmosphere_config.details
        base_dict.update(
            {
                "has_molecular_atmosphere": heterogeneous_config.molecular is not None,
                "has_particle_layers": heterogeneous_config.particle_layers is not None
                and len(heterogeneous_config.particle_layers) > 0,
            }
        )

        if heterogeneous_config.molecular:
            mol_config = heterogeneous_config.molecular
            mol_atm_dict = {
                "altitude_min": mol_config.thermoprops.altitude_min,
                "altitude_max": mol_config.thermoprops.altitude_max,
                "altitude_step": mol_config.thermoprops.altitude_step,
                "constituent_scaling": mol_config.thermoprops.constituent_scaling,
                "absorption_database": mol_config.absorption_database.value
                if mol_config.absorption_database
                else None,
                "has_absorption": mol_config.has_absorption,
                "has_scattering": mol_config.has_scattering,
            }

            if mol_config.thermoprops.thermoprops_file:
                mol_atm_dict["thermoprops_file"] = str(
                    mol_config.thermoprops.thermoprops_file
                )
            else:
                mol_atm_dict["thermoprops_identifier"] = (
                    mol_config.thermoprops.identifier
                )

            base_dict["molecular_atmosphere"] = mol_atm_dict

        if heterogeneous_config.particle_layers:
            base_dict["particle_layers"] = []
            for layer in heterogeneous_config.particle_layers:
                layer_dict = {
                    "aerosol_dataset": (
                        layer.aerosol_dataset.value
                        if hasattr(layer.aerosol_dataset, "value")
                        else layer.aerosol_dataset
                    ),
                    "optical_thickness": layer.optical_thickness,
                    "altitude_bottom": layer.altitude_bottom,
                    "altitude_top": layer.altitude_top,
                    "distribution_type": layer.distribution.type,
                    "reference_wavelength": layer.reference_wavelength,
                    "has_absorption": layer.has_absorption,
                }

                if layer.distribution.type == "exponential":
                    if layer.distribution.rate is not None:
                        layer_dict["rate"] = layer.distribution.rate
                    elif layer.distribution.scale is not None:
                        layer_dict["scale"] = layer.distribution.scale
                elif layer.distribution.type == "gaussian":
                    layer_dict["center_altitude"] = layer.distribution.center_altitude
                    layer_dict["width"] = layer.distribution.width

                base_dict["particle_layers"].append(layer_dict)

    else:
        raise ValueError(f"Unknown atmosphere type: {atmosphere_config.details.type}")

    return base_dict


def create_s2gos_scene(
    scene_name: str,
    mesh_path: str,
    texture_path: str,
    center_lat: float,
    center_lon: float,
    aoi_size_km: float,
    resolution_m: float = 30.0,
    buffer_mesh_path: Optional[str] = None,
    buffer_texture_path: Optional[str] = None,
    buffer_size_km: Optional[float] = None,
    output_dir: Optional[UPath] = None,
    background_elevation: Optional[float] = None,
    buffer_dem_file: Optional[str] = None,
    material_config_path: Optional[UPath] = None,
    material_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    landcover_mapping_overrides: Optional[Dict[str, str]] = None,
    background_selection_texture: Optional[str] = None,
    background_size_km: Optional[float] = None,
    hamster_data_paths: Optional[Dict[str, UPath]] = None,
    additional_material_libraries: Optional[List[Dict[str, Dict[str, Any]]]] = None,
    **kwargs,
) -> SceneDescription:
    """Create standard S2GOS scene configuration.

    Args:
        scene_name: Name of the scene
        mesh_path: UPath to the terrain mesh file
        texture_path: UPath to the selection texture file
        center_lat: Center latitude of the scene
        center_lon: Center longitude of the scene
        aoi_size_km: Size of the area of interest in kilometers
        resolution_m: Target resolution in meters
        buffer_mesh_path: Optional path to buffer mesh file
        buffer_texture_path: Optional path to buffer texture file
        buffer_size_km: Optional buffer size in kilometers
        output_dir: Output directory for the scene
        background_elevation: Background elevation
        buffer_dem_file: Optional buffer DEM file path
        material_config_path: Optional path to custom material configuration JSON
        material_overrides: Optional dictionary of material property overrides
        landcover_mapping_overrides: Optional dictionary of landcover-to-material mapping overrides
        background_selection_texture: Optional path to background selection texture file
        background_size_km: Optional background size in kilometers
        hamster_data_paths: Optional dictionary of paths to processed HAMSTER albedo zarr files for each surface area
        additional_material_libraries: Optional list of material libraries to merge (e.g., from XML imports)
        **kwargs: Additional configuration parameters

    Returns:
        SceneDescription instance with loaded materials and configuration
    """

    material_mapping = get_landcover_mapping(material_config_path)

    if landcover_mapping_overrides:
        material_mapping.update(landcover_mapping_overrides)

    material_indices = {}
    landcover_ids = {
        "tree_cover": 0,
        "shrubland": 1,
        "grassland": 2,
        "cropland": 3,
        "built_up": 4,
        "bare_sparse_vegetation": 5,
        "snow_and_ice": 6,
        "permanent_water_bodies": 7,
        "herbaceous_wetland": 8,
        "mangroves": 9,
        "moss_and_lichen": 10,
    }

    for landcover_class_name, texture_index in landcover_ids.items():
        if landcover_class_name in material_mapping:
            material_name = material_mapping[landcover_class_name]
            material_indices[texture_index] = material_name

    region_material_indices = kwargs.get("region_material_indices", {})
    if region_material_indices:
        for material_name, texture_index in region_material_indices.items():
            if material_name not in material_indices.values():
                material_indices[texture_index] = material_name
                logging.info(
                    f"Added region material '{material_name}' with texture index {texture_index}"
                )

    target = {
        "mesh": mesh_path,
        "selection_texture": texture_path,
        "size_km": aoi_size_km,
    }

    if hamster_data_paths and "target" in hamster_data_paths:
        if output_dir:
            target["hamster_data"] = str(
                hamster_data_paths["target"].relative_to(output_dir)
            )
        else:
            target["hamster_data"] = str(hamster_data_paths["target"])

    atmosphere_config = kwargs.get("atmosphere_config")
    atmosphere = _convert_atmosphere_config_to_dict(atmosphere_config)

    buffer = None
    background = None
    if buffer_mesh_path and buffer_texture_path and buffer_size_km:
        if output_dir is None:
            output_dir = UPath(".")

        buffer_resolution = int(buffer_size_km * 10)
        target_resolution = int(aoi_size_km * 10)

        from ..assets.masks import generate_buffer_mask

        mask_path = (
            output_dir
            / "textures"
            / f"mask_{scene_name}_{int(buffer_size_km)}km_buffer_{int(aoi_size_km)}km_target.bmp"
        )
        generate_buffer_mask(
            mask_size=buffer_resolution,
            target_size=target_resolution,
            output_path=mask_path,
        )

        buffer = {
            "mesh": buffer_mesh_path,
            "selection_texture": buffer_texture_path,
            "size_km": buffer_size_km,
            "target_size_km": aoi_size_km,
            "mask_texture": str(mask_path.relative_to(output_dir)),
        }

        if hamster_data_paths and "buffer" in hamster_data_paths:
            if output_dir:
                buffer["hamster_data"] = str(
                    hamster_data_paths["buffer"].relative_to(output_dir)
                )
            else:
                buffer["hamster_data"] = str(hamster_data_paths["buffer"])

    if background_selection_texture and background_size_km:
        bg_elevation = 0.0
        if background_elevation is not None:
            bg_elevation = background_elevation
        elif buffer_dem_file is not None:
            try:
                import xarray as xr

                if output_dir:
                    dem_path = output_dir / buffer_dem_file
                else:
                    dem_path = UPath(buffer_dem_file)

                from s2gos_utils.io.paths import exists

                if exists(dem_path):
                    dem_data = xr.open_zarr(dem_path)
                    if "elevation" in dem_data.data_vars:
                        bg_elevation = float(dem_data["elevation"].mean().values)
                    else:
                        var_name = list(dem_data.data_vars.keys())[0]
                        bg_elevation = float(dem_data[var_name].mean().values)
            except Exception as e:
                logging.warning(
                    f"Could not calculate average elevation from {buffer_dem_file}: {e}"
                )
                bg_elevation = 0.0

        background = {
            "selection_texture": background_selection_texture,
            "elevation": bg_elevation,
            "size_km": background_size_km,
        }

        if hamster_data_paths and "background" in hamster_data_paths:
            if output_dir:
                background["hamster_data"] = str(
                    hamster_data_paths["background"].relative_to(output_dir)
                )
            else:
                background["hamster_data"] = str(hamster_data_paths["background"])

    materials = load_materials(material_config_path)

    inline_materials = kwargs.get("inline_materials", {})
    if inline_materials:
        logging.info(
            f"Adding {len(inline_materials)} inline material definitions from UserAssets"
        )
        for mat_id, mat_def in inline_materials.items():
            if mat_id in materials:
                logging.warning(
                    f"Inline material '{mat_id}' overwrites existing material from materials.json"
                )
            materials[mat_id] = Material.from_dict(mat_def, id=mat_id)

    if additional_material_libraries:
        for library in additional_material_libraries:
            for mat_id, mat_def in library.items():
                if mat_id in materials:
                    logging.warning(
                        f"Material '{mat_id}' from additional library overwrites existing material."
                    )
                materials[mat_id] = Material.from_dict(mat_def, id=mat_id)

    if material_overrides:
        for material_id, overrides in material_overrides.items():
            if material_id in materials:
                current_dict = materials[material_id].to_dict()
                current_dict.update(overrides)
                materials[material_id] = Material.from_dict(
                    current_dict, id=material_id
                )
            else:
                materials[material_id] = Material.from_dict(overrides, id=material_id)

    metadata = {
        "generation_date": datetime.now().isoformat(),
        "random_seed": kwargs.get("random_seed"),
        "dem_name": kwargs.get("dem_name"),
        "landcover_name": kwargs.get("landcover_name"),
        "landcover_ids": landcover_ids,
        "materials_config_path": str(material_config_path),
    }

    processed_objects = kwargs.get("processed_objects", [])
    objects = processed_objects if processed_objects else []

    vegetation_collection_references = kwargs.get(
        "vegetation_collection_references", None
    )
    if vegetation_collection_references:
        from ..assets.xml_importer import create_tree_shapegroup
        from ..resources.vegetation import load_vegetation_collection_binary

        logging.info(
            f"Loading {len(vegetation_collection_references)} vegetation collections from binary format"
        )
        for vegetation_collection in vegetation_collection_references:
            try:
                data_file = vegetation_collection["data_file"]
                if output_dir:
                    binary_path = output_dir / data_file
                else:
                    binary_path = data_file

                vegetation_data_array = load_vegetation_collection_binary(binary_path)

                if len(vegetation_data_array) > 0:
                    species_name = vegetation_collection["name"]
                    vegetation_xml_file = vegetation_collection.get(
                        "model_file", "tree.xml"
                    )
                    vegetation_xml_path = vegetation_xml_file.upath
                    asset_basename = vegetation_xml_path.stem

                    vegetation_shapegroup, vegetation_materials = (
                        create_tree_shapegroup(vegetation_xml_path, output_dir)
                    )
                    for mat_id, mat_def in vegetation_materials.items():
                        namespaced_mat_id = f"{species_name}_{asset_basename}_{mat_id}"
                        if namespaced_mat_id not in materials:
                            materials[namespaced_mat_id] = Material.from_dict(
                                mat_def, id=namespaced_mat_id
                            )

                    for component_key, component in vegetation_shapegroup.items():
                        if isinstance(component, dict) and "bsdf" in component:
                            original_mat_id = component["bsdf"]["id"]
                            if original_mat_id.startswith("_mat_"):
                                raw_mat_id = original_mat_id[
                                    5:
                                ]  # Remove '_mat_' prefix
                                component["bsdf"]["id"] = (
                                    f"_mat_{species_name}_{asset_basename}_{raw_mat_id}"
                                )

                    species_shapegroup_id = (
                        f"vegetation_shapegroup_{species_name}_{asset_basename}"
                    )
                    species_group_id = (
                        f"vegetation_group_{species_name}_{asset_basename}"
                    )

                    vegetation_shapegroup["id"] = species_group_id

                    objects.append(
                        {
                            "object_id": species_shapegroup_id,
                            "type": "shapegroup",
                            **vegetation_shapegroup,
                        }
                    )

                    vegetation_collection_obj = {
                        "object_id": f"vegetation_collection_{species_name}_{asset_basename}",
                        "type": "vegetation_collection",
                        "shapegroup_ref": species_group_id,
                        "data_file": data_file
                        if not output_dir
                        else str(binary_path.relative_to(output_dir)),
                        "count": len(vegetation_data_array),
                        "collection_name": f"{species_name}_{asset_basename}",
                    }
                    objects.append(vegetation_collection_obj)

                    logging.info(
                        f"Added vegetation collection '{species_name}_{asset_basename}' with {len(vegetation_data_array)} instances (model: {vegetation_xml_file})"
                    )

            except Exception as e:
                logging.warning(
                    f"Failed to load binary vegetation collection '{vegetation_collection.get('name', 'unknown')}': {e}"
                )

    scene_description = SceneDescription(
        name=scene_name,
        location={
            "center_lat": center_lat,
            "center_lon": center_lon,
            "aoi_size_km": aoi_size_km,
        },
        resolution_m=resolution_m,
        materials=materials,
        atmosphere=atmosphere,
        target=target,
        buffer=buffer,
        background=background,
        objects=objects,
        material_indices=material_indices,
        metadata=metadata,
    )

    return scene_description
