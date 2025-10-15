import fnmatch
import logging
import shutil
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

from s2gos_utils.io.paths import exists
from upath import UPath


def import_xml_assets(
    xml_path: str,
    base_coordinate: List[float],
    object_id_prefix: str = "asset",
    elevation_offset: float = 0.0,
    scale: float = 1.0,
    fix_blender_coords: bool = True,
    material_mappings: Optional[Dict[str, str]] = None,
    pattern_type: str = "wildcard",
    validate_materials: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Import Mitsuba XML and convert to S2GOS assets with material library.

    Args:
        xml_path: UPath to Mitsuba XML file
        base_coordinate: [longitude, latitude] for all components
        object_id_prefix: Prefix for asset IDs
        elevation_offset: Height offset above terrain (meters)
        scale: Uniform scaling factor
        fix_blender_coords: Apply Blender→Mitsuba coordinate correction (90° X rotation)
        material_mappings: Dict mapping filename patterns to S2GOS material names
        pattern_type: "wildcard", "exact", or "contains" matching for material_mappings
        validate_materials: If True, validate material references and PLY file existence

    Returns:
        Tuple of (assets_list, material_library):
        - assets_list: List of asset dicts with string material references
        - material_library: Dict of {material_id: material_definition}
    """
    if not isinstance(base_coordinate, (list, tuple)) or len(base_coordinate) != 2:
        raise ValueError(
            f"base_coordinate must be a list/tuple of exactly 2 elements [longitude, latitude], got: {base_coordinate}"
        )

    try:
        float(base_coordinate[0])
        float(base_coordinate[1])
    except (ValueError, TypeError):
        raise ValueError(
            f"base_coordinate values must be numeric, got: {base_coordinate}"
        )

    xml_data = _parse_xml(xml_path)
    material_library = _convert_materials(xml_data["materials"], xml_path)

    assets = []

    for shape in xml_data["shapes"]:
        ply_filename = UPath(shape["file"]).stem

        material_ref = None
        if material_mappings:
            for pattern, mapped_material in material_mappings.items():
                if _match_filename(ply_filename, pattern, pattern_type):
                    material_ref = mapped_material
                    break

        if material_ref is None:
            original_material_id = shape["material"]
            if original_material_id in material_library:
                material_ref = original_material_id
            else:
                logging.warning(
                    f"Material '{original_material_id}' not found for '{ply_filename}'. Using 'concrete' fallback."
                )
                material_ref = "concrete"

        rotation_x = 90.0 if fix_blender_coords else 0.0

        asset_data = {
            "object_id": f"{object_id_prefix}_{ply_filename}",
            "ply_path": shape["file"],
            "coordinate": base_coordinate.copy(),
            "material": material_ref,
            "elevation_offset": elevation_offset,
            "scale": scale,
            "rotation_x": rotation_x,
            "rotation_y": 0.0,
            "rotation_z": 0.0,
        }

        assets.append(asset_data)

    if validate_materials:
        _validate_assets(assets, material_library)

    return assets, material_library


def merge_material_libraries(
    *libraries: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Merge multiple material libraries, warning about conflicts."""
    merged = {}
    for i, library in enumerate(libraries):
        for mat_id, mat_def in library.items():
            if mat_id in merged:
                logging.warning(
                    f"Material '{mat_id}' conflict. Using definition from library {i + 1}."
                )
            merged[mat_id] = mat_def
    return merged


def _parse_xml(xml_path: str) -> Dict[str, Any]:
    """Parse Mitsuba XML file to extract materials and shapes."""
    xml_file = UPath(xml_path)
    if not xml_file.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    tree = ET.parse(xml_path)
    xml_dir = xml_file.parent.resolve()

    materials = {}
    for bsdf in tree.findall(".//bsdf"):
        material_id = bsdf.get("id")
        if material_id:
            materials[material_id] = _parse_bsdf_element(bsdf)

    shapes = []
    for shape in tree.findall('.//shape[@type="ply"]'):
        filename_elem = shape.find('./string[@name="filename"]')
        if filename_elem is not None:
            filename = filename_elem.get("value")
            material_ref = shape.find('./ref[@name="bsdf"]')
            material_id = (
                material_ref.get("id", "default-bsdf")
                if material_ref is not None
                else "default-bsdf"
            )
            shapes.append({"file": str(xml_dir / filename), "material": material_id})

    return {"materials": materials, "shapes": shapes}


def _parse_bsdf_element(bsdf_element) -> Dict[str, Any]:
    """Parse BSDF element to extract type and properties."""
    mat_data = {"type": bsdf_element.get("type", "diffuse"), "properties": {}}

    if mat_data["type"] == "twosided":
        nested_bsdf = bsdf_element.find("./bsdf")
        if nested_bsdf is not None:
            nested_data = _parse_bsdf_element(nested_bsdf)
            mat_data["nested_type"] = nested_data["type"]

            overlapping_props = set(mat_data["properties"].keys()) & set(
                nested_data["properties"].keys()
            )
            if overlapping_props:
                logging.warning(
                    f"Twosided material property collision: {overlapping_props} - nested properties will override parent"
                )

            mat_data["properties"].update(nested_data["properties"])

    for child in bsdf_element:
        if child.tag in ["rgb", "spectrum", "float", "string", "integer", "boolean"]:
            name = child.get("name")
            if name:
                mat_data["properties"][name] = _parse_property(child)

    return mat_data


def _parse_property(element) -> Any:
    """Parse individual property element."""
    tag = element.tag
    value = element.get("value", "")

    if tag == "rgb":
        try:
            rgb_values = [float(x) for x in value.split()]
            if len(rgb_values) == 3:
                return rgb_values
            elif len(rgb_values) == 1:
                return [rgb_values[0]] * 3
            else:
                logging.warning(
                    f"RGB value '{value}' has {len(rgb_values)} components, expected 3. Using default."
                )
                return [0.5, 0.5, 0.5]
        except (ValueError, IndexError):
            return [0.5, 0.5, 0.5]
    elif tag == "float":
        try:
            return float(value)
        except ValueError:
            return 0.0
    elif tag == "integer":
        try:
            return int(value)
        except ValueError:
            return 0
    elif tag == "boolean":
        return value.lower() == "true"
    elif tag == "string":
        return value
    elif tag == "spectrum":
        filename = element.get("filename")
        if filename:
            return {"file": filename}
        try:
            return float(value)
        except ValueError:
            return 0.5

    return value


def _convert_materials(
    mitsuba_materials: Dict[str, Dict], xml_path: str
) -> Dict[str, Dict]:
    """Convert materials to S2GOS format using converter registry.

    Args:
        mitsuba_materials: Dictionary of material definitions from XML
        xml_path: UPath to source XML file (for resolving relative spectral paths)

    Returns:
        Dictionary of S2GOS material definitions
    """

    s2gos_materials = {}
    xml_dir = UPath(xml_path).parent

    for mat_id, mat_data in mitsuba_materials.items():
        try:
            sanitized_mat_id = mat_id.replace(".", "_").replace("-", "_")

            mat_type = mat_data.get("type", "diffuse")
            nested_type = mat_data.get("nested_type")

            if mat_type == "twosided" and nested_type:
                mat_type = nested_type

            converter = MATERIAL_CONVERTERS.get(mat_type, convert_diffuse)
            s2gos_materials[sanitized_mat_id] = converter(
                mat_data["properties"], xml_dir
            )

        except Exception as e:
            logging.warning(
                f"Failed to convert material '{mat_id}': {e}. Using diffuse fallback."
            )
            sanitized_mat_id = mat_id.replace(".", "_").replace("-", "_")
            s2gos_materials[sanitized_mat_id] = convert_diffuse({}, xml_dir)

    return s2gos_materials


def convert_diffuse(props: Dict[str, Any], xml_dir) -> Dict[str, Any]:
    """Convert diffuse material.

    Args:
        props: Material properties from XML
        xml_dir: Directory containing source XML file (for resolving relative paths)

    Returns:
        S2GOS material definition with absolute paths

    Raises:
        FileNotFoundError: If spectral file does not exist
    """

    reflectance = props.get("reflectance", [0.5, 0.5, 0.5])
    if isinstance(reflectance, dict) and "file" in reflectance:
        file_path = UPath(reflectance["file"])
        if not file_path.is_absolute():
            file_path = (xml_dir / file_path).resolve()

        if not exists(file_path):
            raise FileNotFoundError(
                f"Spectral data file not found: {file_path}\n"
                f"Original path: {reflectance['file']}\n"
                f"XML directory: {xml_dir}"
            )

        reflectance_spec = {
            "path": str(file_path),
            "variable": "reflectance",
        }
    elif isinstance(reflectance, (list, tuple)):
        reflectance_spec = {"type": "uniform", "value": list(reflectance)}
    elif isinstance(reflectance, (int, float)):
        reflectance_spec = {"type": "uniform", "value": float(reflectance)}
    else:
        reflectance_spec = {"type": "uniform", "value": [0.5, 0.5, 0.5]}
    return {"type": "diffuse", "reflectance": reflectance_spec}


def convert_conductor(props: Dict[str, Any]) -> Dict[str, Any]:
    """Convert conductor material."""
    result = {"type": "conductor"}

    material_preset = props.get("material")
    if material_preset:
        result["material"] = material_preset
    else:
        result["material"] = "Cu"  # Default copper

    spec_refl = props.get("specular_reflectance")
    if spec_refl is not None:
        if isinstance(spec_refl, (list, tuple)):
            result["specular_reflectance"] = {
                "type": "uniform",
                "value": list(spec_refl),
            }
        elif isinstance(spec_refl, (int, float)):
            result["specular_reflectance"] = {
                "type": "uniform",
                "value": float(spec_refl),
            }

    return result


def convert_roughconductor(props: Dict[str, Any], xml_dir) -> Dict[str, Any]:
    """Convert rough conductor material."""
    result = convert_conductor(props, xml_dir)
    result["type"] = "rough_conductor"
    result["distribution"] = props.get("distribution", "ggx")

    alpha_u = props.get("alpha_u")
    alpha_v = props.get("alpha_v")
    if alpha_u is not None or alpha_v is not None:
        if alpha_u is not None:
            result["alpha_u"] = float(alpha_u)
        if alpha_v is not None:
            result["alpha_v"] = float(alpha_v)
    else:
        alpha = props.get("alpha", props.get("roughness", 0.1))
        result["roughness"] = float(alpha)

    return result


def convert_dielectric(props: Dict[str, Any]) -> Dict[str, Any]:
    """Convert dielectric material."""
    result = {
        "type": "dielectric",
        "int_ior": float(props.get("int_ior", 1.5)),
        "ext_ior": float(props.get("ext_ior", 1.0)),
    }

    for prop_name in ["specular_reflectance", "specular_transmittance"]:
        prop_val = props.get(prop_name)
        if prop_val is not None:
            if isinstance(prop_val, (list, tuple)):
                result[prop_name] = {"type": "uniform", "value": list(prop_val)}
            elif isinstance(prop_val, (int, float)):
                result[prop_name] = {"type": "uniform", "value": float(prop_val)}

    return result


def convert_plastic(props: Dict[str, Any]) -> Dict[str, Any]:
    """Convert plastic material."""
    diffuse_refl = props.get("diffuse_reflectance", [0.5, 0.5, 0.5])

    if isinstance(diffuse_refl, (int, float)):
        diffuse_spec = {"type": "uniform", "value": float(diffuse_refl)}
    elif isinstance(diffuse_refl, (list, tuple)):
        diffuse_spec = {"type": "uniform", "value": list(diffuse_refl)}
    else:
        diffuse_spec = {"type": "uniform", "value": [0.5, 0.5, 0.5]}

    return {
        "type": "plastic",
        "diffuse_reflectance": diffuse_spec,
        "int_ior": float(props.get("int_ior", 1.49)),
        "ext_ior": float(props.get("ext_ior", 1.0)),
        "roughness": float(props.get("alpha", 0.01)),
        "nonlinear": bool(props.get("nonlinear", False)),
    }


def convert_bilambertian(props: Dict[str, Any], xml_dir) -> Dict[str, Any]:
    """Convert bi-lambertian (two-sided diffuse) material.

    Args:
        props: Material properties from XML
        xml_dir: Directory containing source XML file (for resolving relative paths)

    Returns:
        S2GOS material definition with absolute paths

    Raises:
        FileNotFoundError: If spectral file does not exist
    """

    reflectance = props.get("reflectance", [0.5, 0.5, 0.5])
    transmittance = props.get("transmittance", [0.0, 0.0, 0.0])

    if isinstance(reflectance, dict) and "file" in reflectance:
        file_path = UPath(reflectance["file"])
        if not file_path.is_absolute():
            file_path = (xml_dir / file_path).resolve()

        if not exists(file_path):
            raise FileNotFoundError(
                f"Reflectance spectral data file not found: {file_path}\n"
                f"Original path: {reflectance['file']}\n"
                f"XML directory: {xml_dir}"
            )

        reflectance_spec = {
            "path": str(file_path),
            "variable": "reflectance",
        }
    elif isinstance(reflectance, (list, tuple)):
        reflectance_spec = {"type": "uniform", "value": list(reflectance)}
    elif isinstance(reflectance, (int, float)):
        reflectance_spec = {"type": "uniform", "value": float(reflectance)}
    else:
        reflectance_spec = {"type": "uniform", "value": [0.5, 0.5, 0.5]}

    if isinstance(transmittance, dict) and "file" in transmittance:
        file_path = UPath(transmittance["file"])
        if not file_path.is_absolute():
            file_path = (xml_dir / file_path).resolve()

        if not exists(file_path):
            raise FileNotFoundError(
                f"Transmittance spectral data file not found: {file_path}\n"
                f"Original path: {transmittance['file']}\n"
                f"XML directory: {xml_dir}"
            )

        transmittance_spec = {
            "path": str(file_path),
            "variable": "transmittance",
        }
    elif isinstance(transmittance, (list, tuple)):
        transmittance_spec = {"type": "uniform", "value": list(transmittance)}
    elif isinstance(transmittance, (int, float)):
        transmittance_spec = {"type": "uniform", "value": float(transmittance)}
    else:
        transmittance_spec = {"type": "uniform", "value": [0.0, 0.0, 0.0]}

    return {
        "type": "bilambertian",
        "reflectance": reflectance_spec,
        "transmittance": transmittance_spec,
    }


def convert_rpv(props: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Rahman Pinty Verstraete reflection model."""
    return {
        "type": "rpv",
        "rho_0": float(props.get("rho_0", 0.1)),
        "k": float(props.get("k", 0.5)),
        "g": float(props.get("g", -0.1)),
        "rho_c": float(props.get("rho_c", 0.0)),
    }


def convert_rtls(props: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Ross-Thick Li-Sparse reflection model."""
    return {
        "type": "rtls",
        "f_iso": float(props.get("f_iso", 1.0)),
        "f_geo": float(props.get("f_geo", 0.0)),
        "f_vol": float(props.get("f_vol", 0.0)),
    }


def convert_hapke(props: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Hapke surface model."""
    return {
        "type": "hapke",
        "w": float(props.get("w", 0.5)),
        "b": float(props.get("b", 0.0)),
        "c": float(props.get("c", 0.0)),
        "theta": float(props.get("theta", 0.0)),
        "B_0": float(props.get("B_0", 1.0)),
        "h": float(props.get("h", 0.06)),
    }


def convert_oceanic_grasp(props: Dict[str, Any]) -> Dict[str, Any]:
    """Convert GRASP oceanic model."""
    return {
        "type": "oceanic_grasp",
        "wavelength": float(props.get("wavelength", 550.0)),
        "wind_speed": float(props.get("wind_speed", 5.0)),
        "water_ior": float(props.get("water_ior", 1.33)),
    }


def convert_oceanic_6s(props: Dict[str, Any]) -> Dict[str, Any]:
    """Convert legacy 6S oceanic model."""
    return {
        "type": "oceanic_6s",
        "wavelength": float(props.get("wavelength", 550.0)),
        "wind_speed": float(props.get("wind_speed", 5.0)),
        "chlorinity": float(props.get("chlorinity", 0.0)),
        "pigmentation": float(props.get("pigmentation", 0.0)),
    }


def convert_oceanic_mishchenko(props: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Mishchenko oceanic model."""
    return {
        "type": "oceanic_mishchenko",
        "wind_speed": float(props.get("wind_speed", 5.0)),
        "water_ior": float(props.get("water_ior", 1.33)),
    }


def convert_selectbsdf(props: Dict[str, Any]) -> Dict[str, Any]:
    """Convert selector BSDF (texture-based material selection)."""
    return {
        "type": "selectbsdf",
        "texture": props.get("texture", "uniform"),
        "materials": props.get("materials", {}),
    }


def convert_measured(props: Dict[str, Any]) -> Dict[str, Any]:
    """Convert measured quasi-diffuse material."""
    return {
        "type": "measured",
        "data": props.get("data", ""),
        "scale": float(props.get("scale", 1.0)),
    }


MATERIAL_CONVERTERS = {
    "diffuse": convert_diffuse,
    "conductor": convert_conductor,
    "roughconductor": convert_roughconductor,
    "dielectric": convert_dielectric,
    "plastic": convert_plastic,
    "bilambertian": convert_bilambertian,
    "rpv": convert_rpv,
    "rtls": convert_rtls,
    "hapke": convert_hapke,
    "oceanic_grasp": convert_oceanic_grasp,
    "oceanic_6s": convert_oceanic_6s,
    "oceanic_mishchenko": convert_oceanic_mishchenko,
    "selectbsdf": convert_selectbsdf,
    "measured": convert_measured,
}


def _ensure_list(value: Any) -> List[float]:
    """Ensure value is a list of floats."""
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    elif isinstance(value, (int, float)):
        return [float(value)] * 3
    else:
        return [0.5, 0.5, 0.5]


def _match_filename(
    filename: str, pattern: str, pattern_type: str = "wildcard"
) -> bool:
    """Check if filename matches pattern using specified matching strategy.

    Args:
        filename: PLY filename (stem, no extension)
        pattern: Pattern to match against
        pattern_type: "wildcard", "exact", or "contains"

    Returns:
        True if filename matches pattern
    """
    if pattern_type == "exact":
        return filename == pattern
    elif pattern_type == "contains":
        return pattern in filename
    elif pattern_type == "wildcard":
        return fnmatch.fnmatch(filename, pattern)
    else:
        raise ValueError(f"Unknown pattern_type: {pattern_type}")


def create_tree_shapegroup(
    tree_xml_path: str, output_dir: Optional["UPath"] = None
) -> Dict[str, Any]:
    """Create Mitsuba shapegroup from tree XML file.

    Args:
        tree_xml_path: UPath to tree XML file
        output_dir: Scene output directory where mesh files will be copied

    Returns:
        Dictionary containing shapegroup definition for Mitsuba scene
    """
    xml_data = _parse_xml(tree_xml_path)
    materials = _convert_materials(xml_data["materials"], tree_xml_path)

    shapegroup = {"type": "shapegroup", "id": "tree_group"}

    if output_dir:
        tree_meshes_dir = UPath(output_dir) / "meshes" / "tree"
        tree_meshes_dir.mkdir(parents=True, exist_ok=True)

    for i, shape in enumerate(xml_data["shapes"]):
        shape_name = f"tree_component_{i}"

        # Get material reference - use proper reference format
        # Sanitize material ID to match the sanitized IDs from _convert_materials
        material_id = shape["material"].replace(".", "_").replace("-", "_")

        source_file_path = UPath(shape["file"])

        if output_dir and source_file_path.exists():
            dest_filename = source_file_path.name
            dest_path = tree_meshes_dir / dest_filename

            if not dest_path.exists():
                shutil.copy2(source_file_path, dest_path)
                logging.info(f"Copied tree mesh: {dest_filename}")

            mesh_filename = f"meshes/tree/{dest_filename}"
        else:
            mesh_filename = str(source_file_path)
            if output_dir and not source_file_path.exists():
                logging.warning(f"Tree mesh file not found: {source_file_path}")

        shapegroup[shape_name] = {
            "type": "ply",
            "filename": mesh_filename,
            "face_normals": True,
            "bsdf": {"type": "ref", "id": f"_mat_{material_id}"},
        }

    return shapegroup, materials


def _validate_assets(
    assets: List[Dict[str, Any]], material_library: Dict[str, Dict[str, Any]]
) -> None:
    """Validate asset material references and PLY file existence."""
    errors = []

    for asset in assets:
        asset_id = asset["object_id"]

        # Note: Material references may be external (from scene library) so we don't validate them here
        ply_path = UPath(asset["ply_path"])
        if not ply_path.exists():
            errors.append(f"Asset '{asset_id}': PLY file not found: {ply_path}")

    if errors:
        raise ValueError(
            "Asset validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )
