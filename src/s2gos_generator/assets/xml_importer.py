import fnmatch
import logging
import re
import shutil
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

from s2gos_utils.io.paths import exists
from s2gos_utils.scene.materials.spectrum import (
    FileSpectrum,
    InterpolatedSpectrum,
    SpectrumSpec,
    UniformSpectrum,
)
from upath import UPath


def _sanitize_material_id(material_id: str) -> str:
    """Sanitize material ID to ensure valid S2GOS identifier.

    Replaces characters that may cause issues in material references:
    - Dots (.) → underscores (_)
    - Hyphens (-) → underscores (_)

    Args:
        material_id: Original material ID from XML

    Returns:
        Sanitized material ID safe for use in S2GOS
    """
    return material_id.replace(".", "_").replace("-", "_")


def import_xml_assets(
    xml_path: str,
    base_coordinate: Tuple[float, float],
    coord_type: str,
    object_id_prefix: str = "asset",
    elevation_offset: float = 0.0,
    scale: float = 1.0,
    fix_blender_coords: bool = True,
    rotation_x: float = 0.0,
    rotation_y: float = 0.0,
    rotation_z: float = 0.0,
    material_mappings: Optional[List[Dict[str, Any]]] = None,
    validate_materials: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Import Mitsuba XML and convert to S2GOS assets with material library.

    Args:
        xml_path: Path to Mitsuba XML file
        base_coordinate: Base coordinates (lon, lat) or (x, y)
        coord_type: "geographic" or "scene"
        object_id_prefix: Prefix for asset IDs
        elevation_offset: Height offset above terrain (meters)
        scale: Uniform scaling factor
        fix_blender_coords: Apply Blender→Mitsuba coordinate correction (90° X rotation)
        rotation_x: Global rotation around X-axis in degrees (applied after fix_blender_coords)
        rotation_y: Global rotation around Y-axis in degrees (applied after fix_blender_coords)
        rotation_z: Global rotation around Z-axis in degrees (applied after fix_blender_coords)
        material_mappings: List of MaterialMapping dicts (with pattern, material, mode fields)
        validate_materials: If True, validate material references and PLY file existence

    Returns:
        Tuple of (assets_list, material_library):
        - assets_list: List of asset dicts with string material references
        - material_library: Dict of {material_id: material_definition}
    """
    if len(base_coordinate) != 2:
        raise ValueError(
            f"base_coordinate must be a list/tuple of exactly 2 elements, got: {base_coordinate}"
        )

    logging.info(f"Importing assets from XML: {xml_path}")

    xml_data = _parse_xml(xml_path)
    material_library = _convert_materials(xml_data["materials"], xml_path)

    logging.info(f"Successfully converted {len(material_library)} materials from XML")

    assets = []

    for shape in xml_data["shapes"]:
        ply_filename = UPath(shape["file"]).stem

        material_ref = None
        if material_mappings:
            for mapping in material_mappings:
                pattern = mapping["pattern"]
                material = mapping["material"]
                mode = mapping.get("mode", "glob")  # Default to glob

                if _match_filename(ply_filename, pattern, mode):
                    material_ref = material
                    break

        if material_ref is None:
            original_material_id = shape["material"]
            sanitized_material_id = _sanitize_material_id(original_material_id)

            if sanitized_material_id in material_library:
                material_ref = sanitized_material_id
            else:
                available_materials = ", ".join(sorted(material_library.keys()))
                logging.warning(
                    f"Material '{original_material_id}' (sanitized: '{sanitized_material_id}') "
                    f"not found for '{ply_filename}'. "
                    f"Available materials: [{available_materials}]. "
                    f"Using 'concrete' fallback."
                )
                material_ref = "concrete"

        # Apply rotations with proper axis mapping
        # When fix_blender_coords=True, a 90° X-rotation is applied which swaps Y/Z axes
        # We need to remap user's rotations so rotation_z always means "rotate around up"
        if fix_blender_coords:
            # After 90° X rotation: Y-axis becomes up (world Z), Z-axis becomes -Y (world)
            # Swap user's Y/Z rotations to maintain intuitive behavior
            final_rotation_x = 90.0 + rotation_x
            final_rotation_y = (
                rotation_z  # User's Z rotation → Y-axis (now up in rotated frame)
            )
            final_rotation_z = (
                -rotation_y
            )  # User's Y rotation → -Z-axis (negated due to flip)
        else:
            # No coordinate fix: rotations are straightforward
            final_rotation_x = rotation_x
            final_rotation_y = rotation_y
            final_rotation_z = rotation_z

        if abs(final_rotation_x) < 1e-10:
            final_rotation_x = 0.0
        if abs(final_rotation_y) < 1e-10:
            final_rotation_y = 0.0
        if abs(final_rotation_z) < 1e-10:
            final_rotation_z = 0.0

        asset_data = {
            "object_id": f"{object_id_prefix}_{ply_filename}",
            "ply_path": shape["file"],
            "material": material_ref,
            "elevation_offset": elevation_offset,
            "scale": scale,
            "rotation_x": final_rotation_x,
            "rotation_y": final_rotation_y,
            "rotation_z": final_rotation_z,
            "blender_fix": fix_blender_coords,
        }

        if "face_normals" in shape:
            asset_data["face_normals"] = shape["face_normals"]

        assets.append(asset_data)

    if validate_materials:
        _validate_assets(assets, material_library)

    logging.info(f"Successfully imported {len(assets)} assets from {xml_path}")

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
    """Parse Mitsuba XML file to extract materials and shapes.

    Args:
        xml_path: Path to Mitsuba XML file

    Returns:
        Dictionary with "materials" and "shapes" keys

    Raises:
        FileNotFoundError: If XML file doesn't exist
        ET.ParseError: If XML is malformed
    """
    xml_file = UPath(xml_path)
    if not xml_file.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as e:
        raise ET.ParseError(
            f"Failed to parse XML file '{xml_path}': {e}. "
            "Please check that the file is valid Mitsuba XML."
        ) from e

    xml_dir = xml_file.parent.resolve()

    # Parse materials
    materials = {}
    for bsdf in tree.findall(".//bsdf"):
        material_id = bsdf.get("id")
        if material_id:
            materials[material_id] = _parse_bsdf_element(bsdf)
        else:
            logging.warning(
                f"Found <bsdf> element without 'id' attribute in {xml_path}. Skipping."
            )

    # Parse shapes
    shapes = []
    for shape in tree.findall('.//shape[@type="ply"]'):
        filename_elem = shape.find('./string[@name="filename"]')
        if filename_elem is not None:
            filename = filename_elem.get("value")
            if not filename:
                logging.warning(f"Shape in {xml_path} has empty filename. Skipping.")
                continue

            material_ref = shape.find('./ref[@name="bsdf"]')
            material_id = (
                material_ref.get("id", "default-bsdf")
                if material_ref is not None
                else "default-bsdf"
            )

            shape_data = {"file": str(xml_dir / filename), "material": material_id}
            face_normals_elem = shape.find('./boolean[@name="face_normals"]')
            if face_normals_elem is not None:
                face_normals_value = face_normals_elem.get("value", "false").lower()
                shape_data["face_normals"] = face_normals_value == "true"

            shapes.append(shape_data)
        else:
            logging.warning(
                f"Shape in {xml_path} missing <string name='filename'> element. Skipping."
            )

    logging.info(
        f"Parsed {xml_path}: found {len(materials)} materials and {len(shapes)} shapes"
    )

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
    """Parse individual property element following Mitsuba 3 specification.

    For spectrum elements, supports:
    - Inline wavelength:value pairs: "400:0.56, 500:0.18, 600:0.58"
    - Uniform spectrum: "0.5"
    - File-based: filename="spectrum.spd"

    Returns:
        Parsed property value in S2GOS-compatible format
    """
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
            logging.warning(
                f"Failed to parse RGB value '{value}'. Using default [0.5, 0.5, 0.5]."
            )
            return [0.5, 0.5, 0.5]

    elif tag == "float":
        try:
            return float(value)
        except ValueError:
            logging.warning(
                f"Failed to parse float value '{value}'. Using default 0.0."
            )
            return 0.0

    elif tag == "integer":
        try:
            return int(value)
        except ValueError:
            logging.warning(
                f"Failed to parse integer value '{value}'. Using default 0."
            )
            return 0

    elif tag == "boolean":
        return value.lower() == "true"

    elif tag == "string":
        return value

    elif tag == "spectrum":
        return _parse_spectrum_element(element)

    return value


def _parse_spectrum_element(element) -> Any:
    """Parse spectrum element following Mitsuba 3 specification.

    Args:
        element: XML element for spectrum

    Returns:
        dict: {"file": path} for file-based spectra
        dict: {"type": "interpolated", "wavelengths": [...], "values": [...]} for inline spectra
        float: scalar value for uniform spectra
    """
    spectrum_type = element.get("type")
    filename = element.get("filename")
    value = element.get("value", "")

    # Handle explicit type attribute from XML
    if spectrum_type == "uniform":
        # Uniform spectrum: single scalar value
        if not value:
            logging.warning("Uniform spectrum element has no value. Using default 0.5.")
            return 0.5

        try:
            return float(value)
        except ValueError:
            logging.error(
                f"Failed to parse uniform spectrum value '{value}' as float. "
                "Using default 0.5."
            )
            return 0.5

    elif spectrum_type == "interpolated":
        # Interpolated spectrum: can be file-based OR inline wavelength:value pairs
        if filename:
            # File-based interpolated spectrum
            return {"file": filename}

        if not value:
            logging.error(
                "Interpolated spectrum element has no value or filename. Using default 0.5."
            )
            return 0.5

        # Inline wavelength:value pairs format: "400:0.56, 500:0.18, 600:0.58"
        if ":" in value:
            try:
                pairs = [p.strip().split(":") for p in value.split(",")]
                wavelengths = [float(p[0].strip()) for p in pairs]
                values_list = [float(p[1].strip()) for p in pairs]

                # Validate wavelengths are in ascending order
                if wavelengths != sorted(wavelengths):
                    logging.warning(
                        f"Spectrum wavelengths are not in ascending order: {wavelengths}. "
                        "This may cause interpolation issues."
                    )

                # Validate wavelength range (typical range: 200-4000 nm)
                for wl in wavelengths:
                    if not (200 <= wl <= 4000):
                        logging.warning(
                            f"Wavelength {wl} nm is outside typical range [200, 4000] nm"
                        )

                # Validate values are non-negative (reflectance should be in [0, 1])
                for val in values_list:
                    if val < 0 or val > 1:
                        logging.warning(
                            f"Spectrum value {val} is outside typical reflectance range [0, 1]"
                        )

                logging.debug(
                    f"Parsed inline spectrum: {len(wavelengths)} wavelength points "
                    f"from {wavelengths[0]:.1f} to {wavelengths[-1]:.1f} nm"
                )

                return {
                    "type": "interpolated",
                    "wavelengths": wavelengths,
                    "values": values_list,
                }

            except (ValueError, IndexError) as e:
                logging.error(
                    f"Failed to parse inline spectrum value '{value}': {e}. "
                    "Expected format: 'wavelength:value, wavelength:value, ...'. "
                    "Using default 0.5."
                )
                return 0.5
        else:
            logging.error(
                f"Interpolated spectrum has value '{value}' but no ':' delimiter. "
                "Expected format: 'wavelength:value, wavelength:value, ...'. "
                "Using default 0.5."
            )
            return 0.5

    else:
        if filename:
            return {"file": filename}

        if not value:
            logging.warning(
                "Spectrum element has no value or filename. Using default 0.5."
            )
            return 0.5

        try:
            float_val = float(value)
            return float_val
        except ValueError:
            pass

        if ":" in value:
            try:
                pairs = [p.strip().split(":") for p in value.split(",")]
                wavelengths = [float(p[0].strip()) for p in pairs]
                values_list = [float(p[1].strip()) for p in pairs]

                # Validate wavelengths are in ascending order
                if wavelengths != sorted(wavelengths):
                    logging.warning(
                        f"Spectrum wavelengths are not in ascending order: {wavelengths}. "
                        "This may cause interpolation issues."
                    )

                # Validate wavelength range (typical range: 200-4000 nm)
                for wl in wavelengths:
                    if not (200 <= wl <= 4000):
                        logging.warning(
                            f"Wavelength {wl} nm is outside typical range [200, 4000] nm"
                        )

                # Validate values are non-negative (reflectance should be in [0, 1])
                for val in values_list:
                    if val < 0 or val > 1:
                        logging.warning(
                            f"Spectrum value {val} is outside typical reflectance range [0, 1]"
                        )

                logging.debug(
                    f"Parsed inline spectrum: {len(wavelengths)} wavelength points "
                    f"from {wavelengths[0]:.1f} to {wavelengths[-1]:.1f} nm"
                )

                return {
                    "type": "interpolated",
                    "wavelengths": wavelengths,
                    "values": values_list,
                }

            except (ValueError, IndexError) as e:
                logging.error(
                    f"Failed to parse inline spectrum value '{value}': {e}. "
                    "Expected format: 'wavelength:value, wavelength:value, ...'. "
                    "Using default 0.5."
                )
                return 0.5

        logging.warning(
            f"Could not parse spectrum value '{value}'. "
            "Expected numeric value or 'wavelength:value, ...' format. Using default 0.5."
        )
        return 0.5


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
        sanitized_mat_id = _sanitize_material_id(mat_id)

        try:
            mat_type = mat_data.get("type", "diffuse")
            nested_type = mat_data.get("nested_type")

            if mat_type == "twosided" and nested_type:
                mat_type = nested_type

            logging.debug(
                f"Converting material '{mat_id}' → '{sanitized_mat_id}' of type '{mat_type}'"
            )
            converter = MATERIAL_CONVERTERS.get(mat_type, convert_diffuse)
            s2gos_materials[sanitized_mat_id] = converter(
                mat_data["properties"], xml_dir
            )

        except Exception as e:
            logging.warning(
                f"Failed to convert material '{mat_id}' → '{sanitized_mat_id}': {e}. "
                "Using diffuse fallback."
            )
            s2gos_materials[sanitized_mat_id] = convert_diffuse({}, xml_dir)

    return s2gos_materials


def convert_diffuse(props: Dict[str, Any], xml_dir=None) -> Dict[str, Any]:
    """Convert diffuse material following Mitsuba 3 specification.

    Uses Pydantic spectrum models for validation and clear error messages.

    Args:
        props: Material properties from XML (already parsed by _parse_property)
        xml_dir: Directory containing source XML file (for resolving relative paths)

    Returns:
        S2GOS material definition with validated spectrum specification

    Raises:
        FileNotFoundError: If spectral file does not exist
        ValueError: If spectrum specification is invalid
    """
    reflectance = props.get("reflectance", [0.5, 0.5, 0.5])

    spectrum_spec: SpectrumSpec

    try:
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

            logging.debug(f"Using file-based spectrum: {file_path}")
            spectrum_spec = FileSpectrum(path=str(file_path), variable="reflectance")

        elif (
            isinstance(reflectance, dict) and reflectance.get("type") == "interpolated"
        ):
            logging.debug(
                f"Using interpolated spectrum: {len(reflectance['wavelengths'])} wavelength points"
            )
            spectrum_spec = InterpolatedSpectrum(
                wavelengths=reflectance["wavelengths"], values=reflectance["values"]
            )

        elif isinstance(reflectance, (list, tuple, int, float)):
            value = (
                list(reflectance)
                if isinstance(reflectance, (list, tuple))
                else float(reflectance)
            )
            spectrum_spec = UniformSpectrum(value=value)

        else:
            logging.warning(
                f"Unexpected reflectance type {type(reflectance)} with value {reflectance}. "
                "Using default uniform [0.5, 0.5, 0.5]."
            )
            spectrum_spec = UniformSpectrum(value=[0.5, 0.5, 0.5])

    except Exception as e:
        logging.error(f"Spectrum validation failed: {e}")
        logging.warning("Using fallback uniform spectrum [0.5, 0.5, 0.5]")
        spectrum_spec = UniformSpectrum(value=[0.5, 0.5, 0.5])

    return {"type": "diffuse", "reflectance": spectrum_spec.model_dump()}


def convert_conductor(props: Dict[str, Any], xml_dir=None) -> Dict[str, Any]:
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


def convert_roughconductor(props: Dict[str, Any], xml_dir=None) -> Dict[str, Any]:
    """Convert rough conductor material."""
    result = convert_conductor(props)
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


def convert_dielectric(props: Dict[str, Any], xml_dir=None) -> Dict[str, Any]:
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


def convert_plastic(props: Dict[str, Any], xml_dir=None) -> Dict[str, Any]:
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


def convert_bilambertian(props: Dict[str, Any], xml_dir=None) -> Dict[str, Any]:
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


def convert_rpv(props: Dict[str, Any], xml_dir=None) -> Dict[str, Any]:
    """Convert Rahman Pinty Verstraete reflection model."""
    return {
        "type": "rpv",
        "rho_0": float(props.get("rho_0", 0.1)),
        "k": float(props.get("k", 0.5)),
        "g": float(props.get("g", -0.1)),
        "rho_c": float(props.get("rho_c", 0.0)),
    }


def convert_rtls(props: Dict[str, Any], xml_dir=None) -> Dict[str, Any]:
    """Convert Ross-Thick Li-Sparse reflection model."""
    return {
        "type": "rtls",
        "f_iso": float(props.get("f_iso", 1.0)),
        "f_geo": float(props.get("f_geo", 0.0)),
        "f_vol": float(props.get("f_vol", 0.0)),
    }


def convert_hapke(props: Dict[str, Any], xml_dir=None) -> Dict[str, Any]:
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


def convert_oceanic_grasp(props: Dict[str, Any], xml_dir=None) -> Dict[str, Any]:
    """Convert GRASP oceanic model."""
    return {
        "type": "oceanic_grasp",
        "wavelength": float(props.get("wavelength", 550.0)),
        "wind_speed": float(props.get("wind_speed", 5.0)),
        "water_ior": float(props.get("water_ior", 1.33)),
    }


def convert_oceanic_6s(props: Dict[str, Any], xml_dir=None) -> Dict[str, Any]:
    """Convert legacy 6S oceanic model."""
    return {
        "type": "oceanic_6s",
        "wavelength": float(props.get("wavelength", 550.0)),
        "wind_speed": float(props.get("wind_speed", 5.0)),
        "chlorinity": float(props.get("chlorinity", 0.0)),
        "pigmentation": float(props.get("pigmentation", 0.0)),
    }


def convert_oceanic_mishchenko(props: Dict[str, Any], xml_dir=None) -> Dict[str, Any]:
    """Convert Mishchenko oceanic model."""
    return {
        "type": "oceanic_mishchenko",
        "wind_speed": float(props.get("wind_speed", 5.0)),
        "water_ior": float(props.get("water_ior", 1.33)),
    }


def convert_selectbsdf(props: Dict[str, Any], xml_dir=None) -> Dict[str, Any]:
    """Convert selector BSDF (texture-based material selection)."""
    return {
        "type": "selectbsdf",
        "texture": props.get("texture", "uniform"),
        "materials": props.get("materials", {}),
    }


def convert_measured(props: Dict[str, Any], xml_dir=None) -> Dict[str, Any]:
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


def _match_filename(filename: str, pattern: str, mode: str = "glob") -> bool:
    """Check if filename matches pattern using specified matching strategy.

    Args:
        filename: PLY filename (stem, no extension)
        pattern: Pattern to match against
        mode: Matching mode - "glob" for shell-style wildcards or "regex" for regular expressions

    Returns:
        True if filename matches pattern
    """
    if mode == "glob":
        return fnmatch.fnmatch(filename, pattern)
    elif mode == "regex":
        try:
            return bool(re.match(pattern, filename))
        except re.error as e:
            logging.error(
                f"Invalid regex pattern '{pattern}': {e}. Pattern will not match anything."
            )
            return False
    else:
        raise ValueError(
            f"Unknown matching mode: {mode}. Use 'glob' or 'regex'.\n"
            f"Examples:\n"
            f"  - glob: 'tree_*', '?_ground', '*vegetation*'\n"
            f"  - regex: r'tree_\\d+', r'(oak|pine)_.*'"
        )


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

        # Get material reference - sanitize to match IDs from _convert_materials
        material_id = _sanitize_material_id(shape["material"])

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
