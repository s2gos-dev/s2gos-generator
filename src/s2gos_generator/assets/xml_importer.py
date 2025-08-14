"""
Mitsuba XML importer for converting multi-material assets to S2GOS format.

This module provides utilities for importing Mitsuba scene XML files
and converting them to S2GOS-compatible asset data with proper material
library management and string-based references.
"""

import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import re
import json
import fnmatch
import hashlib
from collections import defaultdict


def import_xml_assets(
    xml_path: str,
    base_coordinate: List[float],
    object_id_prefix: str = "asset",
    elevation_offset: float = 0.0,
    scale: float = 1.0,
    fix_blender_coords: bool = True,
    material_mappings: Optional[Dict[str, str]] = None,
    pattern_type: str = "wildcard",
    validate_materials: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Import Mitsuba XML and convert to S2GOS assets with material library.
    
    Args:
        xml_path: Path to Mitsuba XML file
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
        
    Raises:
        ValueError: If material validation fails or XML parsing errors
        FileNotFoundError: If XML or PLY files are not accessible
    """
    # Parse XML to get all materials and shapes
    xml_data = _parse_mitsuba_xml(xml_path)
    
    # Convert Mitsuba materials to S2GOS material library
    material_library = {}
    if xml_data['materials']:
        material_library = _convert_materials_to_library(xml_data['materials'])
    
    # Create asset list with string material references
    assets = []
    xml_dir = Path(xml_path).parent.resolve()
    
    for shape in xml_data['shapes']:
        # Extract filename stem for material mapping
        ply_filename = Path(shape['file']).stem
        
        # Determine material reference (string only)
        material_ref = None
        if material_mappings:
            for pattern, mapped_material in material_mappings.items():
                if _match_filename(ply_filename, pattern, pattern_type):
                    material_ref = mapped_material
                    break
        
        # Use extracted material from XML if no mapping provided
        if material_ref is None:
            original_material_id = shape['material']
            if original_material_id in material_library:
                material_ref = original_material_id
            else:
                # Warning for missing material with fallback
                print(f"Warning: Material '{original_material_id}' referenced by '{ply_filename}' not found in XML. Using 'concrete' fallback.")
                material_ref = 'concrete'  # Fallback to library reference
        
        # Standard Blender→Mitsuba coordinate correction
        rotation_x = 90.0 if fix_blender_coords else 0.0
        rotation_y = 0.0
        rotation_z = 0.0
        
        # Create asset data with string material reference
        asset_data = {
            'object_id': f"{object_id_prefix}_{ply_filename}",
            'ply_path': shape['file'],
            'coordinate': base_coordinate.copy(),
            'material': material_ref,  # STRING REFERENCE ONLY
            'elevation_offset': elevation_offset,
            'scale': scale,
            'rotation_x': rotation_x,
            'rotation_y': rotation_y,
            'rotation_z': rotation_z,
        }
        
        assets.append(asset_data)
    
    # Validate if requested
    if validate_materials:
        _validate_assets_and_materials(assets, material_library)
    
    return assets, material_library


def _parse_mitsuba_xml(xml_path: str) -> Dict[str, Any]:
    """Extract materials and shapes data from Mitsuba XML file with proper property parsing.
    
    Args:
        xml_path: Path to Mitsuba XML file
        
    Returns:
        Dictionary with 'materials' and 'shapes' keys
        
    Raises:
        FileNotFoundError: If XML file doesn't exist
        ET.ParseError: If XML file is malformed
        ValueError: If essential XML structure is missing
    """
    xml_file = Path(xml_path)
    if not xml_file.exists():
        raise FileNotFoundError(f"Mitsuba XML file not found: {xml_path}")
        
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as e:
        raise ET.ParseError(f"Failed to parse Mitsuba XML file '{xml_path}': {e}")
        
    xml_dir = xml_file.parent.resolve()
    
    # Extract materials with full property parsing
    materials = {}
    bsdf_elements = tree.findall('.//bsdf')
    
    if not bsdf_elements:
        print(f"Warning: No BSDF materials found in XML file '{xml_path}'")
        
    for bsdf in bsdf_elements:
        material_id = bsdf.get('id')
        if material_id:
            try:
                materials[material_id] = _parse_bsdf_properties(bsdf)
            except Exception as e:
                print(f"Warning: Failed to parse material '{material_id}' in '{xml_path}': {e}. Skipping.")
                continue
        else:
            print(f"Warning: BSDF element without 'id' attribute found in '{xml_path}'. Skipping.")
    
    # Extract shapes (filename + material reference)
    shapes = []
    shape_elements = tree.findall('.//shape[@type="ply"]')
    
    if not shape_elements:
        print(f"Warning: No PLY shapes found in XML file '{xml_path}'")
        
    for i, shape in enumerate(shape_elements):
        filename_elem = shape.find('./string[@name="filename"]')
        if filename_elem is not None:
            filename = filename_elem.get('value')
            if not filename:
                print(f"Warning: Empty filename in shape {i} of '{xml_path}'. Skipping.")
                continue
                
            material_ref = shape.find('./ref[@name="bsdf"]')
            material_id = material_ref.get('id', 'default-bsdf') if material_ref is not None else 'default-bsdf'
            
            shapes.append({
                'file': str(xml_dir / filename),
                'material': material_id
            })
        else:
            print(f"Warning: Shape {i} in '{xml_path}' has no filename. Skipping.")
    
    if not materials and not shapes:
        raise ValueError(f"No usable materials or shapes found in Mitsuba XML file '{xml_path}'")
        
    return {'materials': materials, 'shapes': shapes}


def _parse_bsdf_properties(bsdf_element) -> Dict[str, Any]:
    """Parse BSDF element properties from Mitsuba XML.
    
    Args:
        bsdf_element: XML element for BSDF
        
    Returns:
        Dictionary with parsed material properties
        
    Raises:
        ValueError: If BSDF element structure is invalid
    """
    mat_data = {
        'type': bsdf_element.get('type', 'diffuse'),
        'properties': {}
    }
    
    # Parse nested BSDF for twosided materials
    if mat_data['type'] == 'twosided':
        nested_bsdf = bsdf_element.find('./bsdf')
        if nested_bsdf is not None:
            nested_props = _parse_bsdf_properties(nested_bsdf)
            mat_data['nested_type'] = nested_props['type']
            mat_data['properties'].update(nested_props['properties'])
        else:
            mat_data['nested_type'] = 'diffuse'
    
    # Parse properties based on element type and name
    for child in bsdf_element:
        if child.tag in ['rgb', 'spectrum', 'float', 'string', 'integer', 'boolean']:
            name = child.get('name')
            if name:
                try:
                    if child.tag == 'rgb':
                        # Parse RGB color values
                        value_str = child.get('value', '0.5 0.5 0.5')
                        try:
                            rgb_values = [float(x) for x in value_str.split()]
                            if len(rgb_values) != 3:
                                print(f"Warning: RGB value '{value_str}' doesn't have 3 components. Using default.")
                                rgb_values = [0.5, 0.5, 0.5]
                            mat_data['properties'][name] = rgb_values
                        except (ValueError, IndexError):
                            print(f"Warning: Invalid RGB value '{value_str}' for property '{name}'. Using default.")
                            mat_data['properties'][name] = [0.5, 0.5, 0.5]
                    elif child.tag == 'float':
                        try:
                            mat_data['properties'][name] = float(child.get('value', '0.0'))
                        except ValueError:
                            print(f"Warning: Invalid float value '{child.get('value')}' for property '{name}'. Using 0.0.")
                            mat_data['properties'][name] = 0.0
                    elif child.tag == 'string':
                        mat_data['properties'][name] = child.get('value', '')
                    elif child.tag == 'integer':
                        try:
                            mat_data['properties'][name] = int(child.get('value', '0'))
                        except ValueError:
                            print(f"Warning: Invalid integer value '{child.get('value')}' for property '{name}'. Using 0.")
                            mat_data['properties'][name] = 0
                    elif child.tag == 'boolean':
                        mat_data['properties'][name] = child.get('value', 'false').lower() == 'true'
                    elif child.tag == 'spectrum':
                        # For spectrum, we might have a filename reference
                        filename = child.get('filename')
                        if filename:
                            mat_data['properties'][name] = {'file': filename}
                        else:
                            # Single value spectrum
                            try:
                                mat_data['properties'][name] = float(child.get('value', '0.5'))
                            except ValueError:
                                print(f"Warning: Invalid spectrum value '{child.get('value')}' for property '{name}'. Using 0.5.")
                                mat_data['properties'][name] = 0.5
                except Exception as e:
                    print(f"Warning: Error parsing property '{name}' in BSDF: {e}. Skipping property.")
                    continue
    
    return mat_data


def _convert_materials_to_library(mitsuba_materials: Dict[str, Dict]) -> Dict[str, Dict]:
    """Convert Mitsuba materials to S2GOS material library format.
    
    Args:
        mitsuba_materials: Dictionary of Mitsuba material definitions with parsed properties
        
    Returns:
        Dictionary of S2GOS-compatible material definitions
    """
    s2gos_materials = {}
    
    for mat_id, mat_data in mitsuba_materials.items():
        try:
            s2gos_material = _convert_single_material(mat_id, mat_data)
            s2gos_materials[mat_id] = s2gos_material
        except (KeyError, ValueError, TypeError) as e:
            # These are material definition errors - provide clear error message
            raise ValueError(
                f"Material '{mat_id}' conversion failed: {e}. "
                f"Raw material data: {mat_data}"
            ) from e
        except Exception as e:
            # Unexpected errors should not be hidden
            raise RuntimeError(
                f"Unexpected error converting material '{mat_id}': {e}. "
                f"Raw material data: {mat_data}"
            ) from e
    
    return s2gos_materials


def _convert_single_material(mat_id: str, mat_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single Mitsuba material to S2GOS format.
    
    Args:
        mat_id: Material identifier
        mat_data: Parsed Mitsuba material data
        
    Returns:
        S2GOS material dictionary
        
    Raises:
        ValueError: If material type is unsupported or properties are invalid
    """
    mat_type = mat_data.get('type', 'diffuse')
    nested_type = mat_data.get('nested_type')
    properties = mat_data.get('properties', {})
    
    # Handle twosided materials by using the nested type
    if mat_type == 'twosided' and nested_type:
        mat_type = nested_type
    
    if mat_type == 'diffuse':
        # Extract reflectance property
        reflectance = properties.get('reflectance')
        if isinstance(reflectance, list) and len(reflectance) == 3:
            # RGB reflectance
            reflectance_spec = {'type': 'uniform', 'value': reflectance}
        elif isinstance(reflectance, (int, float)):
            # Scalar reflectance
            reflectance_spec = {'type': 'uniform', 'value': float(reflectance)}
        elif isinstance(reflectance, dict) and 'file' in reflectance:
            # File-based spectrum (convert to S2GOS format)
            reflectance_spec = {
                'path': f"spectra/{reflectance['file']}",
                'variable': 'reflectance'
            }
        else:
            # Default neutral gray
            reflectance_spec = {'type': 'uniform', 'value': [0.5, 0.5, 0.5]}
            
        return {
            'type': 'diffuse',
            'reflectance': reflectance_spec
        }
        
    elif mat_type == 'conductor':
        # Extract material preset or IOR values
        material_preset = properties.get('material')
        eta = properties.get('eta')
        k = properties.get('k')
        
        if material_preset:
            # Use material preset (e.g., "Cu", "Au", "Al")
            ior_spec = {'preset': material_preset}
        elif eta is not None and k is not None:
            # Use explicit eta/k values
            ior_spec = {'eta': float(eta), 'k': float(k)}
        else:
            # Default to copper
            ior_spec = {'preset': 'Cu'}
            
        result = {
            'type': 'conductor',
            'ior': ior_spec
        }
        
        # Add specular reflectance if present
        spec_refl = properties.get('specular_reflectance')
        if spec_refl is not None:
            if isinstance(spec_refl, list):
                result['specular_reflectance'] = {'type': 'uniform', 'value': spec_refl}
            elif isinstance(spec_refl, (int, float)):
                result['specular_reflectance'] = {'type': 'uniform', 'value': float(spec_refl)}
                
        return result
        
    elif mat_type == 'roughconductor':
        # Similar to conductor but with roughness
        material_preset = properties.get('material')
        eta = properties.get('eta')
        k = properties.get('k')
        distribution = properties.get('distribution', 'ggx')
        
        if material_preset:
            ior_spec = {'preset': material_preset}
        elif eta is not None and k is not None:
            ior_spec = {'eta': float(eta), 'k': float(k)}
        else:
            ior_spec = {'preset': 'Cu'}
            
        result = {
            'type': 'rough_conductor',
            'ior': ior_spec,
            'distribution': str(distribution)
        }
        
        # Handle anisotropic roughness (alpha_u, alpha_v) or isotropic (alpha, roughness)
        alpha_u = properties.get('alpha_u')
        alpha_v = properties.get('alpha_v')
        
        if alpha_u is not None or alpha_v is not None:
            # Use anisotropic roughness
            if alpha_u is not None:
                result['alpha_u'] = float(alpha_u)
            if alpha_v is not None:
                result['alpha_v'] = float(alpha_v)
        else:
            # Fall back to isotropic roughness
            alpha = properties.get('alpha', properties.get('roughness', 0.1))
            result['roughness'] = float(alpha)
        
        # Add specular reflectance if present
        spec_refl = properties.get('specular_reflectance')
        if spec_refl is not None:
            if isinstance(spec_refl, list):
                result['specular_reflectance'] = {'type': 'uniform', 'value': spec_refl}
            elif isinstance(spec_refl, (int, float)):
                result['specular_reflectance'] = {'type': 'uniform', 'value': float(spec_refl)}
                
        return result
        
    elif mat_type == 'dielectric':
        int_ior = properties.get('int_ior', 1.5)
        ext_ior = properties.get('ext_ior', 1.0)
        
        result = {
            'type': 'dielectric',
            'int_ior': float(int_ior),
            'ext_ior': float(ext_ior)
        }
        
        # Add optional properties
        for prop_name in ['specular_reflectance', 'specular_transmittance']:
            prop_val = properties.get(prop_name)
            if prop_val is not None:
                if isinstance(prop_val, list):
                    result[prop_name] = {'type': 'uniform', 'value': prop_val}
                elif isinstance(prop_val, (int, float)):
                    result[prop_name] = {'type': 'uniform', 'value': float(prop_val)}
                    
        return result
        
    elif mat_type == 'plastic':
        # Extract diffuse reflectance and coating properties
        diffuse_refl = properties.get('diffuse_reflectance', [0.5, 0.5, 0.5])
        int_ior = properties.get('int_ior', 1.49)
        ext_ior = properties.get('ext_ior', 1.0)
        alpha = properties.get('alpha', 0.01)
        nonlinear = properties.get('nonlinear', False)
        
        if isinstance(diffuse_refl, (int, float)):
            diffuse_spec = {'type': 'uniform', 'value': float(diffuse_refl)}
        elif isinstance(diffuse_refl, list):
            diffuse_spec = {'type': 'uniform', 'value': diffuse_refl}
        else:
            diffuse_spec = {'type': 'uniform', 'value': [0.5, 0.5, 0.5]}
            
        return {
            'type': 'plastic',
            'diffuse_reflectance': diffuse_spec,
            'int_ior': float(int_ior),
            'ext_ior': float(ext_ior),
            'roughness': float(alpha),
            'nonlinear': bool(nonlinear)
        }
        
    else:
        # Unsupported material type - convert to diffuse with warning
        print(f"Warning: Unsupported Mitsuba material type '{mat_type}' for material '{mat_id}'. Converting to diffuse.")
        return {
            'type': 'diffuse',
            'reflectance': {'type': 'uniform', 'value': [0.5, 0.5, 0.5]}
        }


def _match_filename(filename: str, pattern: str, pattern_type: str = "wildcard") -> bool:
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


def _validate_assets_and_materials(assets: List[Dict[str, Any]], material_library: Dict[str, Dict[str, Any]]) -> None:
    """Validate asset material references and PLY file existence.
    
    Args:
        assets: List of asset dictionaries
        material_library: Material library dictionary
        
    Raises:
        ValueError: If validation fails
    """
    validation_errors = []
    
    for asset in assets:
        asset_id = asset['object_id']
        
        # Validate material reference
        material_ref = asset['material']
        if material_ref not in material_library:
            # Check if it might be a library material (not from XML)
            validation_errors.append(
                f"Asset '{asset_id}': Material reference '{material_ref}' not found in material library. "
                f"Available from XML: {list(material_library.keys())}"
            )
        
        # Validate PLY file existence
        ply_path = Path(asset['ply_path'])
        if not ply_path.exists():
            validation_errors.append(f"Asset '{asset_id}': PLY file not found: {ply_path}")
    
    if validation_errors:
        error_msg = "Asset validation failed:\n" + "\n".join(f"  - {error}" for error in validation_errors)
        raise ValueError(error_msg)


def merge_material_libraries(*libraries: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Merge multiple material libraries, warning about conflicts.
    
    Args:
        *libraries: Variable number of material library dictionaries
        
    Returns:
        Merged material library
    """
    merged = {}
    
    for i, library in enumerate(libraries):
        for mat_id, mat_def in library.items():
            if mat_id in merged:
                print(f"Warning: Material '{mat_id}' already exists in library. Overwriting with definition from library {i+1}.")
            merged[mat_id] = mat_def
    
    return merged