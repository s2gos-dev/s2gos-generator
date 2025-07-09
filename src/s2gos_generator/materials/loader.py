"""Material configuration loader for JSON-based material definitions."""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Union

from .definitions import Material
from ..core.paths import read_json, exists


class MaterialConfigLoader:
    """Loads material configurations from JSON files."""
    
    def __init__(self, config_path: Optional[Union[Path, str]] = None):
        """Initialize the loader with a configuration file path.
        
        Args:
            config_path: Path to the JSON configuration file. If None, uses default.
        """
        if config_path is None:
            # Use the default materials.json file
            config_path = Path(__file__).parent.parent / "data" / "materials.json"
        
        self.config_path = config_path
        self._config_cache: Optional[Dict[str, Any]] = None
    
    def _load_config(self) -> Dict[str, Any]:
        """Load the JSON configuration file.
        
        Returns:
            Dictionary containing the full configuration
            
        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            json.JSONDecodeError: If the JSON is invalid
        """
        if self._config_cache is None:
            if not exists(self.config_path):
                raise FileNotFoundError(f"Material configuration file not found: {self.config_path}")
            
            self._config_cache = read_json(self.config_path)
        
        return self._config_cache
    
    def load_materials(self) -> Dict[str, Material]:
        """Load all materials from the configuration file.
        
        Returns:
            Dictionary mapping material names to Material instances
        """
        config = self._load_config()
        materials = {}
        
        for material_id, material_config in config["materials"].items():
            materials[material_id] = Material.from_dict(material_config, id=material_id)
        
        return materials
    
    def load_material(self, material_id: str) -> Material:
        """Load a specific material by ID.
        
        Args:
            material_id: The material identifier
            
        Returns:
            Material instance
            
        Raises:
            KeyError: If the material ID is not found
        """
        config = self._load_config()
        
        if material_id not in config["materials"]:
            raise KeyError(f"Material '{material_id}' not found in configuration")
        
        material_config = config["materials"][material_id]
        return Material.from_dict(material_config, id=material_id)
    
    def get_landcover_mapping(self) -> Dict[str, str]:
        """Get the landcover to material mapping.
        
        Returns:
            Dictionary mapping landcover class names to material IDs
        """
        config = self._load_config()
        return config.get("landcover_mapping", {})
    
    def get_available_materials(self) -> list[str]:
        """Get list of available material IDs.
        
        Returns:
            List of material identifiers
        """
        config = self._load_config()
        return list(config["materials"].keys())
    
    def reload(self):
        """Clear the configuration cache and reload from file."""
        self._config_cache = None


# Global default loader instance
_default_loader = MaterialConfigLoader()


def load_materials(config_path: Optional[Path] = None) -> Dict[str, Material]:
    """Load materials from configuration file.
    
    Args:
        config_path: Optional path to configuration file. Uses default if None.
        
    Returns:
        Dictionary mapping material names to Material instances
    """
    if config_path is None:
        return _default_loader.load_materials()
    else:
        loader = MaterialConfigLoader(config_path)
        return loader.load_materials()


def get_landcover_mapping(config_path: Optional[Path] = None) -> Dict[str, str]:
    """Get landcover to material mapping from configuration file.
    
    Args:
        config_path: Optional path to configuration file. Uses default if None.
        
    Returns:
        Dictionary mapping landcover class names to material IDs
    """
    if config_path is None:
        return _default_loader.get_landcover_mapping()
    else:
        loader = MaterialConfigLoader(config_path)
        return loader.get_landcover_mapping()