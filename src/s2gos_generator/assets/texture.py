"""Texture map generation from land cover data for 3D rendering."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_MATERIALS = [
    {"name": "Tree cover", "esa_class": 10, "color_8bit": (40, 75, 30), "roughness": 0.6},
    {"name": "Shrubland", "esa_class": 20, "color_8bit": (185, 170, 130), "roughness": 0.7},
    {"name": "Grassland", "esa_class": 30, "color_8bit": (140, 155, 95), "roughness": 0.7},
    {"name": "Cropland", "esa_class": 40, "color_8bit": (240, 150, 255), "roughness": 0.6},
    {"name": "Built-up", "esa_class": 50, "color_8bit": (150, 150, 150), "roughness": 0.3},
    {"name": "Bare / sparse vegetation", "esa_class": 60, "color_8bit": (220, 140, 90), "roughness": 0.8},
    {"name": "Snow and ice", "esa_class": 70, "color_8bit": (240, 240, 240), "roughness": 0.2},
    {"name": "Permanent water bodies", "esa_class": 80, "color_8bit": (0, 100, 200), "roughness": 0.1},
    {"name": "Herbaceous wetland", "esa_class": 90, "color_8bit": (80, 120, 90), "roughness": 0.4},
    {"name": "Mangroves", "esa_class": 95, "color_8bit": (0, 207, 117), "roughness": 0.4},
    {"name": "Moss and lichen", "esa_class": 100, "color_8bit": (250, 230, 160), "roughness": 0.8},
]


class TextureGenerator:
    """
    Generates texture maps from land cover data for use in 3D rendering.
    """

    def __init__(self, materials: Optional[List[Dict]] = None):
        """
        Initialize the texture generator.

        Args:
            materials: List of material definitions. If None, uses default materials.
        """
        self.materials = materials if materials is not None else DEFAULT_MATERIALS
        self.class_to_index = {mat["esa_class"]: idx for idx, mat in enumerate(self.materials)}
        logging.info(f"TextureGenerator initialized with {len(self.materials)} materials")

    def landcover_to_selection_texture(
        self,
        landcover_data: xr.DataArray,
        output_path: Path,
        flip_vertical: bool = False,
        default_material_index: int = 5
    ) -> np.ndarray:
        """
        Converts land cover classification data to a material selection texture.

        Args:
            landcover_data: xarray DataArray containing land cover class values.
            output_path: Path where the texture PNG will be saved.
            flip_vertical: If True, flips the texture vertically (for Mitsuba compatibility).
            default_material_index: Material index to use for unknown classes.

        Returns:
            The selection texture as a numpy array.
        """
        logging.info("Converting land cover data to selection texture...")
        landcover_data.load()
        
        class_values = landcover_data.values
        
        selection_texture = np.full_like(class_values, default_material_index, dtype=np.uint8)
        
        for esa_class, material_index in self.class_to_index.items():
            mask = class_values == esa_class
            selection_texture[mask] = material_index
            logging.debug(f"Mapped {np.sum(mask)} pixels from ESA class {esa_class} to material index {material_index}")
        
        if flip_vertical:
            selection_texture = np.flipud(selection_texture)
            logging.info("Applied vertical flip for rendering engine compatibility")
        
        self._save_selection_texture(selection_texture, output_path)
        
        logging.info(f"Selection texture saved to {output_path}")
        return selection_texture

    def create_preview_texture(
        self,
        landcover_data: xr.DataArray,
        output_path: Path,
        flip_vertical: bool = False  # Match the selection texture behavior
    ) -> np.ndarray:
        """
        Creates a color preview texture showing the actual material colors.

        Args:
            landcover_data: xarray DataArray containing land cover class values.
            output_path: Path where the preview PNG will be saved.
            flip_vertical: If True, flips the texture vertically.

        Returns:
            The preview texture as a numpy array with shape (height, width, 3).
        """
        logging.info("Creating color preview texture...")
        landcover_data.load()
        class_values = landcover_data.values
        
        height, width = class_values.shape
        color_texture = np.zeros((height, width, 3), dtype=np.uint8)
        
        for material in self.materials:
            esa_class = material["esa_class"]
            color = material["color_8bit"]
            mask = class_values == esa_class
            color_texture[mask] = color
        
        known_classes = set(mat["esa_class"] for mat in self.materials)
        unknown_mask = ~np.isin(class_values, list(known_classes))
        color_texture[unknown_mask] = (128, 128, 128)
        
        if flip_vertical:
            color_texture = np.flipud(color_texture)
        
        self._save_color_texture(color_texture, output_path)
        
        logging.info(f"Preview texture saved to {output_path}")
        return color_texture

    def _save_selection_texture(self, texture: np.ndarray, output_path: Path) -> None:
        """Save selection texture as a grayscale PNG."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.fromarray(texture, mode='L')  # Grayscale
        image.save(output_path)

    def _save_color_texture(self, texture: np.ndarray, output_path: Path) -> None:
        """Save color texture as RGB PNG."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.fromarray(texture, mode='RGB')
        image.save(output_path)

    def get_material_info(self) -> Dict:
        """
        Returns information about the configured materials.

        Returns:
            Dictionary containing material configuration details.
        """
        return {
            "num_materials": len(self.materials),
            "materials": self.materials,
            "class_mapping": self.class_to_index
        }

    def analyze_landcover_classes(self, landcover_data: xr.DataArray) -> Dict:
        """
        Analyzes the land cover data to show class distribution.

        Args:
            landcover_data: xarray DataArray containing land cover class values.

        Returns:
            Dictionary with class statistics.
        """
        landcover_data.load()
        class_values = landcover_data.values
        
        unique_classes, counts = np.unique(class_values, return_counts=True)
        total_pixels = class_values.size
        
        class_stats = {}
        for cls, count in zip(unique_classes, counts):
            percentage = (count / total_pixels) * 100
            material_name = "Unknown"
            
            for material in self.materials:
                if material["esa_class"] == cls:
                    material_name = material["name"]
                    break
            
            class_stats[int(cls)] = {
                "name": material_name,
                "count": int(count),
                "percentage": round(percentage, 2)
            }
        
        return {
            "total_pixels": total_pixels,
            "unique_classes": len(unique_classes),
            "class_distribution": class_stats
        }

    def generate_textures_from_file(
        self,
        landcover_file_path: Path,
        output_dir: Path,
        base_name: str,
        create_preview: bool = True
    ) -> Tuple[Path, Optional[Path]]:
        """
        Complete pipeline: loads land cover from file and generates textures.

        Args:
            landcover_file_path: Path to the land cover NetCDF file.
            output_dir: Directory where textures will be saved.
            base_name: Base name for output files.
            create_preview: Whether to create a color preview texture.

        Returns:
            Tuple of (selection_texture_path, preview_texture_path).
        """
        logging.info(f"Loading land cover data from {landcover_file_path}")
        
        landcover_data = xr.open_dataarray(landcover_file_path)
        
        if isinstance(landcover_data, xr.Dataset):
            if 'landcover' in landcover_data.data_vars:
                landcover_data = landcover_data['landcover']
            else:
                landcover_data = landcover_data[list(landcover_data.data_vars.keys())[0]]
        
        selection_path = output_dir / f"{base_name}_selection.png"
        preview_path = output_dir / f"{base_name}_preview.png" if create_preview else None
        
        self.landcover_to_selection_texture(landcover_data, selection_path)
        
        if create_preview:
            self.create_preview_texture(landcover_data, preview_path)
        
        analysis = self.analyze_landcover_classes(landcover_data)
        logging.info(f"Land cover analysis: {analysis['unique_classes']} classes found")
        
        return selection_path, preview_path


if __name__ == '__main__':
    try:
        INPUT_LANDCOVER = Path("./landcover_gobabeb_30m.nc")
        OUTPUT_DIR = Path("./data/textures")
        BASE_NAME = "landcover_gobabeb_30m"
        
        generator = TextureGenerator()
        
        material_info = generator.get_material_info()
        print(f"\nConfigured {material_info['num_materials']} materials:")
        for i, mat in enumerate(material_info['materials']):
            print(f"  {i}: {mat['name']} (ESA class {mat['esa_class']})")
        
        selection_path, preview_path = generator.generate_textures_from_file(
            landcover_file_path=INPUT_LANDCOVER,
            output_dir=OUTPUT_DIR,
            base_name=BASE_NAME,
            create_preview=True
        )
        
        print(f"\n--- Texture Generation Complete ---")
        print(f"Selection texture: {selection_path}")
        if preview_path:
            print(f"Preview texture: {preview_path}")
            
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        logging.error("Please check the input land cover file path.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")