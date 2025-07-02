"""3D mesh generation from DEM data with UV mapping support."""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import trimesh
import xarray as xr



class MeshGenerator:
    """Converts DEM data to 3D meshes with optional UV mapping and texture support."""

    def __init__(self):
        """Initialize the mesh generator."""
        logging.info("MeshGenerator initialized.")

    def dem_to_mesh(self, dem_data: xr.DataArray, handle_nans: bool = True) -> trimesh.Trimesh:
        """Convert a DEM DataArray to a Trimesh object."""
        logging.info("Converting DEM grid to 3D mesh...")
        
        dem_data.load()
        
        if 'x' in dem_data.dims and 'y' in dem_data.dims:
            x_coords = dem_data.x.values
            y_coords = dem_data.y.values
        elif 'lon' in dem_data.dims and 'lat' in dem_data.dims:
            x_coords = dem_data.lon.values
            y_coords = dem_data.lat.values
        else:
            raise ValueError("DEM data must have either (x, y) or (lon, lat) coordinates")
        
        elevation = dem_data.values
        nx, ny = len(x_coords), len(y_coords)
        
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        vertices = np.vstack([x_grid.ravel(), y_grid.ravel(), elevation.ravel()]).T

        faces = self._create_grid_faces(nx, ny)
        
        if handle_nans:
            valid_vertex_mask = ~np.isnan(vertices[:, 2])
            valid_face_mask = np.all(valid_vertex_mask[faces], axis=1)
            faces = faces[valid_face_mask]
            logging.info(f"Filtered out {np.sum(~valid_face_mask)} faces with NaN vertices")

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        mesh.remove_unreferenced_vertices()
        
        logging.info(f"Generated mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
        return mesh

    def _create_grid_faces(self, nx: int, ny: int) -> np.ndarray:
        """
        Creates triangular faces for a regular grid.

        Args:
            nx: Number of points in x direction.
            ny: Number of points in y direction.

        Returns:
            Array of face indices with shape (num_faces, 3).
        """
        i = np.arange(nx * ny).reshape(ny, nx)
        
        quad_indices = i[:-1, :-1].ravel()
        
        faces1 = np.vstack([quad_indices, quad_indices + 1, quad_indices + nx + 1]).T
        faces2 = np.vstack([quad_indices, quad_indices + nx + 1, quad_indices + nx]).T
        
        return np.vstack([faces1, faces2])

    def add_uv_coordinates(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Adds planar UV coordinates to a mesh based on its bounding box.

        Args:
            mesh: The input mesh.

        Returns:
            The mesh with UV coordinates added.
        """
        logging.info("Adding UV coordinates to mesh...")
        
        bounds = mesh.bounds
        extent = mesh.extents
        
        if extent[0] == 0:
            extent[0] = 1.0
        if extent[1] == 0:
            extent[1] = 1.0
        
        uv_coords = (mesh.vertices[:, :2] - bounds[0, :2]) / extent[:2]
        
        uv_coords = np.clip(uv_coords, 0.0, 1.0)
        
        mesh.visual.uv = uv_coords
        
        logging.info("UV coordinates added successfully")
        return mesh

    def save_mesh(self, mesh: trimesh.Trimesh, output_path: Path, format: str = "ply") -> None:
        """
        Saves a mesh to file.

        Args:
            mesh: The mesh to save.
            output_path: Path where the mesh will be saved.
            format: File format (e.g., 'ply', 'obj', 'stl').
        """
        logging.info(f"Saving mesh to {output_path}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not output_path.suffix:
            output_path = output_path.with_suffix(f".{format}")
        
        mesh.export(output_path)
        logging.info(f"Mesh saved successfully to {output_path}")

    def generate_mesh_from_dem_file(
        self,
        dem_file_path: Path,
        output_path: Path,
        add_uvs: bool = True,
        handle_nans: bool = True
    ) -> trimesh.Trimesh:
        """
        Complete pipeline: loads DEM from file, generates mesh, optionally adds UVs, and saves.

        Args:
            dem_file_path: Path to the DEM NetCDF file.
            output_path: Path where the mesh will be saved.
            add_uvs: Whether to add UV coordinates.
            handle_nans: Whether to handle NaN values in the DEM.

        Returns:
            The generated mesh.
        """
        logging.info(f"Loading DEM from {dem_file_path}")
        
        dem_data = xr.open_dataarray(dem_file_path)
        
        if isinstance(dem_data, xr.Dataset):
            if 'elevation' in dem_data.data_vars:
                dem_data = dem_data['elevation']
            else:
                dem_data = dem_data[list(dem_data.data_vars.keys())[0]]
        
        mesh = self.dem_to_mesh(dem_data, handle_nans=handle_nans)
        
        if add_uvs:
            mesh = self.add_uv_coordinates(mesh)
        
        self.save_mesh(mesh, output_path)
        
        return mesh

    def get_mesh_info(self, mesh: trimesh.Trimesh) -> dict:
        """
        Returns summary information about a mesh.

        Args:
            mesh: The mesh to analyze.

        Returns:
            Dictionary containing mesh statistics.
        """
        return {
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "bounds": mesh.bounds.tolist(),
            "extents": mesh.extents.tolist(),
            "center": mesh.center_mass.tolist(),
            "volume": mesh.volume,
            "surface_area": mesh.area,
            "is_watertight": mesh.is_watertight,
            "has_uvs": hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None
        }