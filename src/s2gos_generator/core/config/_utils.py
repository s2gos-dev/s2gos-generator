from __future__ import annotations

from s2gos_utils.io.paths import PathRef
from s2gos_utils.io.resolver import resolver


def _resolve_asset_path(filename: PathRef, asset_type: str = "asset") -> PathRef:
    """Resolve an asset path using the global resolver.

    Args:
        filename: Asset filename or path to resolve
        asset_type: Type of asset for error messages (e.g., "vegetation XML", "PLY mesh")

    Returns:
        Resolved absolute path as string

    Raises:
        ValueError: If asset cannot be found in any search path
    """
    try:
        filename = PathRef(filename)
        resolved = PathRef(resolver.resolve(filename, strict=True), cid=filename.cid)
        return resolved
    except FileNotFoundError as e:
        search_paths = [str(p) for p in resolver.paths]
        search_paths_str = (
            "\n  - ".join(search_paths) if search_paths else "(none configured)"
        )
        raise ValueError(
            f"{asset_type} file '{filename.value}' not found in any search path.\n"
            f"Searched in:\n  - {search_paths_str}\n\n"
            f"To fix this:\n"
            f"  1. Add the directory containing '{filename.value}' to asset_search_paths in s2gos_settings.yaml\n"
            f"  2. Set S2GOS_SEARCH_PATHS environment variable with additional search paths\n"
            f"  3. Provide the full absolute path instead of just the filename"
        ) from e
