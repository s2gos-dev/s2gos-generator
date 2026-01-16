"""Resource caching for scene generation pipeline."""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from upath import UPath


@dataclass
class HashSpec:
    """Specification for computing a resource's cache hash."""

    config_paths: List[str] = field(default_factory=list)
    include_deps: bool = True
    custom_fn: Optional[Callable] = None


class CacheManifest:
    """Manages cache hash→path storage."""

    def __init__(self, cache_dir: UPath):
        self.cache_file = cache_dir / ".cache_manifest.json"
        self._data: Dict[str, Dict[str, str]] = {}
        self._load()

    def _load(self) -> None:
        if self.cache_file.exists():
            try:
                with self.cache_file.open("r") as f:
                    raw = json.load(f)
                self._data = raw.get("entries", {})
            except (json.JSONDecodeError, IOError) as e:
                logging.warning(f"Could not load cache manifest: {e}")
                self._data = {}

    def save(self) -> None:
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_file.open("w") as f:
            json.dump({"version": 1, "entries": self._data}, f, indent=2)

    def get(self, resource_id: str, hash_key: str) -> Optional[Any]:
        """Get cached entry for resource+hash, or None if not found.

        Returns:
            - UPath if the cached result is a file path that exists
            - True if the resource was computed but doesn't produce a file
            - None if no cache entry exists
        """
        entries = self._data.get(resource_id, {})
        value = entries.get(hash_key[:8])
        if value is None:
            return None
        if value == "__computed__":
            return True  # Resource was computed but doesn't produce a file
        full_path = self.cache_file.parent / value
        if full_path.exists():
            return full_path
        return None

    def set(self, resource_id: str, hash_key: str, result: Any) -> None:
        """Store hash→result mapping for a resource.

        Args:
            resource_id: Resource identifier
            hash_key: Computed hash for this resource
            result: Resource output - can be Path, UPath, or any other value
        """
        if resource_id not in self._data:
            self._data[resource_id] = {}

        if result is None:
            return

        # Only store path for Path-like results
        if isinstance(result, (Path, UPath)):
            try:
                rel_path = result.relative_to(self.cache_file.parent)
                self._data[resource_id][hash_key[:8]] = str(rel_path)
            except ValueError:
                # Path is not relative to cache dir, store absolute
                self._data[resource_id][hash_key[:8]] = str(result)
        else:
            # Non-path result (e.g., list from vegetation) - mark as computed
            self._data[resource_id][hash_key[:8]] = "__computed__"

    def get_all_entries(self, resource_id: str) -> Dict[str, str]:
        """Get all cached entries for a resource."""
        return dict(self._data.get(resource_id, {}))

    def clear(self, resource_ids: Optional[List[str]] = None) -> List[UPath]:
        """Clear cache entries and return paths of files to delete."""
        files_to_delete = []
        target_ids = resource_ids or list(self._data.keys())

        for rid in target_ids:
            if rid in self._data:
                for path_str in self._data[rid].values():
                    full_path = self.cache_file.parent / path_str
                    if full_path.exists():
                        files_to_delete.append(full_path)
                del self._data[rid]

        return files_to_delete


def _get_nested(obj: Any, path: str) -> Any:
    """Get value from nested object using dot notation."""
    current = obj
    for part in path.split("."):
        if hasattr(current, part):
            current = getattr(current, part)
        elif isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _serialize(value: Any) -> Any:
    """Serialize value for consistent hashing."""
    if isinstance(value, (UPath, Path)):
        return str(value)
    elif hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    elif isinstance(value, (list, tuple)):
        return [_serialize(v) for v in value]
    elif isinstance(value, dict):
        return {k: _serialize(v) for k, v in sorted(value.items())}
    return value


def compute_hash(
    resource_id: str,
    config: Any,
    spec: HashSpec,
    dep_hashes: Dict[str, str],
) -> str:
    """Compute deterministic hash for a resource."""
    data = {"id": resource_id}

    for path in spec.config_paths:
        data[path] = _serialize(_get_nested(config, path))

    if spec.include_deps and dep_hashes:
        data["deps"] = {k: v for k, v in sorted(dep_hashes.items())}

    if spec.custom_fn:
        data["custom"] = _serialize(spec.custom_fn(config))

    hash_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(hash_str.encode()).hexdigest()[:16]


