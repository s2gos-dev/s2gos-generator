"""Material definition classes."""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class Material:
    """Material definition for radiative transfer simulations."""
    type: str
    spectra: Dict[str, str] = field(default_factory=dict)  # param -> filename
    params: Dict[str, Any] = field(default_factory=dict)   # static parameters
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"type": self.type}
        # Add spectral file references
        for param, filename in self.spectra.items():
            result[param] = {"path": f"spectra/{filename}", "variable": param}
        # Add static parameters
        result.update(self.params)
        return result


@dataclass
class MaterialDefinition:
    """Simple material definition for rendering."""
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)