"""Exception hierarchy for S2GOS scene generator."""


class S2GOSError(Exception):
    """Base exception for all S2GOS-related errors."""
    pass


class DataNotFoundError(S2GOSError):
    """Raised when required data files or directories cannot be found."""
    
    def __init__(self, message: str, path: str = None):
        super().__init__(message)
        self.path = path


class ConfigurationError(S2GOSError):
    """Raised when configuration is invalid or incomplete."""
    
    def __init__(self, message: str, config_field: str = None):
        super().__init__(message)
        self.config_field = config_field


class ProcessingError(S2GOSError):
    """Raised when data processing fails."""
    
    def __init__(self, message: str, stage: str = None, original_error: Exception = None):
        super().__init__(message)
        self.stage = stage
        self.original_error = original_error


class RegridError(ProcessingError):
    """Raised when regridding operations fail."""
    pass


class GeospatialError(ProcessingError):
    """Raised when geospatial operations fail."""
    pass


class MaterialError(S2GOSError):
    """Raised when material operations fail."""
    
    def __init__(self, message: str, material_id: str = None):
        super().__init__(message)
        self.material_id = material_id