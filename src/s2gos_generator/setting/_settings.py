from dynaconf import Validator
from s2gos_utils.setting import settings as util_settings


def _dem_index_path(settings=None, validator=None) -> str:
    return "dem_index.feather"


def _landcover_index_path(settings=None, validator=None) -> str:
    return "landcover_index.feather"


def _material_config_path(settings=None, validator=None) -> str:
    return "materials.json"


# Validate Generator config
util_settings.validators.register(
    Validator("generator.data.dem_root_dir", cast=str, must_exist=True),
    Validator("generator.data.dem_index_path", cast=str, default=_dem_index_path),
    Validator("generator.data.landcover_root_dir", cast=str, must_exist=True),
    Validator(
        "generator.data.landcover_index_path", cast=str, default=_landcover_index_path
    ),
    Validator(
        "generator.data.material_config_path", cast=str, default=_material_config_path
    ),
)
util_settings.validators.validate(only="generator")


# Forward s2gos_utils settings
settings = util_settings
