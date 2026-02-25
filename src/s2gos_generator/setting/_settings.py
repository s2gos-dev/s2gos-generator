from dynaconf import Validator
from dynaconf.utils.boxing import DynaBox
from s2gos_utils.setting import settings as util_settings


def _material_config_path(settings=None, validator=None) -> str:
    return "./materials.json"


# Validate Generator config
# Note that dataset validation will be done at dataset instantiation.
util_settings.validators.register(
    # DEM
    Validator("generator.dataset.dem", cast=DynaBox, must_exist=True),
    Validator("generator.dataset.dem.type", cast=str, must_exist=True),
    # Landcover
    Validator("generator.dataset.landcover", cast=DynaBox, must_exist=True),
    Validator("generator.dataset.landcover.type", cast=str, must_exist=True),
    # Files
    Validator(
        "generator.files.material_config", cast=str, default=_material_config_path
    ),
)
util_settings.validators.validate(only="generator")


# Forward s2gos_utils settings
settings = util_settings
