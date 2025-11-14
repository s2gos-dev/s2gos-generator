from dynaconf import Validator
from s2gos_utils.setting import settings as util_settings

# Validate Generator config
# util_settings.validators.register(
#     Validator("generator.gen_test", must_exist=True), # Add Validators here
# )
# util_settings.validators.validate(only="generator")


# Forward s2gos_utils settings
settings = util_settings
