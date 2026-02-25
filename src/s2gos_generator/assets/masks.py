import numpy as np
from PIL import Image
from upath import UPath

MAX_PIXEL_VALUE = 255


def generate_buffer_mask(mask_size: int, target_size: int, output_path: UPath) -> UPath:
    """
    Generates a square buffer mask texture with center hole for target area.

    Args:
        mask_size: Total size of the mask in pixels (buffer area)
        target_size: Size of the center hole in pixels (target area)
        output_path: UPath where the mask will be saved

    Returns:
        Path to the generated mask file
    """

    mask = np.ones((mask_size, mask_size), dtype=np.uint8) * MAX_PIXEL_VALUE

    center = mask_size // 2
    half_target = target_size // 2

    start_y = center - half_target
    end_y = center + half_target
    start_x = center - half_target
    end_x = center + half_target

    mask[start_y:end_y, start_x:end_x] = 0

    from s2gos_utils.io.paths import mkdir

    mkdir(output_path.parent)
    image = Image.fromarray(mask, mode="L")
    image.save(output_path)

    return output_path
