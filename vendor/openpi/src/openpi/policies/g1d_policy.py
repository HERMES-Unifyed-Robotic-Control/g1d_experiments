import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image):
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class G1DInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data):
        base = _parse_image(data["observation/base_image"])
        left = _parse_image(data["observation/left_wrist_image"])
        right = _parse_image(data["observation/right_wrist_image"])
        result = {
            "state": np.asarray(data["state"], dtype=np.float32),
            "image": {
                "base_0_rgb": base,
                "left_wrist_0_rgb": left,
                "right_wrist_0_rgb": right,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "actions" in data:
            result["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class G1DOutputs(transforms.DataTransformFn):
    def __call__(self, data):
        result = dict(data)
        if "actions" in result:
            result["actions"] = np.asarray(result["actions"])[..., :16]
        return result
