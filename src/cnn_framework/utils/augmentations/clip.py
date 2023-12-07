from albumentations.core.transforms_interface import ImageOnlyTransform
import numpy as np


class Clip(ImageOnlyTransform):
    """
    Clip a tensor image between min and max values.
    """

    def __init__(
        self,
        min_value,
        max_value,
        always_apply=False,
        p=1.0,
    ):
        super(Clip, self).__init__(always_apply, p)
        self.min_value = min_value
        self.max_value = max_value

    def apply(self, img, **params):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W) to be clipped.
        Returns:
            Tensor: Clipped image.
        """
        return np.clip(img, self.min_value, self.max_value)

    def get_transform_init_args_names(self):
        return ("min_value", "max_value")
