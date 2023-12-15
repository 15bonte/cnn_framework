from albumentations.core.transforms_interface import ImageOnlyTransform
import numpy as np


def un_normalize_numpy(img, mean, std):
    img = img.astype(np.float32)
    img *= std
    img += mean
    return img


def un_normalize(img, mean, std):
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    return un_normalize_numpy(img, mean, std)


class UnNormalize(ImageOnlyTransform):
    """
    Unnormalize a tensor image with mean and standard deviation.
    Inspired by https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
    """

    def __init__(
        self,
        mean,
        std,
        always_apply=False,
        p=1.0,
    ):
        super(UnNormalize, self).__init__(always_apply, p)
        self.mean = mean
        self.std = std

    def apply(self, img, **params):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        return un_normalize(img, self.mean, self.std)

    def get_transform_init_args_names(self):
        return ("mean", "std")
