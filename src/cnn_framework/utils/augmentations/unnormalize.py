from albumentations.core.transforms_interface import ImageOnlyTransform


class UnNormalize(ImageOnlyTransform):
    """
    Unnormalize a tensor image with mean and standard deviation.
    Taken from https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
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
        for t, m, s in zip(img, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return img

    def get_transform_init_args_names(self):
        return ("mean", "std")
