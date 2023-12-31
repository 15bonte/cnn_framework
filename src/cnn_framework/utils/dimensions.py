class Dimensions:
    """
    Class to store 3d images dimensions.
    """

    def __init__(self, height=None, width=None, depth=None):
        self.height = height
        self.width = width
        self.depth = depth

    def __getitem__(self, idx):
        if idx == 0:
            return self.depth
        if idx == 1:
            return self.height
        if idx == 2:
            return self.width
        raise ValueError("Index out of range.")

    def get_updated_dimensions(self, input_dimension):
        """
            Resize dimensions are current dimensions, if not None, or input dimensions

            input_dimension: tuple (D, H, W) or Dimensions
        """

        # Depth
        if self.depth is None:
            depth = input_dimension[0]
        else:
            depth = self.depth

        # Height
        if self.height is None:
            height = input_dimension[1]
        else:
            height = self.height

        # Width
        if self.width is None:
            width = input_dimension[2]
        else:
            width = self.width

        return (depth, height, width)

    def to_tuple(self, is_3d=False):
        if is_3d:
            return (self.depth, self.height, self.width)
        return (self.height, self.width)

    def has_none(self, is_3d):
        if is_3d:
            return self.height is None or self.width is None or self.depth is None
        return self.height is None or self.width is None

    def is_strict_bigger(self, other):
        return (
            (self.height is None or self.height > other.height)
            and (self.width is None or self.width > other.width)
            and (self.depth is None or self.depth > other.depth)
        )

    def difference(self, other):
        new_height = (
            None if (self.height is None or other.height is None) else self.height - other.height
        )
        new_width = (
            None if (self.width is None or other.width is None) else self.width - other.width
        )
        new_depth = (
            None if (self.depth is None or other.depth is None) else self.depth - other.depth
        )
        return Dimensions(new_height, new_width, new_depth)
