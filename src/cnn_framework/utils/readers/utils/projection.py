from typing import Optional, List

from ...enum import ProjectMethods


class Projection:
    def __init__(
        self, method: ProjectMethods, axis=0, channels: Optional[List[int]] = None, proportion=1
    ) -> None:
        self.method = method
        self.axis = axis
        self.channels = channels
        self.proportion = proportion
