from enum import Enum
from typing import Tuple

import numpy as np

class TargetingMixin():
    """
    Targeting management mixin. Use only on classes that provide for a `_target` internal property.
    """

    def __init_subclass__(cls, **kwargs):
        cls._target = (None, None)

    @property
    def target(self) -> Tuple[float, float]:
        return self._target

    @target.setter
    def target(self, value):
        raise ValueError("Cannot set target directly. Please use the set_ methods to target your object.")

    def clear_target(self):
        self._target = (None, None)

    def set_absolute_target(self, x: float, y: float):
        self._target = (x, y)

    def set_relative_target(self, x: float, y: float):
        self._target = (self.x + x, self.y + y)

    def set_polar_target(self, theta: float, r: float):
        self._target = (self.x + np.cos(theta) * r,
                        self.y + np.sin(theta) * r)

    @property
    def target_distance(self) -> float:
        return self.distance_from((self.x, self.y))

    @property
    def target_azimuth(self) -> float:
        return np.arctan2(self._target[1] - self.y, self._target[0] - self.x)

