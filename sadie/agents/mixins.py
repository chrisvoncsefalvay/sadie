from typing import Tuple
import numpy as np
from numpy import pi as π
from sadie.agents.exceptions import NoTargetError

EPSILON: float = 1e-6


class TargetingMixin:
    """
    Targeting management mixin. Provides for target tracking, retrieval and azimuth/distance calculations to target.
    """

    def __init_subclass__(cls, **kwargs):
        cls._target = (None, None)

    @property
    def target(self) -> Tuple[float, float]:
        """
        Returns the current target of the targetable agent.

        :return: the tuple of current target coordinates
        """
        return self._target

    @target.setter
    def target(self, value):
        raise ValueError("Cannot set target directly. Please use the set_ methods to target your object.")

    def clear_target(self):
        """
        Resets the targetable object to untargeted.
        """
        self._target = (None, None)

    def set_absolute_target(self, x: float, y: float):
        """
        Sets the target of the agent to `(x, y)`.

        :param x: absolute x position to set target to
        :param y: absolute y position to set target to
        """
        self._target = (x, y)

    def set_relative_target(self, x: float, y: float):
        """
        Sets the target of the agent relative to its current position by `(x, y)`.

        :param x: x-offset of the target
        :param y: y-offset of the target
        """
        self._target = (self.x + x, self.y + y)

    def set_polar_target(self, theta: float, r: float):
        """
        Sets the target of the agent relative to its current position by reference to an azimuth and a distance.

        :param theta: azimuth to target point
        :param r: distance to target point
        """
        self._target = (self.x + np.cos(theta) * r,
                        self.y + np.sin(theta) * r)

    @property
    def target_distance(self) -> float:
        """
        Returns distance of the agent from its current target.

        :return: distance of the agent from its current target
        :raise NoTargetError: the agent is not targeted
        """
        if self._target == (None, None):
            raise NoTargetError
        else:
            return self.distance_from((self._target[0], self._target[1]))

    @property
    def target_azimuth(self) -> float:
        """
        Returns the azimuth of the agent towards its current target.

        :return: azimuth of the agent towards its current target
        :raise NoTargetError: the agent is not targeted
        """
        if self._target == (None, None):
            raise NoTargetError
        else:
            theta = np.arctan2(self._target[1] - self.y, self._target[0] - self.x)
            if theta >= 0:
                return theta
            else:
                return 2 * π + theta

    @property
    def is_on_target(self) -> bool:
        """
        Returns `True` if the object is within a tolerance margin (set as `EPSILON`, by default :math:`10^{-6}`) from
        its destination point.

        :return: `True` if the object is within the tolerance margin of the target, else `False`
        :raise NoTargetError: the agent is not targeted
        """
        if self._target == (None, None):
            raise NoTargetError
        else:
            return self.target_distance <= EPSILON
