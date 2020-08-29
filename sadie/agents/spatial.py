from enum import Enum
from typing import Union, Optional, Tuple
import numpy as np
from numpy import pi as π
from scipy import stats
from sadie.agents import AbstractAgent
from sadie.agents.mixins import TargetingMixin


class AgentStates(Enum):
    HALTED = 0
    MOVING = 1
    WAITING = 2


class SpatialAgent(AbstractAgent):
    """
    The `SpatialAgent` class creates spatially enabled agents in Euclidean space. These objects are located in space,
    endowed with an `x` and `y` coordinate. These can be set directly.
    """

    def __init__(self, x_init: float, y_init: float, *args, **kwargs):
        super(SpatialAgent, self).__init__()
        self._x = x_init
        self._y = y_init

    @property
    def position(self) -> Tuple[float, float]:
        """
        Returns the position vector of the agent as a tuple :math:`(x, y)`.
        """
        return self._x, self._y

    @position.setter
    def position(self, pos: Tuple[float, float]):
        """
        Sets agent position to the tuple `pos`.

        @param pos: agent position
        """
        self._x, self._y = pos

    @property
    def x(self):
        """
        Returns the x coordinate of the agent. Equivalent to `self.position()[0]`.
        """
        return self._x

    @x.setter
    def x(self, value):
        raise ValueError("Cannot set x coordinate directly. Use either a movement method or set the position using the "
                         "`position` setter.")

    @property
    def y(self):
        """
        Returns the y coordinate of the agent. Equivalent to `self.position()[1]`.
        """
        return self._y

    @y.setter
    def y(self, value):
        raise ValueError("Cannot set y coordinate directly. Use either a movement method or set the position using the "
                         "`position` setter.")

    def distance_from(self, point: Tuple[float, float]) -> float:
        """
        Returns the agent's distance from an arbitrary point `point` defined as a tuple.

        @param point: point to which distance is calculated
        @return: distance between the agent's position and the provided point
        """
        return np.sqrt((self._x - point[0])**2 + (self._y - point[1]) ** 2)

    def vector_to(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """
        Returns the angle and magnitude :math:`(\\theta, r)` to the arbitrary point `point`.

        @param point: point to which vector is calculated
        @return: polar form vector `(theta, r)` between the agent's current location and `point`
        """
        return np.arctan2(point[1] - self.y, point[0] - self.x), self.distance_from(point)

    def update(self):
        pass

    def report(self):
        pass

class MovingSpatialAgent(SpatialAgent):
    """
    A `MovingSpatialAgent` is a `SpatialAgent` that has movement abilities, in particular:

    * `move_to(x, y)`: move to an arbitrary point :math:`(x, y)`
    * `move_by(x, y)`: move by a vector :math:`(x, y)`, and
    * `move_p(theta, r)`: move by a vector :math:`(\\theta, r)`
    """
    def __init__(self, x_init: Union[int, float], y_init: Union[int, float], *args, **kwargs):
        super(MovingSpatialAgent, self).__init__(x_init=x_init, y_init=y_init)
        self.distance_traveled = 0

    def move_to(self, x: Union[int, float], y: Union[int, float]):
        """
        Moves the agent to the coordinates :math:`(x, y)`.

        :param x: destination x coordinate
        :param y: destination y coordinate
        """
        self.distance_traveled += np.sqrt((x - self.x) ** 2 + (y - self.y) ** 2)
        self.position = (x, y)

    def move_by(self, x: Union[int, float], y: Union[int, float]):
        """
        Moves the agent by the vector :math:`(x, y)`, i.e. performs a vector addition to the current position vector.

        :param x: x increment to move by
        :param y: y increment to move by
        """
        self.distance_traveled += np.sqrt(x ** 2 + y ** 2)
        self.position = (self.x + x, self.y + y)

    def move_p(self, theta: Union[int, float], r: Union[int, float]):
        """
        Moves the agent by a vector defined by the polar coordinates :math:`(\\theta, r)`.

        :param theta: angular component :math:`\\theta`
        :param r: magnitude :math:`r`
        """
        self.distance_traveled += r
        self.position = (self.x + np.cos(theta) * r, self.y + np.sin(theta) * r)

class TargetableAgent(MovingSpatialAgent, TargetingMixin):
    """
    A `TargetableAgent` is a `MovingSpatialAgent` that has, in addition to its position, a target, which it pursues
    every time it is updated.
    """


# class TargetableAgent(MovingSpatialAgent, TargetingMixin):
#
#     def __init__(self, x_init: Union[int, float], y_init: Union[int, float], velocity=1, epsilon=1e-4):
#         super(TargetableAgent, self).__init__(x_init=x_init, y_init=y_init)
#         self._velocity: Union[int, float] = velocity
#         self._epsilon = epsilon
#         self._target: list = [None, None]
#         self._state = AgentStates.HALTED
#
#     @property
#     def is_targeted(self) -> bool:
#         """
#         Returns whether the agent has valid targeting coordinates.
#         """
#         return self._target == [None, None]
#
#     @property
#     def on_target(self) -> bool:
#         """
#         Returns whether the agent is currently on the target (defined as being within epsilon distance).
#         """
#         return np.sqrt((self._target[0] - self.x) ** 2 + (self._target[1] - self.y) ** 2) <= self._epsilon
#

#
#
# class TargetableAgent(SpatialAgent):
#     def __init__(self, ):
#         super(TargetableAgent, self).__init__(x_init=x_init, y_init=y_init)
#         self.velocity, self.heading = velocity, None
#         self._target, self.is_targeted = [None, None], False
#         self.epsilon = epsilon
#
#     @property
#     def tgt_vector(self) -> tuple:
#         if self.is_targeted:
#             return self._target[0] - self.x, self._target[1] - self.y
#         else:
#             return 0, 0
#
#     @tgt_vector.setter
#     def tgt_vector(self, value):
#         raise ValueError()
#
#     @property
#     def tgt_distance(self) -> float:
#         return np.sqrt(self.tgt_vector[0] ** 2 + self.tgt_vector[1] ** 2)
#
#     @property
#     def on_target(self) -> bool:
#         return self.tgt_distance < self.epsilon
#
#     def set_target(self, t_x, t_y):
#         self._target = [t_x, t_y]
#         self.heading = np.arctan2(self._target[1] - self.y, self._target[0] - self.x)
#         self.is_targeted = True
#
#     def set_polar_target(self, theta, r):
#         self.set_target(self.x + np.cos(theta) * r, self.y + np.sin(theta) * r)
#
#     def retarget(self):
#         pass
#
#     def move(self):
#         if self.tgt_distance > self.velocity:
#             self.x += np.cos(self.heading) * self.velocity
#             self.y += np.sin(self.heading) * self.velocity
#             self.distance_traveled += self.velocity
#         else:
#             self.distance_traveled += self.tgt_distance
#             self.x, self.y = self._target
#
#     def wait(self):
#         pass
#
#     def update(self):
#         if self.is_targeted:
#             if self.on_target:
#                 self.retarget()
#                 self.move()
#             else:
#                 # Targeted but not on target -> move towards target
#                 self.move()
#         else:
#             self.retarget()
#
#     def report(self):
#         return {"id": self.id,
#                 "x": self.x,
#                 "y": self.y,
#                 "d": self.tgt_distance}
#
