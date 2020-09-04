from enum import Enum
from typing import Tuple
import numpy as np
from sadie.agents import AbstractAgent
from sadie.agents.exceptions import NoTargetError
from sadie.agents.mixins import TargetingMixin


class AgentStates(Enum):
    """
    Enumeration of possible agent states.
    """
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

    @property
    def position(self) -> Tuple[float, float]:
        """
        Returns the position vector of the agent as a tuple :math:`(x, y)`.
        """
        return self.x, self.y

    @position.setter
    def position(self, pos: Tuple[float, float]):
        """
        Sets agent position to the tuple `pos`.

        :param pos: agent position
        """
        self._x, self._y = pos

    def distance_from(self, point: Tuple[float, float]) -> float:
        """
        Returns the agent's distance from an arbitrary point `point` defined as a tuple.

        :param point: point to which distance is calculated
        :return: distance between the agent's position and the provided point
        """
        return np.sqrt((self.x - point[0])**2 + (self.y - point[1]) ** 2)

    def vector_to(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """
        Returns the angle and magnitude :math:`(\\theta, r)` to the arbitrary point `point`.

        :param point: point to which vector is calculated
        :return: polar form vector `(theta, r)` between the agent's current location and `point`
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
    def __init__(self, x_init: float, y_init: float, *args, **kwargs):
        super(MovingSpatialAgent, self).__init__(x_init=x_init, y_init=y_init)
        self._distance_traveled: float = 0.0
        self._trip_distance: float = 0.0

    @property
    def distance_traveled(self):
        return self._distance_traveled

    @distance_traveled.setter
    def distance_traveled(self, value):
        self._distance_traveled = value
        self._trip_distance = value

    @property
    def trip_distance(self):
        return self._trip_distance

    @trip_distance.setter
    def trip_distance(self, value):
        raise ValueError("Cannot set trip distance directly.")

    def reset_trip_odometer(self):
        self._trip_distance = 0.0

    def move_to(self, x: float, y: float):
        """
        Moves the agent to the coordinates :math:`(x, y)`.

        :param x: destination x coordinate
        :param y: destination y coordinate
        """
        self.distance_traveled += np.sqrt((x - self.x) ** 2 + (y - self.y) ** 2)
        self.position = (x, y)

    def move_by(self, x: float, y: float):
        """
        Moves the agent by the vector :math:`(x, y)`, i.e. performs a vector addition to the current position vector.

        :param x: x increment to move by
        :param y: y increment to move by
        """
        self.distance_traveled += np.sqrt(x ** 2 + y ** 2)
        self.position = (self.x + x, self.y + y)

    def move_p(self, theta: float, r: float):
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

    In general, you should use the concrete classes derived from this class rather than use it directly, as its update
    class is not defined.
    """
    def __init__(self, x_init: float, y_init: float, velocity: float = 1.0):
        super(TargetableAgent, self).__init__(x_init=x_init, y_init=y_init)
        self._velocity = velocity
        self._state = AgentStates.HALTED

    @property
    def state(self) -> AgentStates:
        """
        Returns the agent's current state.

        :return: agent state
        """
        return self._state

    @state.setter
    def state(self, value: AgentStates):
        self._state = value

    @property
    def velocity(self) -> float:
        """
        Returns the agent's set velocity.

        :return: the agent's velocity
        """
        return self._velocity

    @velocity.setter
    def velocity(self, value: float):
        self._velocity = value

    def move(self):
        """
        Moves agent along to its target.
        """
        if self.target == (None, None):
            raise NoTargetError
        else:
            self.state = AgentStates.MOVING
            if self.target_distance < self.velocity:
                self.distance_traveled += self.target_distance
                self.move_to(*self.target)
            else:
                self.move_p(self.target_azimuth, self.velocity)

    def wait(self):
        """
        Waiting method, implementing a time step during which the agent is not changing state and does not move.
        """
        self.state = AgentStates.WAITING

    def report(self) -> dict:
        """
        Returns a dictionary of agent properties, implementing the reporting method to the model.

        :return: agent property dictionary
        """
        return {"x": self.x,
                "y": self.y,
                "tx": self.target[0],
                "ty": self.target[1],
                "ta": self.target_azimuth if self.target[0] else None,
                "tr": self.target_distance if self.target[0] else None,
                "d": self.distance_traveled,
                "st": self.state}

    def update(self):
        pass
