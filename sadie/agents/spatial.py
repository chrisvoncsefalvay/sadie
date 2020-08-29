from typing import Union, Optional
import numpy as np
from numpy import pi as π
from scipy import stats
from agents import AbstractAgent


class SpatialAgent(AbstractAgent):
    """
    The `SpatialAgent` class creates spatially enabled agents in Euclidean space. These objects implement three base
    properties:

    * position (x, y), initialised by `x_init` and `y_init`,
    * distance (initialised at 0), and
    * movement (to a point, by a vector increment and by polar coordinates)
    """

    def __init__(self, x_init: Union[int, float], y_init: Union[int, float], *args, **kwargs):
        super(SpatialAgent, self).__init__()
        self.x = x_init
        self.y = y_init
        self.distance_traveled = 0

    def move_to(self, x: Union[int, float], y: Union[int, float]):
        """
        Moves the agent to the coordinates $x, y$.

        :param x: destination x coordinate
        :param y: destination y coordinate
        """
        self.x, self.y = x, y

    def move_by(self, x: Union[int, float], y: Union[int, float]):
        """
        Moves the agent by the vector $
        :param x:
        :param y:
        :return:
        """
        self.x += x
        self.y += y

    def update(self):
        pass

    def report(self):
        pass


class TargetableAgent(SpatialAgent):
    def __init__(self, x_init: Union[int, float], y_init: Union[int, float], velocity=1, epsilon=1e-4):
        super(TargetableAgent, self).__init__(x_init=x_init, y_init=y_init)
        self.velocity, self.heading = velocity, None
        self._target, self.is_targeted = [None, None], False
        self.epsilon = epsilon

    @property
    def tgt_vector(self) -> tuple:
        if self.is_targeted:
            return self._target[0] - self.x, self._target[1] - self.y
        else:
            return 0, 0

    @tgt_vector.setter
    def tgt_vector(self, value):
        raise ValueError()

    @property
    def tgt_distance(self) -> float:
        return np.sqrt(self.tgt_vector[0] ** 2 + self.tgt_vector[1] ** 2)

    @property
    def on_target(self) -> bool:
        return self.tgt_distance < self.epsilon

    def set_target(self, t_x, t_y):
        self._target = [t_x, t_y]
        self.heading = np.arctan2(self._target[1] - self.y, self._target[0] - self.x)
        self.is_targeted = True

    def set_polar_target(self, theta, r):
        self.set_target(self.x + np.cos(theta) * r, self.y + np.sin(theta) * r)

    def retarget(self):
        pass

    def move(self):
        if self.tgt_distance > self.velocity:
            self.x += np.cos(self.heading) * self.velocity
            self.y += np.sin(self.heading) * self.velocity
            self.distance_traveled += self.velocity
        else:
            self.distance_traveled += self.tgt_distance
            self.x, self.y = self._target

    def wait(self):
        pass

    def update(self):
        if self.is_targeted:
            if self.on_target:
                self.retarget()
                self.move()
            else:
                # Targeted but not on target -> move towards target
                self.move()
        else:
            self.retarget()

    def report(self):
        return {"id": self.id,
                "x": self.x,
                "y": self.y,
                "d": self.tgt_distance}


class UniformRandomWalkAgent(TargetableAgent):
    def __init__(self, x_init: Union[int, float], y_init: Union[int, float], velocity=1, epsilon=1e-4,
                 r_min: Union[int, float] = 1, r_max: Union[int, float] = 10):
        super(UniformRandomWalkAgent, self).__init__(x_init=x_init,
                                                     y_init=y_init,
                                                     velocity=velocity,
                                                     epsilon=epsilon)
        self.r_min, self.r_max = r_min, r_max

    def retarget(self):
        theta, r = np.random.uniform(0, 2 * π), np.random.uniform(self.r_min, self.r_max)
        self.set_polar_target(theta, r)


class UniformLevyRandomWalkAgent(TargetableAgent):
    def __init__(self, x_init: Union[int, float], y_init: Union[int, float], velocity=1, epsilon=1e-4):
        super(UniformLevyRandomWalkAgent, self).__init__(x_init=x_init,
                                                         y_init=y_init,
                                                         velocity=velocity,
                                                         epsilon=epsilon)

    def retarget(self):
        theta, r = np.random.uniform(0, 2 * π), stats.levy().rvs()
        self.set_polar_target(theta, r)


class BoundedDistanceLevyRandomWalkAgent(UniformLevyRandomWalkAgent):
    def __init__(self, x_init: Union[int, float], y_init: Union[int, float], velocity=1, epsilon=1e-4,
                 mu: Union[int, float] = 100, sigma: Union[int, float] = 12.5):
        super(UniformLevyRandomWalkAgent, self).__init__(x_init=x_init,
                                                         y_init=y_init,
                                                         velocity=velocity,
                                                         epsilon=epsilon)
        self.mu, self.sigma = mu, sigma

    def update(self):
        if self.is_targeted:
            if self.on_target:
                self.retarget()
                self.move()
            else:
                if self.distance_traveled > stats.norm(self.mu, self.sigma).rvs():
                    self.retarget()
                    self.distance_traveled = 0
                self.move()
        else:
            self.retarget()


class BoundedDistanceLevyRandomWalkAgentWithWait(BoundedDistanceLevyRandomWalkAgent):
    def __init__(self,
                 x_init: Union[int, float],
                 y_init: Union[int, float],
                 velocity=1, epsilon=1e-4,
                 mu: Union[int, float] = 100,
                 sigma: Union[int, float] = 12.5,
                 rest_transition_probability: float = 0.8):

        super(UniformLevyRandomWalkAgent, self).__init__(x_init=x_init,
                                                         y_init=y_init,
                                                         velocity=velocity,
                                                         epsilon=epsilon)
        self.mu, self.sigma = mu, sigma
        self.rtp = rest_transition_probability

    def wait(self):
        if stats.uniform(0, 1).rvs() > self.rtp:
            self.retarget()

    def update(self):
        if self.is_targeted:
            if self.on_target:
                self.wait()
            else:
                if self.distance_traveled > stats.norm(self.mu, self.sigma).rvs():
                    self._target = [self.x, self.y]
                    self.distance_traveled = 0
                    self.wait()
                else:
                    self.move()
        else:
            self.retarget()


class EnergyDependentLevyAgent(BoundedDistanceLevyRandomWalkAgentWithWait):
    def __init__(self,
                 x_init: Union[int, float],
                 y_init: Union[int, float],
                 velocity=1, epsilon=1e-4,
                 mu: Union[int, float] = 100,
                 sigma: Union[int, float] = 12.5,
                 mu_energy: Union[int, float] = 100,
                 sigma_energy: Union[int, float] = 20,
                 replenishment_rate: Union[int, float] = 10):

        super(UniformLevyRandomWalkAgent, self).__init__(x_init=x_init,
                                                         y_init=y_init,
                                                         velocity=velocity,
                                                         epsilon=epsilon)
        self.mu, self.sigma = mu, sigma
        self.maximum_energy = stats.norm(self.mu, self.sigma).rvs()
        self.replenishment_rate = replenishment_rate
        self.energy = self.maximum_energy
        self.state = "halted"

    def wait(self):
        self.energy += self.maximum_energy / self.replenishment_rate

    def move(self):
        if self.tgt_distance > self.velocity:
            self.x += np.cos(self.heading) * self.velocity
            self.y += np.sin(self.heading) * self.velocity
            self.distance_traveled += self.velocity
            self.energy -= self.velocity
        else:
            self.distance_traveled += self.tgt_distance
            self.energy -= self.tgt_distance
            self.x, self.y = self._target


    def update(self):
        if self.state == "halted":
            self.retarget()
            self.move()
            self.state = "moving"
        elif self.is_targeted:
            if self.on_target:
                self.wait()
                if self.energy >= 0.4 * self.maximum_energy:
                    self.retarget()
            else:
                if self.distance_traveled > stats.norm(self.mu, self.sigma).rvs():
                    self._target = [self.x, self.y]
                    self.distance_traveled = 0
                    self.wait()
                    self.state = "waiting"
                    self.retarget()
                elif self.state == "waiting":
                    if self.energy <= self.maximum_energy * 0.8:
                        self.wait()
                    else:
                        self.state = "moving"
                        self.move()
                else:
                    self.state = "moving"
                    self.move()
        else:
            self.retarget()

    def report(self):
        return {"id": self.id,
                "x": self.x,
                "y": self.y,
                "energy": self.energy,
                "efrac": self.energy/self.maximum_energy,
                "state": self.state,
                "d": self.tgt_distance,
                "trip_length": self.distance_traveled}
