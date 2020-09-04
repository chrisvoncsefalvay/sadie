from typing import Optional, Callable
from scipy import stats

from sadie.agents.exceptions import NoTargetError
from sadie.agents.spatial import TargetableAgent, AgentStates
import numpy as np
from numpy import pi as π


class BaseWalker(TargetableAgent):
    """
    The `BaseWalker` is a base class that implements the basic notion of a _walker_: an agent that sets and follows
    a pattern of sequentially assigned targets autonomously.

    The `BaseWalker` does not implement waiting (i.e. it's always on the move) and it does not exhibit extraneously
    sensitive behaviour (i.e. it does not react to the space it is moving across), and has a constant velocity.
    """
    def retarget(self):
        self.set_polar_target(np.random.uniform(0, 2*π), np.random.randint(1, 100))

    def update(self):
        if self.target == (None, None):
            self.retarget()
        elif self.is_on_target:
            self.retarget()
        else:
            self.move()


class WaitingUniformWalker(BaseWalker):
    def __init__(self, x_init: float, y_init: float, velocity: float = 1.0, wait_transition_probability: float = 0.4):
        super(WaitingUniformWalker, self).__init__(x_init=x_init, y_init=y_init, velocity=velocity)
        self.wait_transition_probability = wait_transition_probability

    def update(self):
        if self.target == (None, None):
            self.retarget()
        elif self.is_on_target:
            if np.random.random() > self.wait_transition_probability:
                self.retarget()
            else:
                self.wait()
        else:
            self.move()


class UniformLevyRandomWalker(BaseWalker):
    """
    The Uniform-Lévy random walker determines its trip distances according to the Lévy distribution

    .. math::

        f(x) =  \\frac{e^{- \\frac{1}{2x}}}{\\sqrt{2\\pi x^3}}

    and the azimuth of each target is drawn from a uniform distribution of :math:`(0, 2\\pi]`.
    """

    def retarget(self):
        self.set_polar_target(np.random.uniform(0, 2*π), stats.levy().rvs())


class BoundedUniformLevyRandomWalker(BaseWalker):
    def __init__(self,
                 x_init: float,
                 y_init: float,
                 velocity: float = 1.0,
                 wait_transition_probability: Optional[float] = None,
                 bounding_distribution: stats.rv_continuous = stats.norm,
                 **kwargs):
        super(BoundedUniformLevyRandomWalker, self).__init__(x_init=x_init, y_init=y_init, velocity=velocity)
        self.wait_transition_probability = wait_transition_probability
        self.bounding_distribution = bounding_distribution
        self.kwargs = kwargs

    def update(self):
        if self.target == (None, None):
            self.retarget()
        elif self.is_on_target:
            self.retarget()
        elif self.trip_distance >= self.bounding_distribution.rvs(**self.kwargs):
            self.retarget()
            self.reset_trip_odometer()
        else:
            self.move()


# class UniformRandomWalkAgent(TargetableAgent):
#     def __init__(self, x_init: float, y_init: float, velocity=1, epsilon=1e-4,
#                  r_min: float = 1, r_max: float = 10):
#         super(UniformRandomWalkAgent, self).__init__(x_init=x_init,
#                                                      y_init=y_init,
#                                                      velocity=velocity,
#                                                      epsilon=epsilon)
#         self.r_min, self.r_max = r_min, r_max
#
#     def retarget(self):
#         theta, r = np.random.uniform(0, 2 * π), np.random.uniform(self.r_min, self.r_max)
#         self.set_polar_target(theta, r)
#
#
# class UniformLevyRandomWalkAgent(TargetableAgent):
#     def __init__(self, x_init: float, y_init: float, velocity=1, epsilon=1e-4):
#         super(UniformLevyRandomWalkAgent, self).__init__(x_init=x_init,
#                                                          y_init=y_init,
#                                                          velocity=velocity,
#                                                          epsilon=epsilon)
#
#     def retarget(self):
#         theta, r = np.random.uniform(0, 2 * π), stats.levy().rvs()
#         self.set_polar_target(theta, r)
#
#
# class BoundedDistanceLevyRandomWalkAgent(UniformLevyRandomWalkAgent):
#     def __init__(self, x_init: float, y_init: float, velocity=1, epsilon=1e-4,
#                  mu: float = 100, sigma: float = 12.5):
#         super(UniformLevyRandomWalkAgent, self).__init__(x_init=x_init,
#                                                          y_init=y_init,
#                                                          velocity=velocity,
#                                                          epsilon=epsilon)
#         self.mu, self.sigma = mu, sigma
#
#     def update(self):
#         if self.is_targeted:
#             if self.on_target:
#                 self.retarget()
#                 self.move()
#             else:
#                 if self.distance_traveled > stats.norm(self.mu, self.sigma).rvs():
#                     self.retarget()
#                     self.distance_traveled = 0
#                 self.move()
#         else:
#             self.retarget()
#
#
# class BoundedDistanceLevyRandomWalkAgentWithWait(BoundedDistanceLevyRandomWalkAgent):
#     def __init__(self,
#                  x_init: float,
#                  y_init: float,
#                  velocity=1, epsilon=1e-4,
#                  mu: float = 100,
#                  sigma: float = 12.5,
#                  rest_transition_probability: float = 0.8):
#
#         super(UniformLevyRandomWalkAgent, self).__init__(x_init=x_init,
#                                                          y_init=y_init,
#                                                          velocity=velocity,
#                                                          epsilon=epsilon)
#         self.mu, self.sigma = mu, sigma
#         self.rtp = rest_transition_probability
#
#     def wait(self):
#         if stats.uniform(0, 1).rvs() > self.rtp:
#             self.retarget()
#
#     def update(self):
#         if self.is_targeted:
#             if self.on_target:
#                 self.wait()
#             else:
#                 if self.distance_traveled > stats.norm(self.mu, self.sigma).rvs():
#                     self._target = [self.x, self.y]
#                     self.distance_traveled = 0
#                     self.wait()
#                 else:
#                     self.move()
#         else:
#             self.retarget()
#
#
# class EnergyDependentLevyAgent(BoundedDistanceLevyRandomWalkAgentWithWait):
#     def __init__(self,
#                  x_init: float,
#                  y_init: float,
#                  velocity=1, epsilon=1e-4,
#                  mu: float = 100,
#                  sigma: float = 12.5,
#                  mu_energy: float = 100,
#                  sigma_energy: float = 20,
#                  replenishment_rate: float = 10):
#
#         super(UniformLevyRandomWalkAgent, self).__init__(x_init=x_init,
#                                                          y_init=y_init,
#                                                          velocity=velocity,
#                                                          epsilon=epsilon)
#         self.mu, self.sigma = mu, sigma
#         self.maximum_energy = stats.norm(self.mu, self.sigma).rvs()
#         self.replenishment_rate = replenishment_rate
#         self.energy = self.maximum_energy
#         self.state = "halted"
#
#     def wait(self):
#         self.energy += self.maximum_energy / self.replenishment_rate
#
#     def move(self):
#         if self.tgt_distance > self.velocity:
#             self.x += np.cos(self.heading) * self.velocity
#             self.y += np.sin(self.heading) * self.velocity
#             self.distance_traveled += self.velocity
#             self.energy -= self.velocity
#         else:
#             self.distance_traveled += self.tgt_distance
#             self.energy -= self.tgt_distance
#             self.x, self.y = self._target
#
#     def update(self):
#         if self.state == "halted":
#             self.retarget()
#             self.move()
#             self.state = "moving"
#         elif self.is_targeted:
#             if self.on_target:
#                 self.wait()
#                 if self.energy >= 0.4 * self.maximum_energy:
#                     self.retarget()
#             else:
#                 if self.distance_traveled > stats.norm(self.mu, self.sigma).rvs():
#                     self._target = [self.x, self.y]
#                     self.distance_traveled = 0
#                     self.wait()
#                     self.state = "waiting"
#                     self.retarget()
#                 elif self.state == "waiting":
#                     if self.energy <= self.maximum_energy * 0.8:
#                         self.wait()
#                     else:
#                         self.state = "moving"
#                         self.move()
#                 else:
#                     self.state = "moving"
#                     self.move()
#         else:
#             self.retarget()
#
#     def report(self):
#         return {"id": self.id,
#                 "x": self.x,
#                 "y": self.y,
#                 "energy": self.energy,
#                 "efrac": self.energy/self.maximum_energy,
#                 "state": self.state,
#                 "d": self.tgt_distance,
#                 "trip_length": self.distance_traveled}
