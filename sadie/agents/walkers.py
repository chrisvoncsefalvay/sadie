from scipy import stats
from sadie.agents.spatial import TargetableAgent
import numpy as np
from numpy import pi as π


class UniformRandomWalkAgent(TargetableAgent):
    def __init__(self, x_init: float, y_init: float, velocity=1, epsilon=1e-4,
                 r_min: float = 1, r_max: float = 10):
        super(UniformRandomWalkAgent, self).__init__(x_init=x_init,
                                                     y_init=y_init,
                                                     velocity=velocity,
                                                     epsilon=epsilon)
        self.r_min, self.r_max = r_min, r_max

    def retarget(self):
        theta, r = np.random.uniform(0, 2 * π), np.random.uniform(self.r_min, self.r_max)
        self.set_polar_target(theta, r)


class UniformLevyRandomWalkAgent(TargetableAgent):
    def __init__(self, x_init: float, y_init: float, velocity=1, epsilon=1e-4):
        super(UniformLevyRandomWalkAgent, self).__init__(x_init=x_init,
                                                         y_init=y_init,
                                                         velocity=velocity,
                                                         epsilon=epsilon)

    def retarget(self):
        theta, r = np.random.uniform(0, 2 * π), stats.levy().rvs()
        self.set_polar_target(theta, r)


class BoundedDistanceLevyRandomWalkAgent(UniformLevyRandomWalkAgent):
    def __init__(self, x_init: float, y_init: float, velocity=1, epsilon=1e-4,
                 mu: float = 100, sigma: float = 12.5):
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
                 x_init: float,
                 y_init: float,
                 velocity=1, epsilon=1e-4,
                 mu: float = 100,
                 sigma: float = 12.5,
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
                 x_init: float,
                 y_init: float,
                 velocity=1, epsilon=1e-4,
                 mu: float = 100,
                 sigma: float = 12.5,
                 mu_energy: float = 100,
                 sigma_energy: float = 20,
                 replenishment_rate: float = 10):

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
