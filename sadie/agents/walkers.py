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


class HomesickLevyWalker(BaseWalker):
    """
    Implements the Homesick Lévy walk described by `Fujihara and Miwa (2014) <https://arxiv.org/abs/1408.0427>`_. This
    walk has a home location, to which the walker returns, and a homesick probability :math:`\\alpha`, which describes
    the likelihood that the agent will set its home point as the next target.

    Note that this follows the Fujihara and Miwa paper's conceptualisation of homesickness, i.e. a turn to the home
    location is only possible at the end of a trip. Therefore, given :math:`\\alpha`, a walker will make `\\alpha^{-1}`
    trips before setting course for its home location. For a Lévy walker that can abandon a trip in progress, try
    `RapidHomesickLevyWalker`.
    """
    def __init__(self, x_init: float, y_init: float, velocity: float = 1.0, alpha: float = 0.2):
        super(HomesickLevyWalker, self).__init__(x_init=x_init, y_init=y_init, velocity=velocity)
        self.home_x, self.home_y = x_init, y_init
        self.alpha = alpha

    def update(self):
        if self.target == (None, None):
            self.retarget()
        elif self.is_on_target:
            if np.random.random() > self.alpha:
                self.retarget()
            else:
                self.set_absolute_target(self.home_x, self.home_y)
        else:
            self.move()


class RapidHomesickLevyWalker(BaseWalker):
    """
    Implements a variant of the Homesick Lévy walk described by `Fujihara and Miwa (2014)
    <https://arxiv.org/abs/1408.0427>`_ implemented in `HomesickLevyWalker`. Like the original `HomesickLevyWalker`,
    the `RapidHomesickLevyWalker` is executing a homesick Lévy walk around its point of origin. Unlike the original
    `HomesickLevyWalker`, however, the `RapidHomesickLevyWalker` determines whether to retarget for home not every time
    it has reached a location but at every movement step. Consequently, the meaning of `\\alpha` is different:
    a walker will travel a distance of `\\alpha^{-1}` before setting course for its home location.
    """
    def __init__(self, x_init: float, y_init: float, velocity: float = 1.0, alpha: float = 0.2):
        super(RapidHomesickLevyWalker, self).__init__(x_init=x_init, y_init=y_init, velocity=velocity)
        self.home_x, self.home_y = x_init, y_init
        self.alpha = alpha

    def update(self):
        if self.target == (None, None):
            self.retarget()
        elif self.is_on_target:
            if np.random.random() > self.alpha:
                self.retarget()
            else:
                self.set_absolute_target(self.home_x, self.home_y)
        else:
            if self.target == (self.home_x, self.home_y):
                self.move()
            else:
                if np.random.random() > self.alpha:
                    self.move()
                else:
                    self.set_absolute_target(self.home_x, self.home_y)
