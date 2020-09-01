import unittest
from sadie.agents.spatial import SpatialAgent
from random import randint
import numpy as np
from numpy import pi as π


class TestSpatialAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.xinit, self.yinit = randint(-100, 100), randint(-100, 100)
        self.agent = SpatialAgent(x_init=self.xinit, y_init=self.yinit)

    def test_position(self):
        self.assertEqual(self.agent.position, (self.xinit, self.yinit),
                         msg="Agent does not return the correct initial position without operations.")

    def test_x(self):
        self.assertEqual(self.agent.x, self.agent.position[0])
        self.assertEqual(self.agent.x, self.xinit)

    def test_y(self):
        self.assertEqual(self.agent.y, self.agent.position[1])
        self.assertEqual(self.agent.y, self.yinit)

    def test_distance_from(self):
        agent = SpatialAgent(x_init=0, y_init=0)
        self.assertEqual(agent.distance_from(point=(1, 1)), np.sqrt(2))
        self.assertEqual(agent.distance_from(point=(0, 0)), 0)
        self.assertEqual(agent.distance_from(point=(-1, 0)), 1)
        self.assertEqual(agent.distance_from(point=(0, -1)), 1)
        self.assertEqual(agent.distance_from(point=(1, 0)), 1)
        self.assertEqual(agent.distance_from(point=(0, 1)), 1)

    def test_vector_to(self):
        agent = SpatialAgent(x_init=0, y_init=0)
        self.assertEqual(agent.vector_to((1, 0)), (0, 1))
        self.assertEqual(agent.vector_to((0, 1)), (π/2, 1))
        self.assertEqual(agent.vector_to((0, -1)), (-π/2, 1))
        self.assertEqual(agent.vector_to((-1, 0)), (π, 1))
