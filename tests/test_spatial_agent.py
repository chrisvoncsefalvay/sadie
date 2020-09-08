import unittest
from sadie.agents.spatial import SpatialAgent, MovingSpatialAgent, TargetableAgent
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


    def test_force_set_x_y(self):
        agent = SpatialAgent(x_init=0, y_init=0)

        with self.assertRaises(ValueError):
            agent.x = 0

        with self.assertRaises(ValueError):
            agent.y = 0


    def test_force_moving_spatial_agent_trip_distance_setting(self):
        agent = MovingSpatialAgent(x_init=0, y_init=0)

        with self.assertRaises(ValueError):
            agent.trip_distance = 0


    def test_trip_odometer_reset(self):
        agent = MovingSpatialAgent(x_init=0, y_init=0)

        agent.move_to(1, 1)
        self.assertAlmostEqual(agent.trip_distance, np.sqrt(2))

        agent.reset_trip_odometer()

        self.assertEqual(agent.trip_distance, 0.0)
