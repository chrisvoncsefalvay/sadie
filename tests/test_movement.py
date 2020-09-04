import unittest
from sadie.agents.spatial import MovingSpatialAgent
from random import randint, random
import numpy as np
from numpy import pi as π


class TestMovingSpatialAgent(unittest.TestCase):
    def test_odometer(self):
        agent = MovingSpatialAgent(randint(-100, 100), randint(-100, 100))
        self.assertEqual(agent.distance_traveled, 0)
        agent.move_by(1, 0)
        self.assertEqual(agent.distance_traveled, 1)
        agent.move_by(1, 0)
        self.assertEqual(agent.distance_traveled, 2)
        xoffs, yoffs = randint(-100, 100) * random(), randint(-100, 100) * random()
        agent.move_by(xoffs, yoffs)
        self.assertAlmostEqual(agent.distance_traveled, 2 + np.sqrt(xoffs ** 2 + yoffs ** 2))

    def test_move_to(self):
        xinit, yinit, xdelta, ydelta = randint(-100, 100), \
                                       randint(-100, 100), \
                                       randint(-100, 100) * random(), \
                                       randint(-100, 100) * random()
        agent = MovingSpatialAgent(x_init=xinit, y_init=yinit)
        agent.move_to(xdelta, ydelta)
        self.assertEqual(agent.position, (xdelta, ydelta))

    def test_move_by(self):
        xinit, yinit, xdelta, ydelta = randint(-100, 100), \
                                       randint(-100, 100), \
                                       randint(-100, 100) * random(), \
                                       randint(-100, 100) * random()
        agent = MovingSpatialAgent(x_init=xinit, y_init=yinit)
        agent.move_by(xdelta, ydelta)
        self.assertAlmostEqual(agent.position, (xinit + xdelta, yinit + ydelta))
        self.assertAlmostEqual(agent.distance_traveled, np.sqrt(xdelta ** 2 + ydelta ** 2))

    def test_move_p(self):
        xinit, yinit = randint(-100, 100), randint(-100, 100)
        theta, r = np.random.uniform(0, 2*π), randint(-100, 100)
        agent = MovingSpatialAgent(x_init=xinit, y_init=yinit)
        agent.move_p(theta, r)
        self.assertAlmostEqual(agent.position, (xinit + np.cos(theta) * r, yinit + np.sin(theta) * r))
        self.assertAlmostEqual(agent.distance_traveled, r)

    def test_successive_move_accumulation_in_odometer(self):
        xinit, yinit = randint(-100, 100), randint(-100, 100)
        thetas, rs = np.random.uniform(0, 2*π, 100), np.random.uniform(1, 100, 100)
        agent = MovingSpatialAgent(x_init=xinit, y_init=yinit)
        for i, j in zip(thetas, rs):
            agent.move_p(i, j)
        self.assertAlmostEqual(agent.distance_traveled, rs.sum())
