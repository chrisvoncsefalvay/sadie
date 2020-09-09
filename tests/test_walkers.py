import unittest
try:
    import mock
except ImportError:
    from unittest import mock

import numpy as np
from numpy import pi as π
from sadie.agents.spatial import AgentStates
from sadie.agents.walkers import BaseWalker, WaitingUniformWalker, UniformLevyRandomWalker, BoundedUniformLevyRandomWalker, HomesickLevyWalker, RapidHomesickLevyWalker, VariableVelocityWalker
from scipy.stats import beta, uniform


class TestBaseWalker(unittest.TestCase):
    def test_retargeting_behaviour_on_creation(self):
        xinit, yinit = np.random.uniform(-100, 100, 2)

        w = BaseWalker(x_init=xinit, y_init=yinit)

        self.assertEqual(w.x, xinit, msg="Starting position must be the initialising position.")
        self.assertEqual(w.y, yinit, msg="Starting position must be the initialising position.")
        self.assertEqual(w.target, (None, None), msg="At initialisation, the walker must be untargeted.")
        self.assertEqual(w.state, AgentStates.HALTED, msg="At initialisation, the walker must be in the HALTED "
                                                            "state.")

        w.update()

        self.assertEqual(w.x, xinit, msg="The first update must not change the x coordinates before the move.")
        self.assertEqual(w.y, yinit, msg="The first update must not change the y coordinates before the move.")
        self.assertIsNotNone(w.target[0], msg="The target x must be set after the first update.")
        self.assertIsNotNone(w.target[1], msg="The target y must be set after the first update.")
        wx1, wy1 = w.x, w.y
        tx1, ty1 = w.target

        w.update()

        self.assertEqual(w.state, AgentStates.MOVING)
        self.assertNotEqual(w.position, (xinit, yinit), msg="The second update should move the agent's position.")
        self.assertNotEqual(w.distance_traveled, 0, msg="At the second update, the agent must have a positive "
                                                          "nonzero distance traveled.")
        self.assertEqual(w.target, (tx1, ty1), msg="At the second update, the target should remain the same.")
        self.assertNotEqual((w.x, w.y), (wx1, wy1), msg="At the second update, the agent must have moved from its "
                                                          "position in the previous state.")

    @mock.patch("numpy.random.uniform", return_value=π)
    @mock.patch("numpy.random.randint", return_value=4)
    def test_random_walk_behaviour(self, m_azimuth, m_distance):
        w = BaseWalker(0, 0)
        self.assertEqual(w.x, 0)
        self.assertEqual(w.y, 0)
        self.assertEqual(w.target, (None, None))
        w.update()
        self.assertNotEqual(w.target, (None, None))
        w.update()
        self.assertAlmostEqual(w.target[0], np.cos(π) * 4)
        self.assertAlmostEqual(w.target[1], np.sin(π) * 4)

        for i in range(3):
            w.update()

        self.assertTrue(w.is_on_target)
        self.assertAlmostEqual(w.distance_traveled, 4)


class TestUniformWalkerBehaviours(unittest.TestCase):
    def test_always_stopping_behaviour(self):
        w = WaitingUniformWalker(0, 0, wait_transition_probability=1)
        for i in range(100):
            w.update()
        self.assertEqual(w.state, AgentStates.WAITING)

    def test_never_stopping_behaviour(self):
        w = WaitingUniformWalker(0, 0, wait_transition_probability=0)
        for i in range(1000):
            w.update()
            if w.is_on_target:
                self.assertNotEqual(w.state, AgentStates.WAITING)

    @mock.patch("numpy.random.uniform", return_value=π)
    @mock.patch("numpy.random.randint", return_value=4)
    def test_wait_uniform_walk_behaviour(self, m_azimuth, m_distance):
        w = WaitingUniformWalker(0, 0, wait_transition_probability=0)
        self.assertEqual(w.x, 0)
        self.assertEqual(w.y, 0)
        self.assertEqual(w.target, (None, None))
        w.update()
        self.assertNotEqual(w.target, (None, None))
        w.update()
        self.assertAlmostEqual(w.target[0], np.cos(π) * 4)
        self.assertAlmostEqual(w.target[1], np.sin(π) * 4)

        for i in range(3):
            w.update()

        self.assertTrue(w.is_on_target)
        self.assertAlmostEqual(w.distance_traveled, 4)

    def test_wait_transition_probability_is_captured(self):
        wtp = np.random.random()
        w = WaitingUniformWalker(0, 0, wait_transition_probability=wtp)
        self.assertEqual(w.wait_transition_probability, wtp)

    def test_retargets_on_no_target(self):
        w = WaitingUniformWalker(0, 0, wait_transition_probability=np.random.random())
        self.assertEqual(w.target, (None, None))
        w.update()
        self.assertNotEqual(w.target, (None, None))

    @mock.patch("numpy.random.uniform", return_value=π)
    def test_uniform_levy_walker_retargeting_behaviour(self, azimuth):
        w = UniformLevyRandomWalker(0, 0)
        self.assertEqual(w.target, (None, None))
        w.retarget()
        self.assertEqual(w.target_azimuth, π)


class TestBoundedLevyWalk(unittest.TestCase):
    def test_retrieve_scale_factor(self):
        w = BoundedUniformLevyRandomWalker(*np.random.random(2), scale_factor=2)
        self.assertEqual(w.scale_factor, 2)

    def test_retrieve_distribution(self):
        w = BoundedUniformLevyRandomWalker(*np.random.random(2), bounding_distribution = beta)
        self.assertEqual(w.bounding_distribution, beta)

    def test_kwarg_configuration_of_distribution(self):
        w = BoundedUniformLevyRandomWalker(*np.random.random(2), bounding_distribution = beta, a = 3)
        self.assertEqual(w.kwargs, {"a": 3})


class TestHomesickLevyWalkers(unittest.TestCase):
    def test_homesick_levy_walker_parametrisation(self):
        ix, iy, a = np.random.random(3)
        w = HomesickLevyWalker(x_init=ix, y_init=iy, alpha=a)
        self.assertEqual(w.home_x, ix)
        self.assertEqual(w.home_y, iy)
        self.assertEqual(w.alpha, a)

    def test_rapid_homesick_levy_walker_parametrisation(self):
        ix, iy, a = np.random.random(3)
        w = RapidHomesickLevyWalker(x_init=ix, y_init=iy, alpha=a)
        self.assertEqual(w.home_x, ix)
        self.assertEqual(w.home_y, iy)
        self.assertEqual(w.alpha, a)


class TestVariableVelocityWalker(unittest.TestCase):
    def test_retrieve_distribution(self):
        w = VariableVelocityWalker(*np.random.randint(-100, 100, 2), reflect=True, velocity_distribution=beta, a=2, b=3)
        self.assertTrue(w.reflect)
        self.assertTrue(w.velocity_distribution, beta)
        self.assertTrue(w.kwargs, {"a": 2, "b": 3})

    def test_velocity_distribution_setting(self):
        w = VariableVelocityWalker(*np.random.randint(-100, 100, 2), velocity=0, velocity_distribution=uniform, loc=10, scale=1)
        self.assertEqual(w.velocity, 0)
        w.update()
        self.assertNotEqual(w.velocity, 0)
        self.assertGreaterEqual(w.velocity, 9)
        self.assertLessEqual(w.velocity, 11)

    def test_reflecting_behaviour(self):
        w = VariableVelocityWalker(*np.random.randint(-100, 100, 2), velocity=0, velocity_distribution=uniform, loc=0, scale=3)
        w.update()
        self.assertGreater(w.velocity, 0)

    def test_capping_behaviour(self):
        w = VariableVelocityWalker(*np.random.randint(-100, 100, 2), velocity=0, reflect=False, velocity_distribution=uniform, loc=0, scale=3)
        w.update()
        self.assertGreaterEqual(w.velocity, 0)
