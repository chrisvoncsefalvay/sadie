import unittest
import numpy as np
from sadie.agents.spatial import AgentStates
from sadie.agents.walkers import BaseWalker


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

    def test_random_walk_behaviour(self):
        w = BaseWalker(0, 0)

