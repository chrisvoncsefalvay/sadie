import unittest
from random import randint
import numpy as np
from numpy import pi as π
from sadie.agents.spatial import TargetableAgent, AgentStates
from sadie.agents.exceptions import NoTargetError


class TestTargetable(unittest.TestCase):
    def test_state(self):
        a = TargetableAgent(x_init=randint(-100, 100), y_init=randint(-100, 100))
        self.assertEqual(a.state, AgentStates.HALTED, msg="The initial agent state should be HALTED.")
        a.set_absolute_target(x=randint(-100, 100), y=randint(-100, 100))
        a.move()
        self.assertEqual(a.state, AgentStates.MOVING, msg="Following a move, the agent state should be MOVING.")
        a.wait()
        self.assertEqual(a.state, AgentStates.WAITING, msg="Following wait(), the agent state should be WAITING.")

    def test_wait_behaviour(self):
        a = TargetableAgent(x_init=randint(-100, 100), y_init=randint(-100, 100))
        pos = (a.x, a.y)
        a.wait()
        self.assertEqual(pos[0], a.x, msg="The expected behaviour for wait() should not change the agent's position.")
        self.assertEqual(pos[1], a.y, msg="The expected behaviour for wait() should not change the agent's position.")

    def test_move(self):
        a = TargetableAgent(0, 0)
        a.set_absolute_target(1, 0)
        a.move()
        self.assertEqual(a.x, 1)
        self.assertEqual(a.y, 0)
        self.assertEqual(a.distance_traveled, 1)

    def test_velocity_dependent_distance_reach(self):
        a = TargetableAgent(0, 0, velocity=0.5)
        a.set_polar_target(0, 2)
        self.assertFalse(a.is_on_target)
        a.move()
        self.assertFalse(a.is_on_target)
        a.move()
        self.assertFalse(a.is_on_target)
        a.move()
        self.assertFalse(a.is_on_target)
        a.move()
        self.assertTrue(a.is_on_target)

    def test_target_error(self):
        a = TargetableAgent(0, 0)
        with self.assertRaises(NoTargetError, msg="Moving an untargeted TargetableAgent should raise a NoTargetError."):
            a.move()

        with self.assertRaises(NoTargetError, msg="An untargeted TargetableAgent's target_distance property should "
                                                  "raise a NoTargetError."):
            a.target_distance()

        with self.assertRaises(NoTargetError, msg="An untargeted TargetableAgent's target_azimuth property should "
                                                  "raise a NoTargetError."):
            a.target_azimuth()

        with self.assertRaises(NoTargetError, msg="An untargeted TargetableAgent's is_on_target property should raise "
                                                  "a NoTargetError."):
            a.is_on_target()

    def test_report(self):
        ax, ay, dx, dy, tx, ty = np.random.uniform(-100, 100, 6)
        a = TargetableAgent(ax, ay)

        # In general, an untargeted TargetableAgent should return x and y position, 0 distance and None to azimuth and
        # distance to target, as well as None for target coordinates. Its state should be HALTED.
        self.assertEqual(a.report()["x"], ax, msg="At start, the agent should report the correct x position.")
        self.assertEqual(a.report()["y"], ay, msg="At start, the agent should report the correct y position.")
        self.assertIsNone(a.report()["tx"], msg="At start, the target x coordinate should be None (undefined).")
        self.assertIsNone(a.report()["ty"], msg="At start, the target y coordinate should be None (undefined).")
        self.assertIsNone(a.report()["ta"], msg="At start, the target azimuth should be None (undefined).")
        self.assertIsNone(a.report()["tr"], msg="At start, the target distance should be None (undefined).")
        self.assertEqual(a.report()["d"], 0, msg="At start, the agent should have a distance travelled of zero.")
        self.assertEqual(a.report()["st"], AgentStates.HALTED, msg="At start, the agent should be in HALTED state.")

        theta, r = np.random.uniform(0, 2*π), np.random.randint(2, 100)
        a.set_polar_target(theta, r)

        # Now that the target has been set, we expect it to show up in the targeting report.
        self.assertAlmostEqual(a.report()["x"], ax, msg="Targeting should not change the x coordinate of the agent.")
        self.assertAlmostEqual(a.report()["y"], ay, msg="Targeting should not change the x coordinate of the agent.")
        self.assertAlmostEqual(a.report()["tx"], ax + r * np.cos(theta))
        self.assertAlmostEqual(a.report()["ty"], ay + r * np.sin(theta))
        self.assertAlmostEqual(a.report()["ta"], theta, msg=f"Angle for a polar angle movement of {theta},{r} should be"
                                                            f" {theta}.")
        self.assertAlmostEqual(a.report()["tr"], r, msg=f"Distance for a polar angle movement of {theta},{r} should be "
                                                        f"{r}.")
        self.assertEqual(a.report()["d"], 0, msg="Targeting should not change the distance travelled of the agent.")
        self.assertEqual(a.report()["st"], AgentStates.HALTED, msg="After targeting, the agent should still be in a "
                                                                   "HALTED state.")

        a.move()

        # We have now moved the object. Our report should now have the same target, but at an advanced location, in a
        # moving state and having recorded a distance.
        self.assertAlmostEqual(a.report()["x"], ax + np.cos(theta), msg=f"After the first movement, the agent's x "
                                                                  f"coordinate should be {np.cos(theta)} away from "
                                                                  f"its origin (i.e. at {ax + np.cos(theta)} given an "
                                                                  f"origin of {ax}.")
        self.assertAlmostEqual(a.report()["y"], ay + np.sin(theta), msg=f"After the first movement, the agent's y "
                                                                  f"coordinate should be {np.sin(theta)} away from "
                                                                  f"its origin (i.e. at {ay + np.sin(theta)} given an "
                                                                  f"origin of {ay}.")
        self.assertAlmostEqual(a.report()["tx"], ax + r * np.cos(theta), msg="After the first movement, the target x "
                                                                             "coordinate should still be the same.")
        self.assertAlmostEqual(a.report()["ty"], ay + r * np.sin(theta), msg="After the first movement, the target y "
                                                                             "coordinate should still be the same.")
        self.assertAlmostEqual(a.report()["ta"], theta, msg="After the first movement, the target azimuth should "
                                                            "still be the same.")
        self.assertAlmostEqual(a.report()["tr"], r - 1, msg="After the first movement, the target distance should be "
                                                            "the initial distance minus one.")
        self.assertEqual(a.report()["d"], 1, msg="After the first movement, the agent's distance travelled should be "
                                                 "one.")
        self.assertEqual(a.report()["st"], AgentStates.MOVING, msg="After the first movement, the agent's state "
                                                                   "should be MOVING")
