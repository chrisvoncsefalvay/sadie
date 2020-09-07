import unittest
import numpy as np
from sadie.models import simple, ModelStatus, AbstractAgent
from sadie.agents.spatial import SpatialAgent

class AbstractModelFunctionalityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = simple.SimpleModel()
        self.agt = SpatialAgent(0, 0)

    def test_add_agent(self):
        self.assertEqual(self.model.agents, [])
        self.model.add_agent(self.agt)
        self.assertEqual(len(self.model.agents), 1)

    def test_agents_listing(self):
        self.model.add_agent(self.agt)
        self.assertEqual(self.model.agents[0], self.agt)
        self.assertIsInstance(self.model.agents, list)

    def test_agents_setting(self):
        agt1, agt2 = SpatialAgent(1, 1), SpatialAgent(2, 2)
        self.model.agents = [agt1, agt2]
        self.assertEqual(self.model.agents[0], agt1)
        self.assertEqual(self.model.agents[1], agt2)

    def test_zero_starting_time(self):
        self.assertEqual(self.model.time, 0)

    def test_time_protection(self):
        with self.assertRaises(ValueError):
            self.model.time = 1

    def test_state_before_run(self):
        self.assertEqual(self.model.state, ModelStatus.NOT_RUN)

    def test_state_protection(self):
        with self.assertRaises(ValueError):
            self.model.state = ModelStatus.DONE

    def test_population_with_arbitrary_class(self):
        class TestObject(SpatialAgent):
            pass

        rx, ry = np.random.randint(-100, 100, 2)

        m = simple.SimpleModel()
        m.populate(10, TestObject, x_init = rx, y_init = ry)

        for i in m.agents:
            self.assertIsInstance(i, TestObject)

        self.assertEqual(len(m.agents), 10)
        self.assertEqual(m.agents[0].x, rx)
        self.assertEqual(m.agents[0].y, ry)

