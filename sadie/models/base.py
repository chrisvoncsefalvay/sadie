from abc import ABC, abstractmethod
from agents.base import AbstractAgent
import pandas as pd

class AbstractModel(ABC):
    def __init__(self, init_time:int = 0, max_time:int = 100, *args, **kwargs):
        self.init_time: int = init_time
        self.max_time: int = max_time
        self._time = self.init_time
        self.time_step = kwargs.get("time_step", 1)
        self._collector: list = []
        self._agents: list = []

    def add_agent(self, agent: AbstractAgent):
        self.agents.append(agent)

    @property
    def agents(self):
        return self._agents

    @agents.setter
    def agents(self, value):
        raise ValueError("Cannot manually set agents. Use the add_agent() method.")

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, value):
        raise ValueError("Cannot set time directly.")

    @property
    def status(self):
        if self._time < self.max_time:
            return "running"
        else:
            return "done"

    @status.setter
    def status(self, value):
        raise ValueError("Cannot set status directly.")

    def populate(self, n_agents: int, agent_class):
        for i in range(n_agents):
            self.add_agent(agent_class())

    def run(self):
        while self._time < self.max_time:
            for agent in self.agents:
                agent.update()
                report = agent.report()
                report["time"] = self._time
                self._collector.append(report)
            self._time += self.time_step

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self._collector)
