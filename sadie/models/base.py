from abc import ABC
from enum import Enum
from typing import Iterable, Type, List
import pandas as pd
from sadie.agents import AbstractAgent
from sadie.models.exceptions import ModelRunNotCompletedError


class ModelStatus(Enum):
    NOT_RUN = 0
    RUNNING = 1
    DONE = 2


class AbstractModel(ABC):
    """
    `AbstractModel` is an abstract base class for models that implements the fundamental functionalities of a model:

    * collecting and managing agents (through the `agents` property and the `populate()` method),
    * keeping track of time (through the `time` property), and
    * running the model (using the `run()` method) and extracting results from the collector to a pandas DataFrame (`to_df()` method.)

    It is a non-self-passing model, i.e. it does not pass any information about itself (or itself to begin with) to the
    agents it runs.
    """
    def __init__(self, init_time: int = 0, max_time: int = 100, *args, **kwargs):
        self.init_time: int = init_time
        self.max_time: int = max_time
        self._time: float = self.init_time
        self.time_step: float = kwargs.get("time_step", 1)
        self._collector: list = []
        self._agents: list = []
        self._state = ModelStatus.NOT_RUN

    def add_agent(self, *agents: AbstractAgent):
        """
        Adds one or more agents `agents` that implements the top-level abstract base class for agents (`AbstractAgent`)
        to the model's population.

        At this point in time, the `add_agent()` method is permissive - it does not check whether the agent(s) added
        implement(s) or descend(s) from `AbstractAgent`, it just adds them regardless. This may change, and result in
        a `TypeError` in the future.

        :param agents: agent(s) to be added
        """
        for agent in agents:
            self._agents.append(agent)

    @property
    def agents(self) -> List[AbstractAgent]:
        """
        Returns the list of agents currently within the model's population.

        :return: list of agents currently within the model's population.
        """
        return self._agents

    @agents.setter
    def agents(self, agents: Iterable):
        self._agents = agents

    @property
    def time(self) -> float:
        """
        Returns the current time of the model.

        :return: current model time
        """
        return self._time

    @time.setter
    def time(self, value):
        raise ValueError("Cannot set time directly.")

    @property
    def state(self) -> ModelStatus:
        """
        Returns the current state of the model.

        :return: current model state
        """
        return self._state

    @state.setter
    def state(self, value):
        raise ValueError("Cannot set status: model state is set directly by the model's own processes.")

    def populate(self, n_agents: int, agent_class: Type[AbstractAgent], *args, **kwargs):
        """
        Populates the model with `n_agents` agents of the class `agent_class`. `*args` and `**kwargs` are passed on to
        the constructor responsible for instantiating agents of the specified class.

        :param n_agents: number of agents to instantiate
        :param agent_class: class from which agents are instantiated
        """
        for i in range(n_agents):
            self.add_agent(agent_class(*args, **kwargs))

    def run(self):
        """
        Runs the model.
        """
        while self._time < self.max_time:
            self._state = ModelStatus.RUNNING
            for agent in self.agents:
                agent.update()
                report = agent.report()
                report["time"] = self._time
                self._collector.append(report)
            self._time += self.time_step
        self._state = ModelStatus.DONE

    def to_df(self) -> pd.DataFrame:
        """
        Exports the model's state as a `pandas` `DataFrame`.

        :return: model state as `pandas` `DataFrame`
        :raises ModelRunNotCompletedError: if the model is still in a running state.
        """
        if self._state == ModelStatus.DONE:
            return pd.DataFrame.from_dict(self._collector)
        else:
            raise ModelRunNotCompletedError
