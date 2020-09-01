from uuid import uuid4
from abc import ABC, abstractmethod


class AbstractAgent(ABC):
    """
    The `AbstractAgent` class is the abstract base class from which agents in general inherit. The `AbstractAgent`
    incorporates the three main properties of an agent:

    * an identifier (typically a UUID4, but can be manually provided as a keyword argument called `id`),
    * the `update()` method, which embodies a time step, and
    * the `report()` method, which allows the agent to report its current state.

    Any class inheriting from `AbstractAgent` absolutely *must* implement the `update()` and `report()` methods.
    """

    def __init__(self, *args, **kwargs):
        if "id" in kwargs:
            self.id = kwargs["id"]
        else:
            self.id = str(uuid4())

    @abstractmethod
    def update(self):
        """
        Updates the agent.
        """
        pass

    @abstractmethod
    def report(self):
        """
        Draws a report from the agent.
        """
        pass
