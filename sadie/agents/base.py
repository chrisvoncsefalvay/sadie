from uuid import uuid4
from abc import ABC, abstractmethod

class AbstractAgent(ABC):

    def __init__(self, *args, **kwargs):
        if "id" in kwargs:
            self.id = kwargs["id"]
        else:
            self.id = str(uuid4())

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def report(self):
        pass
