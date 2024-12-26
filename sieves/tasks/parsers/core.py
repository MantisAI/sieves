import abc
from typing import Iterable

from data import Doc

from sieves.interface import TaskStage
from sieves.tasks.core import Task


class ParserTask(abc.ABC, Task):
    @property
    def stage(self) -> TaskStage:
        return TaskStage.pre

    def __call__(self, docs: Iterable[Doc]) -> Doc:
        pass
