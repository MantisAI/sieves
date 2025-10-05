"""Optimizer implementation."""

from collections.abc import Callable
from typing import Any, Self

import dspy

from sieves.serialization import Attribute, Config

EvalMetric = Callable[[dspy.Example, dspy.Prediction, Any], float]


class Optimizer:
    """Config for task optimization with DSPy.

    Uses MIPROv2 to optimize instructions and few-shot examples.
    """

    def __init__(
        self,
        model: dspy.LM | dspy.BaseLM,
        val_frac: float,
        seed: int | None,
        shuffle: bool = False,
        dspy_init_kwargs: dict[str, Any] | None = None,
        dspy_compile_kwargs: dict[str, Any] | None = None,
    ):
        """Initialize optimizer.

        :param model: Fully initialized DSPy model to use for optimization. Doesn't have to be the same as the model
            used to run the task, but more similar is better. With a lot of data you might want to pick a faster/cheaper
            model.
        :param val_frac: Fraction of examples to use for validation. Everything else is used for optimization.
        :param seed: Random seed for data splitting.
        :param shuffle: Whether to shuffle the data.
        :param dspy_init_kwargs: Optional keyword arguments to pass to DSPy optimizer at init time.
        :param dspy_compile_kwargs: Optional keyword arguments to pass to DSPy optimizer at compile time.
        """
        self._model = model
        self._val_frac = val_frac
        self._seed = seed
        self._shuffle = shuffle
        self._init_kwargs = dspy_init_kwargs
        self._compile_kwargs = dspy_compile_kwargs

    def __call__(
        self, signature: type[dspy.Signature] | type[dspy.Module], data: list[dspy.Example]
    ) -> tuple[str, list[dspy.Example]]:
        """Optimize prompt and few-shot examples w.r.t. given signature and dataset.

        :param signature: Task to optimize.
        :param data: Dataset to use for optimization.

        :return: Best combination of (1) prompt and (2) fewshot-examples.
        """

        # TODO Create Predict program.
        # TODO Convert examples.
        # TODO Define eval metric.
        # TODO Create Predict program.
        # TODO Compile.
        # TODO Extract best prompt, examples.

    @property
    def _state(self) -> dict[str, Any]:
        """Return attributes to serialize.

        :return: Dict of attributes to serialize.
        """
        return {
            "model": self._model,
            "init_kwargs": self._init_kwargs,
            "compile_kwargs": self._compile_kwargs,
        }

    def serialize(self) -> Config:
        """Serialize task.

        :return: Config instance.
        """
        return Config.create(self.__class__, {k: Attribute(value=v) for k, v in self._state.items()})

    @classmethod
    def deserialize(cls, config: Config, **kwargs: dict[str, Any]) -> Self:
        """Generate Optimizer instance from config.

        :param config: Config to generate instance from.
        :param kwargs: Values to inject into loaded config.
        :return: Deserialized Optimizer instance.
        """
        return cls(**config.to_init_dict(cls, **kwargs))
