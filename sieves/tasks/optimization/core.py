"""Optimizer implementation."""

import random
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
        seed: int | None = None,
        shuffle: bool = True,
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
        self._init_kwargs = dspy_init_kwargs or {}
        self._compile_kwargs = {"requires_permission_to_run": False} | (dspy_compile_kwargs or {})

    def __call__(
        self,
        signature: type[dspy.Signature] | type[dspy.Module],
        data: list[dspy.Example],
        evaluate: EvalMetric,
        verbose: bool = False,
    ) -> tuple[str, list[dspy.Example]]:
        """Optimize prompt and few-shot examples w.r.t. given signature and dataset.

        :param signature: Task to optimize.
        :param data: Dataset to use for optimization.
        :param evaluate: Evaluation metric to use for optimization.
        :param verbose: Whether to log DSPy output.
        :return: Best combination of (1) prompt and (2) fewshot-examples.
        """
        predictor = dspy.Predict(signature)
        teleprompter = dspy.MIPROv2(metric=evaluate, **(self._init_kwargs or {}), verbose=False)
        trainset, devset = self._split_data(data, self._val_frac, self._seed, self._shuffle)

        optimized_predictor: dspy.Predict = teleprompter.compile(
            predictor, trainset=trainset, valset=devset, **(self._compile_kwargs or {})
        )

        return optimized_predictor.signature.instructions, optimized_predictor.demos

    @property
    def model(self) -> dspy.LM:
        """Return model used for optimization.

        :return dspy.LM: Model used for optimization.
        """
        assert isinstance(self._model, dspy.LM)
        return self._model

    @property
    def _state(self) -> dict[str, Any]:
        """Return attributes to serialize.

        :return: Dict of attributes to serialize.
        """
        return {
            "model": self._model,
            "val_frac": self._val_frac,
            "seed": self._seed,
            "shuffle": self._shuffle,
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

    @staticmethod
    def _split_data(
        data: list[dspy.Example], val_frac: float, seed: int | None, shuffle: bool
    ) -> tuple[list[dspy.Example], list[dspy.Example]]:
        """Split data into train and validation sets.

        :param data: Dataset to split.
        :param val_frac: Fraction of data to use for validation.
        :param seed: Random seed for shuffling.
        :param shuffle: Whether to shuffle the data before splitting.
        :return: Tuple of (trainset, valset).
        """
        dataset = data.copy()
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(dataset)

        val_size = int(len(dataset) * val_frac)
        trainset = dataset[val_size:]
        valset = dataset[:val_size]

        return trainset, valset
