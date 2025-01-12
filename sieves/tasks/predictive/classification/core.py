from tasks.predictive.classification.bridges import (
    DSPyClassification,
    GliXClassification,
    HuggingFaceClassification,
    OutlinesClassification,
    TaskInferenceMode,
    TaskPromptSignature,
    TaskResult,
)

from sieves.engines import Engine, EngineType
from sieves.engines.core import InferenceMode, Model, PromptSignature, Result
from sieves.tasks.core import Bridge, PredictiveTask


class Classification(PredictiveTask[TaskPromptSignature, TaskResult, Model, TaskInferenceMode]):
    def __init__(
        self,
        labels: list[str],
        engine: Engine[PromptSignature, Result, Model, InferenceMode],
        task_id: str | None = None,
        show_progress: bool = True,
        include_meta: bool = True,
    ) -> None:
        """
        Initializes new PredictiveTask.
        :param labels: Labels to predict.
        :param task_id: Task ID.
        :param show_progress: Whether to show progress bar for processed documents.
        :param include_meta: Whether to include meta information generated by the task.
        """
        self._labels = labels
        super().__init__(engine=engine, task_id=task_id, show_progress=show_progress, include_meta=include_meta)

    def _init_bridge(self, engine_type: EngineType) -> Bridge[TaskPromptSignature, TaskInferenceMode, TaskResult]:
        """Initialize engine task.
        :returns: Engine task.
        :raises ValueError: If engine type is not supported.
        """
        match engine_type:
            case EngineType.dspy:
                bridge = DSPyClassification(self._task_id, self._labels)
            case EngineType.glix:
                bridge = GliXClassification(self._task_id, self._labels)
            case EngineType.huggingface:
                bridge = HuggingFaceClassification(self._task_id, self._labels)
            case EngineType.outlines:
                bridge = OutlinesClassification(self._task_id, self._labels)
            case _:
                raise ValueError(f"Unsupported engine type: {engine_type}")

        assert isinstance(bridge, Bridge)
        return bridge

    @property
    def supports(self) -> set[EngineType]:
        return {EngineType.outlines, EngineType.dspy, EngineType.huggingface, EngineType.glix}
