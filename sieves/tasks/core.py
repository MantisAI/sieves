import abc
from typing import Any, Generic, Iterable, Optional, TypeVar

from sieves.data import Doc
from sieves.engines import (
    Engine,
    EngineType,
    InferenceMode,
    Model,
    PromptSignature,
    Result,
)

TaskInput = TypeVar("TaskInput")
TaskOutput = TypeVar("TaskOutput")
TaskPromptSignature = TypeVar("TaskPromptSignature")
TaskInferenceMode = TypeVar("TaskInferenceMode")
TaskResult = TypeVar("TaskResult")


class Task(Generic[TaskInput, TaskOutput], abc.ABC):
    """Abstract base class for tasks that can be executed on documents."""

    def __init__(self, task_id: Optional[str], show_progress: bool, include_meta: bool):
        """
        Initiates new Task.
        :param task_id: Task ID.
        :param show_progress: Whether to show progress bar for processed documents.
        :param include_meta: Whether to include meta information generated by the task.
        """
        self._show_progress = show_progress
        self._task_id = task_id if task_id else self.__class__.__name__
        self._include_meta = include_meta

    @property
    def id(self) -> str:
        """Returns task ID. Used by pipeline for results and dependency management.
        :returns: Task ID.
        """
        return self._task_id

    @abc.abstractmethod
    def __call__(self, task_input: TaskInput) -> TaskOutput:
        """Execute task.
        :param task_input: Input to process.
        :returns: Task output.
        """


class PredictiveTask(
    Generic[TaskPromptSignature, TaskResult, Model, TaskInferenceMode], Task[Iterable[Doc], Iterable[Doc]], abc.ABC
):
    def __init__(
        self,
        engine: Engine[PromptSignature, Result, Model, InferenceMode],
        task_id: Optional[str],
        show_progress: bool,
        include_meta: bool,
    ):
        """
        Initializes new PredictiveTask.
        :param task_id: Task ID.
        :param show_progress: Whether to show progress bar for processed documents.
        :param include_meta: Whether to include meta information generated by the task.
        """
        super().__init__(task_id=task_id, show_progress=show_progress, include_meta=include_meta)
        self._engine = engine
        self._prompt_signature = self._create_prompt_signature()

    @property
    @abc.abstractmethod
    def supports(self) -> set[EngineType]:
        """Returns set of engines available for this task.
        :returns: Set of engines available for this task.
        """

    @property
    @abc.abstractmethod
    def prompt_template(self) -> str:
        """Returns task's prompt template.
        Note: different engines have different expectations as how a prompt should look like. E.g. outlines supports the
        Jinja 2 templating format for insertion of values and few-shot examples, whereas DSPy integrates these things in
        a different value in the workflow and hence expects the prompt not to include these things.
        :returns: Prompt template as string.
        """

    @abc.abstractmethod
    def _create_prompt_signature(self) -> TaskPromptSignature:
        """Creates output signature (e.g.: `Signature` in DSPy, Pydantic objects in outlines, JSON schema in
        jsonformers). This is engine-specific.
        :returns: Output signature object.
        """

    @property
    @abc.abstractmethod
    def _inference_mode(self) -> TaskInferenceMode:
        """Returns inference mode.
        :returns: Inference mode.
        """

    def __call__(self, docs: Iterable[Doc]) -> Iterable[Doc]:
        """Execute the task on a set of documents.

        Note: the mypy ignore directives are because in practice, TaskX can be a superset of the X types of multiple
        engines, but there is no way in Python's current typing system to model that. E.g.: TaskInferenceMode could be
        outlines_.InferenceMode | dspy_.InferenceMode, depending on the class of the dynamically provided engine
        instance. TypeVars don't support unions however, neither do generics on a higher level of abstraction.
        We hence ignore these mypy errors, as the involved types should nonetheless be consistent.

        :param docs: The documents to process.
        :returns: The processed document
        """
        # 1. Compile expected prompt signatures.
        signature = self._create_prompt_signature()

        # 2. Build executable.
        executable = self._engine.build_executable(self._inference_mode, self.prompt_template, signature)  # type: ignore[arg-type]

        # 3. Extract values we want to inject into prompt templates to render full prompts.
        docs_values = self._extract_from_docs(docs)

        # 4. Execute prompts.
        results = executable(docs_values)

        # 5. Integrate results into docs.
        docs = self._integrate_into_docs(results, docs)  # type: ignore[arg-type]

        return docs

    @abc.abstractmethod
    def _extract_from_docs(self, docs: Iterable[Doc]) -> Iterable[dict[str, Any]]:
        """Extract all values from doc instances that are to be injected into the prompts.
        :param docs: Docs to extract values from.
        :returns: All values from doc instances that are to be injected into the prompts
        """

    @abc.abstractmethod
    def _integrate_into_docs(self, results: Iterable[TaskResult], docs: Iterable[Doc]) -> Iterable[Doc]:
        """Integrate results into Doc instances.
        :param results: Results from prompt executable.
        :param docs: Doc instances to update.
        :returns: Updated doc instances.
        """
