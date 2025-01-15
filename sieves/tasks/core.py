import abc
from collections.abc import Iterable
from typing import Any, Generic, TypeVar

import pydantic

from sieves.data import Doc
from sieves.engines import (
    Engine,
    EngineInferenceMode,
    EnginePromptSignature,
    EngineResult,
    EngineType,
    Model,
)

TaskInput = TypeVar("TaskInput")
TaskOutput = TypeVar("TaskOutput")
TaskPromptSignature = TypeVar("TaskPromptSignature", covariant=True)
TaskInferenceMode = TypeVar("TaskInferenceMode", covariant=True)
TaskResult = TypeVar("TaskResult")
TaskFewshotExample = TypeVar("TaskFewshotExample", bound=pydantic.BaseModel)


class Task(Generic[TaskInput, TaskOutput], abc.ABC):
    """Abstract base class for tasks that can be executed on documents."""

    def __init__(self, task_id: str | None, show_progress: bool, include_meta: bool):
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


# @runtime_checkable
# class Bridge(Protocol[TaskPromptSignature, TaskInferenceMode, TaskResult]):
#     """Implements coupling between one Engine and one PredictiveTask."""


class Bridge(Generic[TaskPromptSignature, TaskInferenceMode, TaskResult], abc.ABC):
    def __init__(self, task_id: str, custom_prompt_template: str | None):
        """Initializes new bridge."""
        self._task_id = task_id
        self._custom_prompt_template = custom_prompt_template

    @property
    @abc.abstractmethod
    def prompt_template(self) -> str | None:
        """Returns task's prompt template.
        Note: different engines have different expectations as how a prompt should look like. E.g. outlines supports the
        Jinja 2 templating format for insertion of values and few-shot examples, whereas DSPy integrates these things in
        a different value in the workflow and hence expects the prompt not to include these things. Mind engine-specific
        expectations when creating a prompt template.
        :returns: Prompt template as string. None if none is required by engine.
        """

    @property
    @abc.abstractmethod
    def prompt_signature(self) -> TaskPromptSignature:
        """Creates output signature (e.g.: `Signature` in DSPy, Pydantic objects in outlines, JSON schema in
        jsonformers). This is engine-specific.
        :returns: Output signature object.
        """

    @property
    @abc.abstractmethod
    def inference_mode(self) -> TaskInferenceMode:
        """Returns inference mode.
        :returns: Inference mode.
        """

    @abc.abstractmethod
    def extract(self, docs: Iterable[Doc]) -> Iterable[dict[str, Any]]:
        """Extract all values from doc instances that are to be injected into the prompts.
        :param docs: Docs to extract values from.
        :returns: All values from doc instances that are to be injected into the prompts
        """

    @abc.abstractmethod
    def integrate(self, results: Iterable[TaskResult], docs: Iterable[Doc]) -> Iterable[Doc]:
        """Integrate results into Doc instances.
        :param results: Results from prompt executable.
        :param docs: Doc instances to update.
        :returns: Updated doc instances.
        """

    @abc.abstractmethod
    def consolidate(self, results: Iterable[TaskResult], docs_offsets: list[tuple[int, int]]) -> Iterable[TaskResult]:
        """Consolidates results for document chunks into document results.
        :param results: Results per document chunk.
        :param docs_offsets: Chunk offsets per document. Chunks per document can be obtained with
            results[docs_chunk_offsets[i][0]:docs_chunk_offsets[i][1]].
        :returns: Results per document.
        """


class PredictiveTask(
    Generic[TaskPromptSignature, TaskResult, Model, TaskInferenceMode, TaskFewshotExample],
    Task[Iterable[Doc], Iterable[Doc]],
    abc.ABC,
):
    def __init__(
        self,
        engine: Engine[EnginePromptSignature, EngineResult, Model, EngineInferenceMode],
        task_id: str | None,
        show_progress: bool,
        include_meta: bool,
        prompt_template: str | None = None,
        fewshot_examples: Iterable[TaskFewshotExample] = (),
    ):
        """
        Initializes new PredictiveTask.
        :param task_id: Task ID.
        :param show_progress: Whether to show progress bar for processed documents.
        :param include_meta: Whether to include meta information generated by the task.
        :param prompt_template: Custom prompt template. If None, task's default template is being used.
        :param fewshot_examples: Few-shot examples.
        """
        super().__init__(task_id=task_id, show_progress=show_progress, include_meta=include_meta)
        self._engine = engine
        self._custom_prompt_template = prompt_template
        self._bridge = self._init_bridge(EngineType.get_engine_type(self._engine))
        self._fewshot_examples = fewshot_examples

        self._validate_fewshot_examples()

    @abc.abstractmethod
    def _validate_fewshot_examples(self) -> None:
        """Validates fewshot examples.
        :raises: ValueError if fewshot examples don't pass validation.
        """

    @abc.abstractmethod
    def _init_bridge(self, engine_type: EngineType) -> Bridge[TaskPromptSignature, TaskInferenceMode, TaskResult]:
        """Initialize engine task.
        :returns: Engine task.
        """

    @property
    @abc.abstractmethod
    def supports(self) -> set[EngineType]:
        """Returns supported engine types.
        :returns: Supported engine types.
        """

    @property
    def prompt_template(self) -> str | None:
        """Returns prompt template.
        :returns: Prompt template.
        """
        return self._bridge.prompt_template

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
        docs = list(docs)

        # 1. Compile expected prompt signatures.
        signature = self._bridge.prompt_signature

        # 2. Build executable.
        executable = self._engine.build_executable(
            inference_mode=self._bridge.inference_mode,  # type: ignore[arg-type]
            prompt_template=self._bridge.prompt_template,
            prompt_signature=signature,  # type: ignore[arg-type]
            fewshot_examples=self._fewshot_examples,
        )

        # 3. Extract values from docs to inject/render those into prompt templates.
        docs_values = self._bridge.extract(docs)

        # 4. Map extracted docs values onto chunks.
        docs_chunks_offsets: list[tuple[int, int]] = []
        docs_chunks_values: list[dict[str, Any]] = []
        for doc, doc_values in zip(docs, docs_values):
            assert doc.text
            doc_chunks_values = [doc_values | {"text": chunk} for chunk in (doc.chunks or [doc.text])]
            docs_chunks_offsets.append((len(docs_chunks_values), len(docs_chunks_values) + len(doc_chunks_values)))
            docs_chunks_values.extend(doc_chunks_values)

        # 4. Execute prompts per chunk.
        results = list(executable(docs_chunks_values))
        assert len(results) == len(docs_chunks_values)

        # 5. Consolidate chunk results.
        results = list(self._bridge.consolidate(results, docs_chunks_offsets))  # type: ignore[arg-type]
        assert len(results) == len(docs)

        # 6. Integrate results into docs.
        docs = self._bridge.integrate(results, docs)  # type: ignore[arg-type]

        return docs
