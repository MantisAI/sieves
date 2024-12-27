import abc
from typing import Any, Generic, Iterable, Optional, TypeVar

from sieves.data import Doc
from sieves.engines import Engine, EngineType, Executable, ExecutableResult, PromptSignature, PromptTemplate

TaskInput = TypeVar("TaskInput")
TaskOutput = TypeVar("TaskOutput")


class Task(abc.ABC):
    """Abstract base class for tasks that can be executed on documents."""

    def __init__(self, task_id: Optional[str] = None, show_progress: bool = True):
        """
        Initiates new Task.
        :param task_id: Task ID.
        :param show_progress: Whether to show progress bar for processed documents.
        """
        self._show_progress = show_progress
        self._task_id = task_id if task_id else str(self.__class__)

    @property
    def id(self) -> str:
        """Returns task ID. Used by pipeline for results and dependency management.
        :returns: Task ID.
        """
        return self._task_id


class PreTask(Generic[TaskInput], Task):
    @abc.abstractmethod
    def __call__(self, task_input: TaskInput) -> Iterable[Doc]:
        """Parse a set of files.
        :param task_input: Input to process. E.g.: files with `Iterable[sieves.data.File]`.
        :returns: Parsed input in form of documents.
        """


class PredictiveTask(Generic[PromptTemplate, PromptSignature, Executable], Task):
    def __init__(
        self,
        engine: Engine[PromptTemplate, PromptSignature, Executable],
        task_id: Optional[str] = None,
        show_progress: bool = True,
    ):
        """
        Initializes new PredictiveTask.
        :param task_id: Task ID.
        :param show_progress: Whether to show progress bar for processed documents.
        """
        super().__init__(task_id=task_id, show_progress=show_progress)
        self._engine = engine
        self._prompt_signature: PromptSignature = self._create_prompt_signature()

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
    def _create_prompt_signature(self) -> PromptSignature:
        """Creates output signature (e.g.: `Signature` in DSPy, Pydantic objects in outlines, JSON schema in
        jsonformers). This is engine-specific.
        :returns: Output signature object.
        """

    @abc.abstractmethod
    def __call__(self, docs: Iterable[Doc]) -> Iterable[Doc]:
        """Execute the task on a set of documents.
        :param docs: The documents to process.
        :returns: The processed document
        """
        # 1. Convert prompt template to its engine-specific representation.
        prompt_template = self._engine.convert_prompt_template(self.prompt_template)

        # 2. Compile expected prompt signatures.
        signature: PromptSignature = self._create_prompt_signature()

        # 3. Build executable.
        executable = self._engine.build_executable(prompt_template, signature)

        # 4. Extract values we want to inject into prompt templates to render full prompts.
        docs_values = self._extract_from_docs(docs)

        # 5. Execute prompts.
        results = executable(docs_values)

        # 6. Integrate results into docs.
        docs = self._integrate_into_docs(results, docs)

        return docs

    @abc.abstractmethod
    def _extract_from_docs(self, docs: Iterable[Doc]) -> Iterable[dict[str, Any]]:
        """Extract all values from doc instances that are to be injected into the prompts.
        :param docs: Docs to extract values from.
        :returns: All values from doc instances that are to be injected into the prompts
        """

    @abc.abstractmethod
    def _integrate_into_docs(self, results: Iterable[ExecutableResult], docs: Iterable[Doc]) -> Iterable[Doc]:
        """Integrate results into Doc instances.
        :param results: Results from prompt executable.
        :param docs: Doc instances to update.
        :returns: Updated doc instances.
        """


class PostTask(Generic[TaskOutput], Task):
    @abc.abstractmethod
    def __call__(self, docs: Iterable[Doc]) -> TaskOutput:
        """Execute the task on a set of documents.
        :param docs: The documents to process.
        :returns: The result of the task, which can be any kind of object.
        """
