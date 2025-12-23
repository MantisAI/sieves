"""Bridge base class and types."""

from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Any

from sieves.data import Doc
from sieves.model_wrappers.types import ModelSettings


class Bridge[TaskPromptSignature, TaskResult, ModelWrapperInferenceMode](abc.ABC):
    """Bridge base class."""

    def __init__(self, task_id: str, prompt_instructions: str | None, overwrite: bool, model_settings: ModelSettings):
        """Initialize new bridge.

        :param task_id: Task ID.
        :param prompt_instructions: Custom prompt instructions. If None, default instructions are used.
        :param overwrite: Whether to overwrite text with produced text. Considered only by bridges for tasks producing
            fluent text - like translation, summarization, PII masking, etc.
        :param model_settings: Model settings including inference_mode.
        """
        self._task_id = task_id
        self._custom_prompt_instructions = prompt_instructions
        self._overwrite = overwrite
        self._model_settings = model_settings

    @property
    @abc.abstractmethod
    def _default_prompt_instructions(self) -> str:
        """Return default prompt instructions.

        Instructions are injected at the beginning of each prompt.

        :return: Default prompt instructions.
        """

    @property
    def _prompt_instructions(self) -> str:
        """Returns prompt instructions.

        :returns: If `_custom_prompt_instructions` is set, this is used. Otherwise, `_default_prompt_instructions` is
            used.
        """
        return self._custom_prompt_instructions or self._default_prompt_instructions

    @property
    @abc.abstractmethod
    def _prompt_example_template(self) -> str | None:
        """Return default prompt template for example injection.

        Examples are injected between instructions and conclusions.

        :return: Default prompt example template.
        """

    @property
    @abc.abstractmethod
    def _prompt_conclusion(self) -> str | None:
        """Return prompt conclusion.

        Prompt conclusions are injected at the end of each prompt.

        :return: Default prompt conclusion.
        """

    @property
    def prompt_template(self) -> str:
        """Return prompt template.

        Chains `_prompt_instructions`, `_prompt_example_template` and `_prompt_conclusion`.

        Note: different model have different expectations as to how a prompt should look like. E.g. outlines supports
        the Jinja 2 templating format for insertion of values and few-shot examples, whereas DSPy integrates these
        things in a different value in the workflow and hence expects the prompt not to include these things. Mind
        model-specific expectations when creating a prompt template.
        :return str | None: Prompt template as string. None if not used by model wrapper.
        """
        return f"""
        {self._custom_prompt_instructions or self._prompt_instructions}
        {self._prompt_example_template or ""}
        {self._prompt_conclusion or ""}
        """

    @property
    @abc.abstractmethod
    def prompt_signature(self) -> type[TaskPromptSignature] | TaskPromptSignature:
        """Create output signature.

        E.g.: `Signature` in DSPy, Pydantic objects in outlines, JSON schema in jsonformers.
        This is model type-specific.

        :return type[_TaskPromptSignature] | _TaskPromptSignature: Output signature object. This can be an instance
            (e.g. a regex string) or a class (e.g. a Pydantic class).
        """

    @property
    @abc.abstractmethod
    def inference_mode(self) -> ModelWrapperInferenceMode:
        """Return inference mode.

        :return ModelWrapperInferenceMode: Inference mode.
        """

    def extract(self, docs: Sequence[Doc]) -> Sequence[dict[str, Any]]:
        """Extract all values from doc instances that are to be injected into the prompts.

        :param docs: Docs to extract values from.
        :return: All values from doc instances that are to be injected into the prompts as a sequence.
        """
        return [{"text": doc.text if doc.text else None} for doc in docs]

    @abc.abstractmethod
    def integrate(self, results: Sequence[TaskResult], docs: list[Doc]) -> list[Doc]:
        """Integrate results into Doc instances.

        :param results: Results from prompt executable.
        :param docs: Doc instances to update.
        :return: Updated doc instances as a list.
        """

    @abc.abstractmethod
    def consolidate(self, results: Sequence[TaskResult], docs_offsets: list[tuple[int, int]]) -> Sequence[TaskResult]:
        """Consolidate results for document chunks into document results.

        :param results: Results per document chunk.
        :param docs_offsets: Chunk offsets per document. Chunks per document can be obtained with
            `results[docs_chunk_offsets[i][0]:docs_chunk_offsets[i][1]]`.
        :return: Results per document as a sequence.
        """
