from __future__ import annotations

import abc
import enum
from collections.abc import Iterable
from typing import Any, Generic

import datasets
import pydantic

from sieves.data import Doc
from sieves.engines import (
    Engine,
    EngineInferenceMode,
    EngineModel,
    EnginePromptSignature,
    EngineResult,
    EngineType,
)
from sieves.serialization import Config, Serializable
from sieves.tasks.core import Task
from sieves.tasks.predictive.bridges import TaskBridge, TaskPromptSignature, TaskResult


class PredictiveTask(
    Generic[TaskPromptSignature, TaskResult, TaskBridge],
    Task,
    abc.ABC,
):
    def __init__(
        self,
        engine: Engine[EnginePromptSignature, EngineResult, EngineModel, EngineInferenceMode],
        task_id: str | None,
        show_progress: bool,
        include_meta: bool,
        overwrite: bool,
        prompt_template: str | None,
        prompt_signature_desc: str | None,
        fewshot_examples: Iterable[pydantic.BaseModel],
    ):
        """
        Initializes new PredictiveTask.
        :param task_id: Task ID.
        :param show_progress: Whether to show progress bar for processed documents.
        :param include_meta: Whether to include meta information generated by the task.
        :param overwrite: Some tasks, e.g. anonymization or translation, output a modified version of the input text.
            If True, these tasks overwrite the original document text. If False, the result will just be stored in the
            documents' `.results` field.
        :param prompt_template: Custom prompt template. If None, default template is being used.
        :param prompt_signature_desc: Custom prompt signature description. If None, default will be used.
        :param fewshot_examples: Few-shot examples.
        """
        super().__init__(task_id=task_id, show_progress=show_progress, include_meta=include_meta)
        self._engine = engine
        self._overwrite = overwrite
        self._custom_prompt_template = prompt_template
        self._custom_prompt_signature_desc = prompt_signature_desc
        self._bridge = self._init_bridge(EngineType.get_engine_type(self._engine))
        self._fewshot_examples = fewshot_examples

        self._validate_fewshot_examples()

    def _validate_fewshot_examples(self) -> None:
        """Validates fewshot examples.
        :raises: ValueError if fewshot examples don't pass validation.
        """
        pass

    @abc.abstractmethod
    def _init_bridge(self, engine_type: EngineType) -> TaskBridge:
        """Initialize bridge.
        :param engine_type: Type of engine to initialize bridge for.
        :return _TaskBridge: Engine task bridge.
        """

    @property
    @abc.abstractmethod
    def supports(self) -> set[EngineType]:
        """Returns supported engine types.
        :return set[EngineType]: Supported engine types.
        """

    @property
    def prompt_template(self) -> str | None:
        """Returns prompt template.
        :return str | None: Prompt template.
        """
        prompt_template = self._bridge.prompt_template
        assert prompt_template is None or isinstance(prompt_template, str)
        return prompt_template

    @property
    def prompt_signature_description(self) -> str | None:
        """Returns prompt signature description.
        :return str | None: Prompt signature description.
        """
        sig_desc = self._bridge.prompt_signature_description
        assert sig_desc is None or isinstance(sig_desc, str)
        return sig_desc

    def __call__(self, docs: Iterable[Doc]) -> Iterable[Doc]:
        """Execute the task on a set of documents.

        Note: the mypy ignore directives are because in practice, TaskX can be a superset of the X types of multiple
        engines, but there is no way in Python's current typing system to model that. E.g.: TaskInferenceMode could be
        outlines_.InferenceMode | dspy_.InferenceMode, depending on the class of the dynamically provided engine
        instance. TypeVars don't support unions however, neither do generics on a higher level of abstraction.
        We hence ignore these mypy errors, as the involved types should nonetheless be consistent.

        :param docs: Documents to process.
        :return Iterable[Doc]: Processed documents.
        """
        docs = list(docs)
        # 1. Compile expected prompt signatures.
        signature = self._bridge.prompt_signature

        # 2. Build executable.
        assert isinstance(self._bridge.inference_mode, enum.Enum)
        executable = self._engine.build_executable(
            inference_mode=self._bridge.inference_mode,
            prompt_template=self.prompt_template,
            prompt_signature=signature,
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
        # 5. Execute prompts per chunk.
        results = list(executable(docs_chunks_values))
        assert len(results) == len(docs_chunks_values)

        # 6. Consolidate chunk results.
        results = list(self._bridge.consolidate(results, docs_chunks_offsets))
        assert len(results) == len(docs)

        # 7. Integrate results into docs.
        docs = self._bridge.integrate(results, docs)

        return docs

    @property
    def _state(self) -> dict[str, Any]:
        return {
            **super()._state,
            "engine": self._engine.serialize(),
            "prompt_template": self._custom_prompt_template,
            "prompt_signature_desc": self._custom_prompt_signature_desc,
            "fewshot_examples": self._fewshot_examples,
        }

    @classmethod
    def deserialize(
        cls, config: Config, **kwargs: dict[str, Any]
    ) -> PredictiveTask[TaskPromptSignature, TaskResult, TaskBridge]:
        """Generate PredictiveTask instance from config.
        :param config: Config to generate instance from.
        :param kwargs: Values to inject into loaded config.
        :return PredictiveTask[_TaskPromptSignature, _TaskResult, _TaskBridge]: Deserialized PredictiveTask instance.
        """
        # Validate engine config.
        assert hasattr(config, "engine")
        assert isinstance(config.engine.value, Config)
        engine_config = config.engine.value
        engine_cls = engine_config.config_cls
        assert issubclass(engine_cls, Serializable)
        assert issubclass(engine_cls, Engine)

        # Deserialize and inject engine.
        engine_param: dict[str, Any] = {"engine": engine_cls.deserialize(engine_config, **kwargs["engine"])}
        return cls(**config.to_init_dict(cls, **(kwargs | engine_param)))

    @abc.abstractmethod
    def to_dataset(self, docs: Iterable[Doc]) -> datasets.Dataset:
        """Creates Hugging Face datasets.Dataset from docs.
        :param docs: Docs to convert.
        :return datasets.Dataset: Hugging Face dataset.
        """
