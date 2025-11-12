"""Bridges for question answering task."""

import abc
from collections.abc import Iterable
from functools import cached_property
from typing import Any, TypeVar, override

import dspy
import jinja2
import pydantic

from sieves.data import Doc
from sieves.engines import EngineInferenceMode, dspy_, langchain_, outlines_
from sieves.engines.types import GenerationSettings
from sieves.tasks.predictive.bridges import Bridge

_BridgePromptSignature = TypeVar("_BridgePromptSignature")
_BridgeResult = TypeVar("_BridgeResult")


class QABridge(Bridge[_BridgePromptSignature, _BridgeResult, EngineInferenceMode], abc.ABC):
    """Abstract base class for question answering bridges."""

    def __init__(
        self,
        task_id: str,
        prompt_instructions: str | None,
        questions: list[str],
        generation_settings: GenerationSettings,
    ):
        """Initialize QuestionAnsweringBridge.

        :param task_id: Task ID.
        :param prompt_instructions: Custom prompt instructions. If None, default instructions are used.
        :param questions: Questions to answer.
        :param generation_settings: Generation settings including inference_mode.
        """
        super().__init__(
            task_id=task_id,
            prompt_instructions=prompt_instructions,
            overwrite=False,
            generation_settings=generation_settings,
        )
        self._questions = questions

    @override
    def extract(self, docs: Iterable[Doc]) -> Iterable[dict[str, Any]]:
        return ({"text": doc.text if doc.text else None, "questions": self._questions} for doc in docs)


class DSPyQA(QABridge[dspy_.PromptSignature, dspy_.Result, dspy_.InferenceMode]):
    """DSPy bridge for question answering."""

    @override
    @property
    def _default_prompt_instructions(self) -> str:
        return """Multi-question answering."""

    @override
    @property
    def _prompt_example_template(self) -> str | None:
        return None

    @override
    @property
    def _prompt_conclusion(self) -> str | None:
        return None

    @override
    @cached_property
    def prompt_signature(self) -> type[dspy_.PromptSignature]:
        n_questions = len(self._questions)

        class QuestionAnswering(dspy.Signature):  # type: ignore[misc]
            text: str = dspy.InputField(description="Text to use for question answering.")
            questions: tuple[str, ...] = dspy.InputField(
                description="Questions to answer based on the text.", min_length=n_questions, max_length=n_questions
            )
            reasoning: str | None = dspy.OutputField(
                default=None, description="Provide reasoning for complex or ambiguous answers."
            )
            answers: tuple[str, ...] = dspy.OutputField(
                description="Answers to questions, in the same sequence as the questions. Each answer corresponds to "
                "exactly one of the specified questions. Answer 1 answers question 1, answer 2 answers "
                "question 2 etc.",
                min_length=n_questions,
                max_length=n_questions,
            )

        QuestionAnswering.__doc__ = jinja2.Template(self._prompt_instructions).render()

        return QuestionAnswering

    @override
    @property
    def inference_mode(self) -> dspy_.InferenceMode:
        return self._generation_settings.inference_mode or dspy_.InferenceMode.predict

    @override
    def integrate(self, results: Iterable[dspy_.Result], docs: Iterable[Doc]) -> Iterable[Doc]:
        for doc, result in zip(docs, results):
            assert len(result.answers) == len(self._questions)
            doc.results[self._task_id] = result.answers
        return docs

    @override
    def consolidate(
        self, results: Iterable[dspy_.Result], docs_offsets: list[tuple[int, int]]
    ) -> Iterable[dspy_.Result]:
        results = list(results)

        # Merge all QAs.
        for doc_offset in docs_offsets:
            doc_results = results[doc_offset[0] : doc_offset[1]]
            answers: list[str] = [""] * len(self._questions)

            for res in doc_results:
                if res is None:
                    continue

                for i, answer in enumerate(res.answers):
                    answers[i] = f"{answers[i]} {answer}".strip()

            yield dspy.Prediction.from_completions({"answers": [answers]}, signature=self.prompt_signature)


class PydanticBasedQA(QABridge[pydantic.BaseModel, pydantic.BaseModel, EngineInferenceMode], abc.ABC):
    """Base class for Pydantic-based question answering bridges."""

    @override
    @property
    def _default_prompt_instructions(self) -> str:
        return """
        Use the given text to answer the following questions. Ensure you answer each question exactly once. Prefix each
        question with the number of the corresponding question. Provide a concise reasoning for your answers.
        """

    @override
    @property
    def _prompt_example_template(self) -> str | None:
        return """
        {% if examples|length > 0 -%}
            <examples>
            {%- for example in examples %}
                <example>
                    <text>"{{ example.text }}"</text>
                    <questions>
                    {% for q in example.questions %}    <question>{{ loop.index }}. {{ q }}</question>
                    {% endfor -%}
                    </questions>
                    <output>
                        <reasoning>{{ example.reasoning }}</reasoning>
                        <answers>
                        {% for a in example.answers %}  <answer>{{ loop.index }}. {{ a }}</answer>
                        {% endfor -%}
                        <answers>
                    </output>
                </example>
            {% endfor %}
            <examples>
        {% endif -%}
        """

    @override
    @property
    def _prompt_conclusion(self) -> str | None:
        questions_block = "\n\t\t" + "\n\t\t".join(
            [f"<question>{i + 1}. {question}</question>" for i, question in enumerate(self._questions)]
        )

        return f"""
        ========
        <text>{{{{ text }}}}</text>
        <questions>{questions_block}</questions>
        <output>
        """

    @override
    @cached_property
    def prompt_signature(self) -> type[pydantic.BaseModel]:
        prompt_sig = pydantic.create_model(
            "QuestionAnswering",
            __base__=pydantic.BaseModel,
            __doc__="Question answering of specified text.",
            reasoning=(
                str | None,
                pydantic.Field(default=None, description="Provide reasoning for complex or ambiguous answers."),
            ),
            answers=(pydantic.conlist(str, min_length=len(self._questions), max_length=len(self._questions)), ...),
        )

        assert isinstance(prompt_sig, type) and issubclass(prompt_sig, pydantic.BaseModel)
        return prompt_sig

    @override
    def integrate(self, results: Iterable[pydantic.BaseModel], docs: Iterable[Doc]) -> Iterable[Doc]:
        for doc, result in zip(docs, results):
            assert hasattr(result, "answers")
            doc.results[self._task_id] = result.answers
        return docs

    @override
    def consolidate(
        self, results: Iterable[pydantic.BaseModel], docs_offsets: list[tuple[int, int]]
    ) -> Iterable[pydantic.BaseModel]:
        results = list(results)

        # Determine label scores for chunks per document.
        for doc_offset in docs_offsets:
            doc_results = results[doc_offset[0] : doc_offset[1]]
            answers: list[str] = [""] * len(self._questions)
            reasonings: list[str] = []

            for rec in doc_results:
                if rec is None:
                    continue  # type: ignore[unreachable]

                assert hasattr(rec, "reasoning")
                assert hasattr(rec, "answers")
                reasonings.append(rec.reasoning or "")
                for i, answer in enumerate(rec.answers):
                    answers[i] += answer + " "

            yield self.prompt_signature(reasoning=str(reasonings), answers=answers)


class OutlinesQA(PydanticBasedQA[outlines_.InferenceMode]):
    """Outlines bridge for question answering."""

    @override
    @property
    def inference_mode(self) -> outlines_.InferenceMode:
        return self._generation_settings.inference_mode or outlines_.InferenceMode.json


class LangChainQA(PydanticBasedQA[langchain_.InferenceMode]):
    """LangChain bridge for question answering."""

    @override
    @property
    def inference_mode(self) -> langchain_.InferenceMode:
        return self._generation_settings.inference_mode or langchain_.InferenceMode.structured
