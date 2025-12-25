"""Bridges for question answering task."""

import abc
from collections.abc import Sequence
from functools import cached_property
from typing import Any, TypeVar, override

import dspy
import jinja2
import pydantic

from sieves.data import Doc
from sieves.model_wrappers import ModelWrapperInferenceMode, dspy_, langchain_, outlines_
from sieves.model_wrappers.types import ModelSettings
from sieves.tasks.predictive.bridges import Bridge
from sieves.tasks.predictive.schemas.question_answering import QuestionAnswer, Result

_BridgePromptSignature = TypeVar("_BridgePromptSignature")
_BridgeResult = TypeVar("_BridgeResult")


class QABridge(Bridge[_BridgePromptSignature, _BridgeResult, ModelWrapperInferenceMode], abc.ABC):
    """Abstract base class for question answering bridges."""

    def __init__(
        self,
        task_id: str,
        prompt_instructions: str | None,
        questions: list[str],
        model_settings: ModelSettings,
    ):
        """Initialize QuestionAnsweringBridge.

        :param task_id: Task ID.
        :param prompt_instructions: Custom prompt instructions. If None, default instructions are used.
        :param questions: Questions to answer.
        :param model_settings: Model settings including inference_mode.
        """
        super().__init__(
            task_id=task_id,
            prompt_instructions=prompt_instructions,
            overwrite=False,
            model_settings=model_settings,
        )
        self._questions = questions

    @override
    def extract(self, docs: Sequence[Doc]) -> Sequence[dict[str, Any]]:
        return [{"text": doc.text if doc.text else None, "questions": self._questions} for doc in docs]


class DSPyQA(QABridge[dspy_.PromptSignature, dspy_.Result, dspy_.InferenceMode]):
    """DSPy bridge for question answering."""

    @override
    @property
    def _default_prompt_instructions(self) -> str:
        return """Multi-question answering. Also provide a confidence score between 0.0 and 1.0 for each answer."""

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
        class QuestionAnswering(dspy.Signature):  # type: ignore[misc]
            text: str = dspy.InputField(description="Text to use for question answering.")
            questions: list[str] = dspy.InputField(description="Questions to answer based on the text.")
            qa_pairs: list[QuestionAnswer] = dspy.OutputField(
                description="List of question-answer pairs, including confidence scores."
            )

        QuestionAnswering.__doc__ = jinja2.Template(self._prompt_instructions).render()

        return QuestionAnswering

    @override
    @property
    def inference_mode(self) -> dspy_.InferenceMode:
        return self._model_settings.inference_mode or dspy_.InferenceMode.predict

    @override
    def integrate(self, results: Sequence[dspy_.Result], docs: list[Doc]) -> list[Doc]:
        for doc, result in zip(docs, results):
            assert len(result.completions.qa_pairs) == 1
            doc.results[self._task_id] = Result(qa_pairs=result.qa_pairs)
        return docs

    @override
    def consolidate(
        self, results: Sequence[dspy_.Result], docs_offsets: list[tuple[int, int]]
    ) -> Sequence[dspy_.Result]:
        # Determine label scores for chunks per document.
        consolidated_results: list[dspy_.Result] = []
        for doc_offset in docs_offsets:
            # Map question -> (list of answers, list of scores).
            qa_map: dict[str, tuple[list[str], list[float]]] = {q: ([], []) for q in self._questions}
            doc_results = results[doc_offset[0] : doc_offset[1]]

            for res in doc_results:
                if res is None:
                    continue

                for qa in res.qa_pairs:
                    if qa.question in qa_map:
                        qa_map[qa.question][0].append(qa.answer)
                        if qa.score is not None:
                            qa_map[qa.question][1].append(qa.score)

            # Reconstruct `QuestionAnswer` objects.
            consolidated_qa_pairs: list[QuestionAnswer] = []
            for question in self._questions:
                answers, scores = qa_map[question]
                consolidated_qa_pairs.append(
                    QuestionAnswer(
                        question=question,
                        answer=" ".join(answers).strip(),
                        score=sum(scores) / len(scores) if scores else None,
                    )
                )

            consolidated_results.append(
                dspy.Prediction.from_completions(
                    {"qa_pairs": [consolidated_qa_pairs]},
                    signature=self.prompt_signature,
                )
            )
        return consolidated_results


class PydanticBasedQA(QABridge[pydantic.BaseModel, pydantic.BaseModel, ModelWrapperInferenceMode], abc.ABC):
    """Base class for Pydantic-based question answering bridges."""

    @override
    @property
    def _default_prompt_instructions(self) -> str:
        return """
        Use the given text to answer the following questions. Ensure you answer each question exactly once. Prefix each
        question with the number of the corresponding question. Also provide a confidence score between 0.0 and 1.0
        for each answer.
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
                        <qa_pairs>
                        {% for i in range(example.questions|length) %}
                            <qa_pair>
                                <question>{{ example.questions[i] }}</question>
                                <answer>{{ example.answers[i] }}</answer>
                                <score>{{ example.scores[i] if example.scores else "" }}</score>
                            </qa_pair>
                        {% endfor -%}
                        </qa_pairs>
                    </output>
                </example>
            {% endfor %}
            </examples>
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
        class QuestionAnswering(pydantic.BaseModel, frozen=True):
            """Question answering output."""

            qa_pairs: list[QuestionAnswer]

        return QuestionAnswering

    @override
    def integrate(self, results: Sequence[pydantic.BaseModel], docs: list[Doc]) -> list[Doc]:
        for doc, result in zip(docs, results):
            assert hasattr(result, "qa_pairs")
            doc.results[self._task_id] = Result(qa_pairs=result.qa_pairs)
        return docs

    @override
    def consolidate(
        self, results: Sequence[pydantic.BaseModel], docs_offsets: list[tuple[int, int]]
    ) -> Sequence[pydantic.BaseModel]:
        # Determine label scores for chunks per document.
        consolidated_results: list[pydantic.BaseModel] = []
        for doc_offset in docs_offsets:
            # Map question -> (list of answers, list of scores).
            qa_map: dict[str, tuple[list[str], list[float]]] = {q: ([], []) for q in self._questions}
            doc_results = results[doc_offset[0] : doc_offset[1]]

            for rec in doc_results:
                if rec is None:
                    continue  # type: ignore[unreachable]

                assert hasattr(rec, "qa_pairs")
                for qa in rec.qa_pairs:
                    if qa.question in qa_map:
                        qa_map[qa.question][0].append(qa.answer)
                        if qa.score is not None:
                            qa_map[qa.question][1].append(qa.score)

            # Reconstruct `QuestionAnswer` objects.
            consolidated_qa_pairs: list[QuestionAnswer] = []
            for question in self._questions:
                answers, scores = qa_map[question]
                consolidated_qa_pairs.append(
                    QuestionAnswer(
                        question=question,
                        answer=" ".join(answers).strip(),
                        score=sum(scores) / len(scores) if scores else None,
                    )
                )

            consolidated_results.append(self.prompt_signature(qa_pairs=consolidated_qa_pairs))
        return consolidated_results


class OutlinesQA(PydanticBasedQA[outlines_.InferenceMode]):
    """Outlines bridge for question answering."""

    @override
    @property
    def inference_mode(self) -> outlines_.InferenceMode:
        return self._model_settings.inference_mode or outlines_.InferenceMode.json


class LangChainQA(PydanticBasedQA[langchain_.InferenceMode]):
    """LangChain bridge for question answering."""

    @override
    @property
    def inference_mode(self) -> langchain_.InferenceMode:
        return self._model_settings.inference_mode or langchain_.InferenceMode.structured
