import abc
from collections.abc import Iterable
from functools import cached_property
from typing import Any, TypeVar

import dspy
import jinja2
import pydantic

from sieves.data import Doc
from sieves.engines import EngineInferenceMode, dspy_, instructor_, langchain_, ollama_, outlines_, vllm_
from sieves.tasks.predictive.bridges import Bridge

_BridgePromptSignature = TypeVar("_BridgePromptSignature")
_BridgeResult = TypeVar("_BridgeResult")


class QABridge(Bridge[_BridgePromptSignature, _BridgeResult, EngineInferenceMode], abc.ABC):
    def __init__(
        self, task_id: str, prompt_template: str | None, prompt_signature_desc: str | None, questions: list[str]
    ):
        """
        Initializes QuestionAnsweringBridge.
        :param task_id: Task ID.
        :param prompt_template: Custom prompt template.
        :param prompt_signature_desc: Custom prompt signature description.
        :param questions: Questions to answer.
        """
        super().__init__(
            task_id=task_id,
            prompt_template=prompt_template,
            prompt_signature_desc=prompt_signature_desc,
            overwrite=False,
        )
        self._questions = questions

    def extract(self, docs: Iterable[Doc]) -> Iterable[dict[str, Any]]:
        return ({"text": doc.text if doc.text else None, "questions": self._questions} for doc in docs)


class DSPyQA(QABridge[dspy_.PromptSignature, dspy_.Result, dspy_.InferenceMode]):
    @property
    def _prompt_template(self) -> str | None:
        return None

    @property
    def _prompt_signature_description(self) -> str | None:
        return """Multi-question answering."""

    @cached_property
    def prompt_signature(self) -> type[dspy_.PromptSignature]:
        n_questions = len(self._questions)

        class QuestionAnswering(dspy.Signature):  # type: ignore[misc]
            text: str = dspy.InputField(description="Text to use for question answering.")
            questions: tuple[str, ...] = dspy.InputField(
                description="Questions to answer based on the text.", min_length=n_questions, max_length=n_questions
            )
            answers: tuple[str, ...] = dspy.OutputField(
                description="Answers to questions, in the same sequence as the questions. Each answer corresponds to "
                "exactly one of the specified questions. Answer 1 answers question 1, answer 2 answers "
                "question 2 etc.",
                min_length=n_questions,
                max_length=n_questions,
            )

        QuestionAnswering.__doc__ = jinja2.Template(self.prompt_signature_description).render()

        return QuestionAnswering

    @property
    def inference_mode(self) -> dspy_.InferenceMode:
        return dspy_.InferenceMode.chain_of_thought

    def integrate(self, results: Iterable[dspy_.Result], docs: Iterable[Doc]) -> Iterable[Doc]:
        for doc, result in zip(docs, results):
            assert len(result.answers) == len(self._questions)
            doc.results[self._task_id] = result.answers
        return docs

    def consolidate(
        self, results: Iterable[dspy_.Result], docs_offsets: list[tuple[int, int]]
    ) -> Iterable[dspy_.Result]:
        results = list(results)

        # Merge all QAs.
        for doc_offset in docs_offsets:
            doc_results = results[doc_offset[0] : doc_offset[1]]
            answers: list[str] = [""] * len(self._questions)

            for res in doc_results:
                for i, answer in enumerate(res.answers):
                    answers[i] = f"{answers[i]} {answer}".strip()

            yield dspy.Prediction.from_completions({"answers": [answers]}, signature=self.prompt_signature)


class PydanticBasedQA(QABridge[pydantic.BaseModel, pydantic.BaseModel, EngineInferenceMode], abc.ABC):
    @property
    def _prompt_template(self) -> str | None:
        questions_block = "\n\t\t" + "\n\t\t".join(
            [f"<question>{i + 1}. {question}</question>" for i, question in enumerate(self._questions)]
        )
        return f"""
        Use the given text to answer the following questions. Ensure you answer each question exactly once. Prefix each 
        question with the number of the corresponding question. Provide a concise reasoning for your answers. 
        
        {{% if examples|length > 0 -%}}
            <examples>
            {{%- for example in examples %}}
                <example>
                    <text>"{{{{ example.text }}}}"</text>
                    <questions>
                    {{% for q in example.questions %}}    <question>{{{{ loop.index }}}}. {{{{ q }}}}</question>
                    {{% endfor -%}}
                    </questions>
                    <output>
                        <reasoning>{{{{ example.reasoning }}}}</reasoning>
                        <answers>
                        {{% for a in example.answers %}}  <answer>{{{{ loop.index }}}}. {{{{ a }}}}</answer>
                        {{% endfor -%}}
                        <answers>
                    </output>
                </example>
            {{% endfor %}}
            <examples>
        {{% endif -%}}

        ========
        <text>{{{{ text }}}}</text>
        <questions>{questions_block}</questions>
        <output>
        """

    @property
    def _prompt_signature_description(self) -> str | None:
        return None

    @cached_property
    def prompt_signature(self) -> type[pydantic.BaseModel]:
        prompt_sig = pydantic.create_model(
            "QuestionAnswering",
            __base__=pydantic.BaseModel,
            reasoning=(str, ...),
            answers=(pydantic.conlist(str, min_length=len(self._questions), max_length=len(self._questions)), ...),
        )

        if self.prompt_signature_description:
            prompt_sig.__doc__ = jinja2.Template(self.prompt_signature_description).render()

        assert isinstance(prompt_sig, type) and issubclass(prompt_sig, pydantic.BaseModel)
        return prompt_sig

    def integrate(self, results: Iterable[pydantic.BaseModel], docs: Iterable[Doc]) -> Iterable[Doc]:
        for doc, result in zip(docs, results):
            assert hasattr(result, "answers")
            doc.results[self._task_id] = result.answers
        return docs

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
                reasonings.append(rec.reasoning)
                for i, answer in enumerate(rec.answers):
                    answers[i] += answer + " "

            yield self.prompt_signature(reasoning=str(reasonings), answers=answers)


class OutlinesQA(PydanticBasedQA[outlines_.InferenceMode]):
    @property
    def inference_mode(self) -> outlines_.InferenceMode:
        return outlines_.InferenceMode.json


class OllamaQA(PydanticBasedQA[ollama_.InferenceMode]):
    @property
    def inference_mode(self) -> ollama_.InferenceMode:
        return ollama_.InferenceMode.structured


class LangChainQA(PydanticBasedQA[langchain_.InferenceMode]):
    @property
    def inference_mode(self) -> langchain_.InferenceMode:
        return langchain_.InferenceMode.structured


class InstructorQA(PydanticBasedQA[instructor_.InferenceMode]):
    @property
    def inference_mode(self) -> instructor_.InferenceMode:
        return instructor_.InferenceMode.structured


class VLLMQA(PydanticBasedQA[vllm_.InferenceMode]):
    @property
    def inference_mode(self) -> vllm_.InferenceMode:
        return vllm_.InferenceMode.json
