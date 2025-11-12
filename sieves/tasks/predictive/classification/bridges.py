"""Bridges for classification task."""

import abc
from collections import Counter
from collections.abc import Iterable
from functools import cached_property
from typing import Literal, TypeVar, override

import dspy
import jinja2
import pydantic

from sieves.data import Doc
from sieves.engines import (
    EngineInferenceMode,
    dspy_,
    huggingface_,
    langchain_,
    outlines_,
)
from sieves.engines.types import GenerationSettings
from sieves.tasks.predictive.bridges import Bridge

_BridgePromptSignature = TypeVar("_BridgePromptSignature")
_BridgeResult = TypeVar("_BridgeResult")


class ClassificationBridge(Bridge[_BridgePromptSignature, _BridgeResult, EngineInferenceMode], abc.ABC):
    """Abstract base class for classification bridges."""

    def __init__(
        self,
        task_id: str,
        prompt_instructions: str | None,
        labels: list[str],
        multi_label: bool,
        generation_settings: GenerationSettings,
        label_descriptions: dict[str, str] | None = None,
    ):
        """Initialize InformationExtractionBridge.

        :param task_id: Task ID.
        :param prompt_instructions: Custom prompt instructions. If None, default instructions are used.
        :param labels: Labels to classify.
        :param multi_label: If True, task returns confidence scores for all specified labels. If False, task returns
            most likely class label. In the latter case label forcing mechanisms are utilized, which can lead to higher
            accuracy.
        :param generation_settings: Generation settings.
        :param label_descriptions: Optional descriptions for each label.
        """
        super().__init__(
            task_id=task_id,
            prompt_instructions=prompt_instructions,
            overwrite=False,
            generation_settings=generation_settings,
        )
        self._labels = labels
        self._multi_label = multi_label
        self._label_descriptions = label_descriptions or {}

    def _get_label_descriptions(self) -> str:
        """Return a string with the label descriptions.

        :return: A string with the label descriptions.
        """
        labels_with_descriptions: list[str] = []
        for label in self._labels:
            if label in self._label_descriptions:
                labels_with_descriptions.append(
                    f"<label_description><label>{label}</label><description>"
                    f"{self._label_descriptions[label]}</description></label_description>"
                )
            else:
                labels_with_descriptions.append(label)

        crlf = "\n\t\t\t"
        label_desc_string = crlf + "\t" + (crlf + "\t").join(labels_with_descriptions)
        return f"{crlf}<label_descriptions>{label_desc_string}{crlf}</label_descriptions>\n\t\t"


class DSPyClassification(ClassificationBridge[dspy_.PromptSignature, dspy_.Result, dspy_.InferenceMode]):
    """DSPy bridge for classification."""

    @override
    @property
    def _default_prompt_instructions(self) -> str:
        if self._multi_label:
            return f"""
            Multi-label classification of the provided text given the labels {self._labels}.
            For each label, provide the confidence with which you believe that the provided text should be assigned
            this label. A confidence of 1.0 means that this text should absolutely be assigned this label. 0 means the
            opposite. Confidence per label should always be between 0 and 1. Confidence across lables does not have to
            add up to 1.

            {self._get_label_descriptions()}
            """

        return f"""
        Single-label classification of the provided text given the labels {self._labels}.
        Return the label that is the best fit for the provided text with the corresponding confidence.
        Exactly one label must be returned. Provide label as simple string, not as list.
        {self._get_label_descriptions()}
        """

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
        labels = self._labels
        LabelType = Literal[*labels]  # type: ignore[valid-type]

        if self._multi_label:

            class MultiLabelTextClassification(dspy.Signature):  # type: ignore[misc]
                text: str = dspy.InputField(description="Text to classify.")
                reasoning: str | None = dspy.OutputField(
                    default=None, description="Provide reasoning for complex or ambiguous classifications."
                )
                confidence_per_label: dict[LabelType, float] = dspy.OutputField(
                    description="Confidence per label that text should be classified with this label."
                )

            cls = MultiLabelTextClassification

        else:

            class SingleLabelTextClassification(dspy.Signature):  # type: ignore[misc]
                text: str = dspy.InputField(description="Text to classify.")
                label: LabelType = dspy.OutputField(
                    description="Correct label for the provided text. You MUST NOT provide a list for this attribute. "
                    "This a single label. Do not wrap this label in []."
                )
                reasoning: str | None = dspy.OutputField(
                    default=None, description="Provide reasoning for complex or ambiguous classifications."
                )
                confidence: float = dspy.OutputField(
                    description="Confidence that this label is correct as a float between 0 and 1."
                )

            cls = SingleLabelTextClassification

        cls.__doc__ = jinja2.Template(self._prompt_instructions).render()

        return cls

    @override
    @property
    def inference_mode(self) -> dspy_.InferenceMode:
        return self._generation_settings.inference_mode or dspy_.InferenceMode.predict

    @override
    def integrate(self, results: Iterable[dspy_.Result], docs: Iterable[Doc]) -> Iterable[Doc]:
        for doc, result in zip(docs, results):
            assert len(result.completions.confidence_per_label) == 1
            sorted_preds = sorted(
                ((label, score) for label, score in result.completions.confidence_per_label[0].items()),
                key=lambda x: x[1],
                reverse=True,
            )
            doc.results[self._task_id] = sorted_preds

            if not self._multi_label:
                if isinstance(sorted_preds, list) and len(sorted_preds) > 0:
                    doc.results[self._task_id] = sorted_preds[0]

        return docs

    @override
    def consolidate(
        self, results: Iterable[dspy_.Result], docs_offsets: list[tuple[int, int]]
    ) -> Iterable[dspy_.Result]:
        results = list(results)

        # Determine label scores for chunks per document.
        for doc_offset in docs_offsets:
            label_scores: dict[str, float] = {label: 0.0 for label in self._labels}
            doc_results = results[doc_offset[0] : doc_offset[1]]

            for res in doc_results:
                if res is None:
                    continue

                # Clamp score to range between 0 and 1. Alternatively we could force this in the prompt signature,
                # but this fails occasionally with some models and feels too strict.
                if self._multi_label:
                    for label, score in res.confidence_per_label.items():
                        label_scores[label] += max(0, min(score, 1))
                else:
                    label_scores[res.label] += max(0, min(res.confidence, 1))

            sorted_label_scores: list[dict[str, str | float]] = sorted(
                (
                    {"label": label, "score": score / (doc_offset[1] - doc_offset[0])}
                    for label, score in label_scores.items()
                ),
                key=lambda x: x["score"],
                reverse=True,
            )

            yield dspy.Prediction.from_completions(
                {
                    "confidence_per_label": [{sls["label"]: sls["score"] for sls in sorted_label_scores}],
                    "reasoning": [str([getattr(res, "reasoning", "") for res in doc_results])],
                },
                signature=self.prompt_signature,
            )


class HuggingFaceClassification(ClassificationBridge[list[str], huggingface_.Result, huggingface_.InferenceMode]):
    """HuggingFace bridge for classification."""

    @override
    @property
    def _default_prompt_instructions(self) -> str:
        return f"""
        This text is about {{}}.
        {self._get_label_descriptions()}
        """

    @override
    @property
    def _prompt_example_template(self) -> str | None:
        if self._multi_label:
            return """
            {% if examples|length > 0 -%}

                Examples:
                <examples>
                {%- for example in examples %}
                    <example>
                        <text>{{ example.text }}</text>
                        <output>
                            <reasoning>{{ example.reasoning }}</reasoning>
                            {%- for l, s in example.confidence_per_label.items() %}
                            <label_score>
                                <label>{{ l }}</label><
                                score>{{ s }}</score>
                            </label_score>{% endfor %}
                        </output>
                    </example>
                {% endfor %}</examples>
            {% endif %}
            """

        return """
        {% if examples|length > 0 -%}

        Examples:
        <examples>
        {%- for example in examples %}
            <example>
                <text>{{ example.text }}</text>
                <reasoning>{{ example.reasoning }}</reasoning>
                <output>
                    <label>{{ example.label }}</label><score>{{ example.confidence }}</score>
                </output>
            </example>
        {% endfor -%}
        </examples>
        {% endif -%}
        """

    @override
    @property
    def _prompt_conclusion(self) -> str | None:
        return None

    @override
    @property
    def prompt_signature(self) -> list[str]:
        return self._labels

    @override
    @property
    def inference_mode(self) -> huggingface_.InferenceMode:
        return self._generation_settings.inference_mode or huggingface_.InferenceMode.zeroshot_cls

    @override
    def integrate(self, results: Iterable[huggingface_.Result], docs: Iterable[Doc]) -> Iterable[Doc]:
        for doc, result in zip(docs, results):
            doc.results[self._task_id] = [(label, score) for label, score in zip(result["labels"], result["scores"])]

            if not self._multi_label:
                if isinstance(doc.results[self._task_id], list) and len(doc.results[self._task_id]) > 0:
                    doc.results[self._task_id] = doc.results[self._task_id][0]
        return docs

    @override
    def consolidate(
        self, results: Iterable[huggingface_.Result], docs_offsets: list[tuple[int, int]]
    ) -> Iterable[huggingface_.Result]:
        results = list(results)

        # Determine label scores for chunks per document.
        for doc_offset in docs_offsets:
            label_scores: dict[str, float] = {label: 0.0 for label in self._labels}

            for res in results[doc_offset[0] : doc_offset[1]]:
                for label, score in zip(res["labels"], res["scores"]):
                    assert isinstance(label, str)
                    assert isinstance(score, float)
                    label_scores[label] += score

            # Average score, sort by it in descending order.
            sorted_label_scores: list[dict[str, str | float]] = sorted(
                (
                    {"label": label, "score": score / (doc_offset[1] - doc_offset[0])}
                    for label, score in label_scores.items()
                ),
                key=lambda x: x["score"],
                reverse=True,
            )
            yield {
                "labels": [rec["label"] for rec in sorted_label_scores],  # type: ignore[dict-item]
                "scores": [rec["score"] for rec in sorted_label_scores],  # type: ignore[dict-item]
            }


class PydanticBasedClassification(
    ClassificationBridge[pydantic.BaseModel | list[str], pydantic.BaseModel | str, EngineInferenceMode], abc.ABC
):
    """Base class for Pydantic-based classification bridges."""

    @override
    @property
    def _default_prompt_instructions(self) -> str:
        if self._multi_label:
            return (
                f"""
            Perform multi-label classification of the provided text given the provided labels: {",".join(self._labels)}.
            {self._get_label_descriptions()}"""
                + """
            For each label, provide the confidence with which you believe that the provided text should be assigned
            this label. A confidence of 1.0 means that this text should absolutely be assigned this label. 0 means the
            opposite. Confidence per label should ALWAYS be between 0 and 1. Provide the reasoning for your decision.

            The output for two labels LABEL_1 and LABEL_2 should look like this:
            <output>
                <reasoning>REASONING</reasoning>
                <label_score><label>LABEL_1</label><score>CONFIDENCE_SCORE_1</score></label_score>
                <label_score><label>LABEL_2</label><score>CONFIDENCE_SCORE_2</score></label_score>
            </output>
            """
            )

        return f"""
        Classify the provided text. Your classification match one of these labels: {",".join(self._labels)}.
        {self._get_label_descriptions()}
        Also provide a confidence score reflecting how likely it is that your chosen label is the correct
        fit for the text.

        The output for two labels LABEL_1 and LABEL_2 should look like this:
        <output>
            <reasoning>REASONING</reasoning>
            <label>LABEL_1</label>
            <score>CONFIDENCE_SCORE_1</score>
        </output>
        """

    @override
    @property
    def _prompt_example_template(self) -> str | None:
        if self._multi_label:
            return """
            {% if examples|length > 0 -%}
                Examples:
                <examples>
                {%- for example in examples %}
                    <example>
                        <text>{{ example.text }}</text>
                        <output>
                            <reasoning>{{ example.reasoning }}</reasoning>
                            {%- for l, s in example.confidence_per_label.items() %}
                            <label_score><label>{{ l }}</label><score>{{ s }}</score></label_score>{% endfor %}
                        </output>
                    </example>
                {% endfor %}</examples>
            {% endif %}
            """

        return """
        {% if examples|length > 0 -%}
            Examples:
            <examples>
            {%- for example in examples %}
                <example>
                    <text>{{ example.text }}</text>
                    <output>
                        <reasoning>{{ example.reasoning }}</reasoning>
                        <label>{{ example.label }}</label>
                        <score>{{ example.confidence }}</score>
                    </output>
                </example>
            {% endfor %}</examples>
        {% endif %}
        """

    @override
    @property
    def _prompt_conclusion(self) -> str | None:
        return """
        ========

        <text>{{ text }}</text>
        <output>
        """

    @override
    @cached_property
    def prompt_signature(self) -> type[pydantic.BaseModel] | list[str]:
        if self._multi_label:
            prompt_sig = pydantic.create_model(  # type: ignore[no-matching-overload]
                "MultilabelClassification",
                __base__=pydantic.BaseModel,
                __doc__="Result of multi-label classification.",
                reasoning=(
                    str | None,
                    pydantic.Field(
                        default=None, description="Provide reasoning for complex or ambiguous classifications."
                    ),
                ),
                **{label: (float, ...) for label in self._labels},
            )
        else:
            labels = self._labels
            LabelType = Literal[*labels]  # type: ignore[valid-type]

            class SingleLabelClassification(pydantic.BaseModel):
                """Result of single-label classification."""

                reasoning: str | None = pydantic.Field(
                    default=None, description="Provide reasoning for complex or ambiguous classifications."
                )
                label: LabelType
                score: float

            prompt_sig = SingleLabelClassification

        assert isinstance(prompt_sig, type) and issubclass(prompt_sig, pydantic.BaseModel)
        return prompt_sig

    @override
    def integrate(self, results: Iterable[pydantic.BaseModel | str], docs: Iterable[Doc]) -> Iterable[Doc]:
        for doc, result in zip(docs, results):
            if self._multi_label:
                assert isinstance(result, pydantic.BaseModel)
                label_scores = {k: v for k, v in result.model_dump().items() if k != "reasoning"}
                doc.results[self._task_id] = sorted(
                    ((label, score) for label, score in label_scores.items()), key=lambda x: x[1], reverse=True
                )
            else:
                assert hasattr(result, "label") and hasattr(result, "score")
                doc.results[self._task_id] = (result.label, result.score)

        return docs

    @override
    def consolidate(
        self, results: Iterable[pydantic.BaseModel | str], docs_offsets: list[tuple[int, int]]
    ) -> Iterable[pydantic.BaseModel | str]:
        results = list(results)

        # Determine label scores for chunks per document.
        reasonings: list[str] = []
        for doc_offset in docs_offsets:
            label_scores: dict[str, float] = {label: 0.0 for label in self._labels}
            doc_results = results[doc_offset[0] : doc_offset[1]]

            for res in doc_results:
                if res is None:
                    continue  # type: ignore[unreachable]

                assert hasattr(res, "reasoning")
                reasonings.append(res.reasoning or "")
                # We clamp the score to 0 <= x <= 1. Alternatively we could force this in the prompt signature, but
                # this fails occasionally with some models and feels too strict.
                if self._multi_label:
                    for label in self._labels:
                        label_scores[label] += max(0, min(getattr(res, label), 1))
                else:
                    label_scores[getattr(res, "label")] += max(0, min(getattr(res, "score"), 1))

            avg_label_scores = {label: score / (doc_offset[1] - doc_offset[0]) for label, score in label_scores.items()}
            prompt_signature = self.prompt_signature
            assert issubclass(prompt_signature, pydantic.BaseModel)  # type: ignore[arg-type]
            assert callable(prompt_signature)

            if self._multi_label:
                yield prompt_signature(reasoning=str(reasonings), **avg_label_scores)
            else:
                max_score_label = max(avg_label_scores, key=avg_label_scores.__getitem__)
                yield prompt_signature(
                    reasoning=str(reasonings),
                    label=max_score_label,
                    score=avg_label_scores[max_score_label],
                )


class LangChainClassification(PydanticBasedClassification[langchain_.InferenceMode]):
    """LangChain bridge for classification."""

    @override
    @property
    def inference_mode(self) -> langchain_.InferenceMode:
        return self._generation_settings.inference_mode or langchain_.InferenceMode.structured


class PydanticBasedClassificationWithLabelForcing(PydanticBasedClassification[EngineInferenceMode], abc.ABC):
    """Base class for Pydantic-based classification bridges with label forcing."""

    @override
    @cached_property
    def prompt_signature(self) -> type[pydantic.BaseModel] | list[str]:
        return super().prompt_signature if self._multi_label else self._labels

    @override
    @property
    def _default_prompt_instructions(self) -> str:
        if self._multi_label:
            return super()._default_prompt_instructions

        return f"""
        Perform single-label classification of the provided text given the provided labels: {",".join(self._labels)}.
        {self._get_label_descriptions()}

        Provide the best-fitting label for given text.

        The output for two labels LABEL_1 and LABEL_2 should look like this:
        <output>
            <reasoning>REASONING</reasoning>
            <label>LABEL_1</label>
        </output>
        """

    @override
    @property
    def _prompt_example_template(self) -> str | None:
        if self._multi_label:
            return super()._prompt_example_template

        return """
        {% if examples|length > 0 -%}
            Examples:
            <examples>
            {%- for example in examples %}
                <example>
                    <text>{{ example.text }}</text>
                    <output>
                        <reasoning>{{ example.reasoning }}</reasoning>
                        <label>{{ example.label }}</label>
                    </output>
                </example>
            {% endfor %}</examples>
        {% endif %}
        """

    @override
    def integrate(self, results: Iterable[pydantic.BaseModel | str], docs: Iterable[Doc]) -> Iterable[Doc]:
        if self._multi_label:
            return super().integrate(results, docs)

        for doc, result in zip(docs, results):
            doc.results[self._task_id] = result
        return docs

    @override
    def consolidate(
        self, results: Iterable[pydantic.BaseModel | str], docs_offsets: list[tuple[int, int]]
    ) -> Iterable[pydantic.BaseModel | str]:
        if self._multi_label:
            yield from super().consolidate(results, docs_offsets)

        else:
            # Determine label scores for chunks per document.
            results = list(results)
            for doc_offset in docs_offsets:
                doc_results = results[doc_offset[0] : doc_offset[1]]
                label_counts = Counter(doc_results)
                yield label_counts.most_common()[0][0]


class OutlinesClassification(PydanticBasedClassificationWithLabelForcing[outlines_.InferenceMode]):
    """Outlines bridge for classification."""

    @override
    @property
    def inference_mode(self) -> outlines_.InferenceMode:
        return self._generation_settings.inference_mode or (
            outlines_.InferenceMode.json if self._multi_label else outlines_.InferenceMode.choice
        )
