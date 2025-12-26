"""Bridges for classification task."""

import abc
from collections.abc import Callable, Sequence
from functools import cached_property
from typing import Any, Literal, TypeVar, override

import dspy
import jinja2
import pydantic

from sieves.data import Doc
from sieves.model_wrappers import (
    ModelWrapperInferenceMode,
    dspy_,
    huggingface_,
    langchain_,
    outlines_,
)
from sieves.model_wrappers.types import ModelSettings
from sieves.tasks.predictive.bridges import Bridge
from sieves.tasks.predictive.consolidation import LabelScoreConsolidation
from sieves.tasks.predictive.schemas.classification import ResultMultiLabel, ResultSingleLabel

_BridgePromptSignature = TypeVar("_BridgePromptSignature")
_BridgeResult = TypeVar("_BridgeResult")


class ClassificationBridge(Bridge[_BridgePromptSignature, _BridgeResult, ModelWrapperInferenceMode], abc.ABC):
    """Abstract base class for classification bridges."""

    def __init__(
        self,
        task_id: str,
        prompt_instructions: str | None,
        labels: list[str] | dict[str, str],
        mode: Literal["single", "multi"],
        model_settings: ModelSettings,
    ):
        """Initialize ClassificationBridge.

        :param task_id: Task ID.
        :param prompt_instructions: Custom prompt instructions. If None, default instructions are used.
        :param labels: Labels to classify. Can be a list of label strings, or a dict mapping labels to descriptions.
        :param mode: If 'multi'', task returns scores for all specified labels. If 'single', task returns
        most likely class label. In the latter case label forcing mechanisms are utilized, which can lead to higher
            accuracy.
        :param model_settings: Model settings.
        """
        super().__init__(
            task_id=task_id,
            prompt_instructions=prompt_instructions,
            overwrite=False,
            model_settings=model_settings,
        )
        if isinstance(labels, dict):
            self._labels = list(labels.keys())
            self._label_descriptions = labels
        else:
            self._labels = labels
            self._label_descriptions = {}
        self._mode = mode
        self._consolidation_strategy = LabelScoreConsolidation(
            labels=self._labels,
            mode=self._mode,
            extractor=self._get_extractor(),
        )

    @abc.abstractmethod
    def _get_extractor(self) -> Callable[[Any], dict[str, float]]:
        """Return a callable that extracts label scores from a raw chunk result.

        :return: Extractor callable.
        """

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
        if self._mode == "multi":
            return f"""
            Multi-label classification of the provided text given the labels {self._labels}.
            For each label, provide the score with which you believe that the provided text should be assigned
            this label. A score of 1.0 means that this text should absolutely be assigned this label. 0 means the
            opposite. Score per label should always be between 0 and 1. Score across lables does not have to
            add up to 1.

            {self._get_label_descriptions()}
            """

        return f"""
        Single-label classification of the provided text given the labels {self._labels}.
        Return the label that is the best fit for the provided text with the corresponding score.
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

        if self._mode == "multi":

            class MultiLabelTextClassification(dspy.Signature):  # type: ignore[misc]
                text: str = dspy.InputField(description="Text to classify.")
                score_per_label: dict[LabelType, float] = dspy.OutputField(
                    description="Score per label that text should be classified with this label."
                )

            cls = MultiLabelTextClassification

        else:

            class SingleLabelTextClassification(dspy.Signature):  # type: ignore[misc]
                text: str = dspy.InputField(description="Text to classify.")
                label: LabelType = dspy.OutputField(
                    description="Correct label for the provided text. You MUST NOT provide a list for this attribute. "
                    "This a single label. Do not wrap this label in []."
                )
                score: float = dspy.OutputField(
                    description="Score that this label is correct as a float between 0 and 1."
                )

            cls = SingleLabelTextClassification

        cls.__doc__ = jinja2.Template(self._prompt_instructions).render()

        return cls

    @override
    @property
    def inference_mode(self) -> dspy_.InferenceMode:
        return self._model_settings.inference_mode or dspy_.InferenceMode.predict

    @override
    def _get_extractor(self) -> Callable[[Any], dict[str, float]]:
        def extractor(res: Any) -> dict[str, float]:
            if self._mode == "multi":
                return dict(res.score_per_label)
            return {res.label: res.score}

        return extractor

    @override
    def integrate(self, results: Sequence[dspy_.Result], docs: list[Doc]) -> list[Doc]:
        for doc, result in zip(docs, results):
            assert len(result.completions.score_per_label) == 1
            sorted_preds = sorted(
                ((label, score) for label, score in result.completions.score_per_label[0].items()),
                key=lambda x: x[1],
                reverse=True,
            )

            if self._mode == "multi":
                doc.results[self._task_id] = ResultMultiLabel(label_scores=sorted_preds)
            else:
                if isinstance(sorted_preds, list) and len(sorted_preds) > 0:
                    doc.results[self._task_id] = ResultSingleLabel(label=sorted_preds[0][0], score=sorted_preds[0][1])

        return docs

    @override
    def consolidate(
        self, results: Sequence[dspy_.Result], docs_offsets: list[tuple[int, int]]
    ) -> Sequence[dspy_.Result]:
        consolidated_results_clean = self._consolidation_strategy.consolidate(results, docs_offsets)

        # Wrap back into dspy.Prediction.
        consolidated_results: list[dspy_.Result] = []
        for scores_list in consolidated_results_clean:
            consolidated_results.append(
                dspy.Prediction.from_completions(
                    {
                        "score_per_label": [{label: score for label, score in scores_list}],
                    },
                    signature=self.prompt_signature,
                )
            )
        return consolidated_results


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
        if self._mode == "multi":
            return """
            {% if examples|length > 0 -%}

                Examples:
                <examples>
                {%- for example in examples %}
                    <example>
                        <text>{{ example.text }}</text>
                        <output>
                            {%- for l, s in example.score_per_label.items() %}
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
                        <output>
                            <label>{{ example.label }}</label><score>{{ example.score }}</score>
                        </output>
                    </example>
                {% endfor %}</examples>
            {% endif %}
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
        return self._model_settings.inference_mode or huggingface_.InferenceMode.zeroshot_cls

    @override
    def _get_extractor(self) -> Callable[[Any], dict[str, float]]:
        return lambda res: dict(zip(res["labels"], res["scores"]))

    @override
    def integrate(self, results: Sequence[huggingface_.Result], docs: list[Doc]) -> list[Doc]:
        for doc, result in zip(docs, results):
            label_scores = [(label, score) for label, score in zip(result["labels"], result["scores"])]
            if self._mode == "multi":
                doc.results[self._task_id] = ResultMultiLabel(label_scores=label_scores)
            else:
                if len(label_scores) > 0:
                    doc.results[self._task_id] = ResultSingleLabel(label=label_scores[0][0], score=label_scores[0][1])
        return docs

    @override
    def consolidate(
        self, results: Sequence[huggingface_.Result], docs_offsets: list[tuple[int, int]]
    ) -> Sequence[huggingface_.Result]:
        consolidated_results_clean = self._consolidation_strategy.consolidate(results, docs_offsets)

        consolidated_results: list[huggingface_.Result] = []
        for scores_list in consolidated_results_clean:
            consolidated_results.append(
                {
                    "labels": [label for label, _ in scores_list],
                    "scores": [score for _, score in scores_list],
                }
            )
        return consolidated_results


class PydanticBasedClassification(
    ClassificationBridge[pydantic.BaseModel | list[str], pydantic.BaseModel | str, ModelWrapperInferenceMode], abc.ABC
):
    """Base class for Pydantic-based classification bridges."""

    @override
    @property
    def _default_prompt_instructions(self) -> str:
        if self._mode == "multi":
            return (
                f"""
            Perform multi-label classification of the provided text given the provided labels: {",".join(self._labels)}.
            {self._get_label_descriptions()}"""
                + """
            For each label, provide the score with which you believe that the provided text should be assigned
            this label. A score of 1.0 means that this text should absolutely be assigned this label. 0 means the
            opposite. Score per label should ALWAYS be between 0 and 1. Provide the reasoning for your decision.

            The output for two labels LABEL_1 and LABEL_2 should look like this:
            <output>
                <reasoning>REASONING</reasoning>
                <label_score><label>LABEL_1</label><score>SCORE_1</score></label_score>
                <label_score><label>LABEL_2</label><score>SCORE_2</score></label_score>
            </output>
            """
            )

        return f"""
        Classify the provided text. Your classification match one of these labels: {",".join(self._labels)}.
        {self._get_label_descriptions()}
        Also provide a score reflecting how likely it is that your chosen label is the correct
        fit for the text.

        The output for two labels LABEL_1 and LABEL_2 should look like this:
        <output>
            <reasoning>REASONING</reasoning>
            <label>LABEL_1</label>
            <score>SCORE_1</score>
        </output>
        """

    @override
    @property
    def _prompt_example_template(self) -> str | None:
        if self._mode == "multi":
            return """
            {% if examples|length > 0 -%}
                Examples:
                <examples>
                {%- for example in examples %}
                    <example>
                        <text>{{ example.text }}</text>
                        <output>
                            <reasoning>{{ example.reasoning }}</reasoning>
                            {%- for l, s in example.score_per_label.items() %}
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
                        <score>{{ example.score }}</score>
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
        if self._mode == "multi":
            prompt_sig = pydantic.create_model(  # type: ignore[no-matching-overload]
                "MultilabelClassification",
                __base__=pydantic.BaseModel,
                __doc__="Result of multi-label classification.",
                **{label: (float, ...) for label in self._labels},
            )
        else:
            labels = self._labels
            LabelType = Literal[*labels]  # type: ignore[valid-type]

            class SingleLabelClassification(pydantic.BaseModel):
                """Result of single-label classification."""

                label: LabelType
                score: float

            prompt_sig = SingleLabelClassification

        assert isinstance(prompt_sig, type) and issubclass(prompt_sig, pydantic.BaseModel)
        return prompt_sig

    @override
    def _get_extractor(self) -> Callable[[Any], dict[str, float]]:
        def extractor(res: Any) -> dict[str, float]:
            if self._mode == "multi":
                return {label: float(getattr(res, label)) for label in self._labels}
            return {str(getattr(res, "label")): float(getattr(res, "score"))}

        return extractor

    @override
    def integrate(self, results: Sequence[pydantic.BaseModel | str], docs: list[Doc]) -> list[Doc]:
        for doc, result in zip(docs, results):
            if self._mode == "multi":
                assert isinstance(result, pydantic.BaseModel)
                label_scores = result.model_dump()
                sorted_label_scores = sorted(
                    ((label, score) for label, score in label_scores.items()), key=lambda x: x[1], reverse=True
                )
                doc.results[self._task_id] = ResultMultiLabel(label_scores=sorted_label_scores)
            else:
                assert hasattr(result, "label") and hasattr(result, "score")
                doc.results[self._task_id] = ResultSingleLabel(label=result.label, score=result.score)

        return docs

    @override
    def consolidate(
        self, results: Sequence[pydantic.BaseModel | str], docs_offsets: list[tuple[int, int]]
    ) -> Sequence[pydantic.BaseModel | str]:
        consolidated_results_clean = self._consolidation_strategy.consolidate(results, docs_offsets)

        consolidated_results: list[pydantic.BaseModel | str] = []
        prompt_signature = self.prompt_signature
        assert issubclass(prompt_signature, pydantic.BaseModel)  # type: ignore[arg-type]

        for scores_list in consolidated_results_clean:
            if self._mode == "multi":
                consolidated_results.append(prompt_signature(**dict(scores_list)))
            else:
                # In single mode, we only take the top label.
                top_label, top_score = scores_list[0]
                consolidated_results.append(
                    prompt_signature(
                        label=top_label,
                        score=top_score,
                    )
                )
        return consolidated_results


class LangChainClassification(PydanticBasedClassification[langchain_.InferenceMode]):
    """LangChain bridge for classification."""

    @override
    @property
    def inference_mode(self) -> langchain_.InferenceMode:
        return self._model_settings.inference_mode or langchain_.InferenceMode.structured


class PydanticBasedClassificationWithLabelForcing(PydanticBasedClassification[ModelWrapperInferenceMode], abc.ABC):
    """Base class for Pydantic-based classification bridges with label forcing."""

    @override
    @cached_property
    def prompt_signature(self) -> type[pydantic.BaseModel] | list[str]:
        return super().prompt_signature if self._mode == "multi" else self._labels

    @override
    @property
    def _default_prompt_instructions(self) -> str:
        if self._mode == "multi":
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
        if self._mode == "multi":
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
    def _get_extractor(self) -> Callable[[Any], dict[str, float]]:
        if self._mode == "multi":
            return super()._get_extractor()

        def extractor(res: Any) -> dict[str, float]:
            if isinstance(res, str):
                return {res: 1.0}
            return {str(getattr(res, "label")): float(getattr(res, "score", 1.0))}

        return extractor

    @override
    def integrate(self, results: Sequence[pydantic.BaseModel | str], docs: list[Doc]) -> list[Doc]:
        if self._mode == "multi":
            return super().integrate(results, docs)

        for doc, result in zip(docs, results):
            # Outlines choice mode returns just the label string.
            if isinstance(result, str):
                doc.results[self._task_id] = ResultSingleLabel(label=result, score=1.0)
            else:
                # Fallback for other pydantic-based bridges.
                assert hasattr(result, "label")
                doc.results[self._task_id] = ResultSingleLabel(label=result.label, score=getattr(result, "score", 1.0))
        return docs

    @override
    def consolidate(
        self, results: Sequence[pydantic.BaseModel | str], docs_offsets: list[tuple[int, int]]
    ) -> Sequence[pydantic.BaseModel | str]:
        if self._mode == "multi":
            return super().consolidate(results, docs_offsets)

        consolidated_results_clean = self._consolidation_strategy.consolidate(results, docs_offsets)

        consolidated_results: list[pydantic.BaseModel | str] = []
        for scores_list in consolidated_results_clean:
            top_label, _ = scores_list[0]
            # If we are in choice mode (Outlines), we just return the string.
            # Otherwise we'd return a Pydantic model.
            # PydanticBasedClassificationWithLabelForcing.prompt_signature returns self._labels (list[str]) if single.
            # So the expected consolidated result type here is str.
            consolidated_results.append(top_label)

        return consolidated_results


class OutlinesClassification(PydanticBasedClassificationWithLabelForcing[outlines_.InferenceMode]):
    """Outlines bridge for classification."""

    @override
    @property
    def inference_mode(self) -> outlines_.InferenceMode:
        return self._model_settings.inference_mode or (
            outlines_.InferenceMode.json if self._mode == "multi" else outlines_.InferenceMode.choice
        )
