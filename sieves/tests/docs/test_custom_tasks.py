"""
Test file containing examples for the Custom Tasks guide.

These code blocks are referenced in docs/guides/custom_tasks.md using snippet injection.
"""

from __future__ import annotations

from typing import Sequence

import pytest


def test_basic_custom_task():
    """Test basic custom task example."""
    # --8<-- [start:custom-task-basic]
    from typing import Iterable
    from sieves.tasks.core import Task
    from sieves.data import Doc

    class CharCountTask(Task):
        def _call(self, docs: Iterable[Doc]) -> Iterable[Doc]:
            """Counts characters in doc.text.

            :param docs: Documents to process.
            :return Iterable[Doc]: Processed documents.
            """
            for doc in docs:
                doc.results[self.id] = len(doc.text)
                yield doc
    # --8<-- [end:custom-task-basic]

    # Test the task
    task = CharCountTask(task_id="CharCount", include_meta=False, batch_size=-1)
    docs = [Doc(text="Hello"), Doc(text="World!")]
    results = list(task(docs))
    assert results[0].results[task.id] == 5
    assert results[1].results[task.id] == 6


def test_custom_bridge_example():
    """Demonstrates custom bridge implementation (for docs only)."""
    # --8<-- [start:custom-bridge-sentiment]
    # --8<-- [start:custom-bridge-sentiment-imports]
    from collections.abc import Iterable
    from functools import cached_property

    import pydantic

    from sieves.data import Doc
    from sieves.model_wrappers import ModelWrapperInferenceMode, outlines_
    from sieves.tasks.predictive.bridges import Bridge
    # --8<-- [end:custom-bridge-sentiment-imports]


    # --8<-- [start:custom-bridge-sentiment-schema]
    # This is how we require our response to look like - we require not just the score, but also a reasoning/justification
    # for why this model assigns this score. We also force the score to be between 0 and 1.
    class SentimentEstimate(pydantic.BaseModel):
       reasoning: str
       score: pydantic.confloat(ge=0, le=1)
    # --8<-- [end:custom-bridge-sentiment-schema]


    # --8<-- [start:custom-bridge-sentiment-class-def]
    # This is the bridge class.
    class OutlinesSentimentAnalysis(Bridge[SentimentEstimate, SentimentEstimate, outlines_.InferenceMode]):
    # --8<-- [end:custom-bridge-sentiment-class-def]
        # --8<-- [start:custom-bridge-sentiment-prompt]
        # This defines the default prompt template as Jinja2 template string.
        # We include an example block allowing us to include fewshot examples.
        @property
        def _default_prompt_instructions(self) -> str:
            return """
            Estimate the sentiment in this text as a float between 0 and 1. 0 is negative, 1 is positive. Provide your
            reasoning for why you estimate this score before you output the score.

            {% if examples|length > 0 -%}
                Examples:
                ----------
                {%- for example in examples %}
                    Text: "{{ example.text }}":
                    Output:
                        Reasoning: "{{ example.reasoning }}":
                        Sentiment: "{{ example.sentiment }}"
                {% endfor -%}
                ----------
            {% endif -%}

            ========
            Text: {{ text }}
            Output:
            """
        # --8<-- [end:custom-bridge-sentiment-prompt]

        # --8<-- [start:custom-bridge-sentiment-properties]
        @property
        def _prompt_example_template(self) -> str | None:
            return None

        @property
        def _prompt_conclusion(self) -> str | None:
            return None

        @property
        def inference_mode(self) -> outlines_.InferenceMode:
            return self._model_settings.inference_mode or outlines_.InferenceMode.json

        # We return our SentimentEstimate as prompt signature.
        @cached_property
        def prompt_signature(self) -> type[pydantic.BaseModel]:
            return SentimentEstimate
        # --8<-- [end:custom-bridge-sentiment-properties]

        # --8<-- [start:custom-bridge-sentiment-integrate]
        # We copy the result score into our doc's results attribute.
        def integrate(self, results: Sequence[SentimentEstimate], docs: list[Doc]) -> list[Doc]:
            """Integrate results into Doc instances."""
            for doc, result in zip(docs, results):
                assert isinstance(result, SentimentEstimate)
                # doc.results is a dict, with the task ID being the key to store our results under for the corresponding
                # task.
                doc.results[self._task_id] = result.score
            return docs
        # --8<-- [end:custom-bridge-sentiment-integrate]

        # --8<-- [start:custom-bridge-sentiment-consolidate]
        # Consolidating multiple chunks for sentiment analysis: we compute the average score over
        # all chunks and assume this to be the sentiment score for the entire document.
        def consolidate(
            self, results: Sequence[SentimentEstimate], docs_offsets: list[tuple[int, int]]
        ) -> Sequence[SentimentEstimate]:
            """Consolidate results for document chunks into document results."""
            consolidated_results: list[SentimentEstimate] = []

            # docs_offsets contains (start, end) tuples indicating which result indices belong to which document.
            # Example: [(0, 3), (3, 5)] means doc1 uses results[0:3], doc2 uses results[3:5]
            # This mapping is necessary because long documents are split into multiple chunks for processing.
            for doc_offset in docs_offsets:
                # Accumulate reasonings and scores from all chunks of this document
                reasonings: list[str] = []
                scores = 0.

                # Process each chunk's result for this document
                for chunk_result in results[doc_offset[0] : doc_offset[1]]:
                    # Model wrappers may return None results if they encounter errors in permissive mode.
                    # Skip None results to avoid crashes while still processing valid chunks.
                    if chunk_result:
                        assert isinstance(chunk_result, SentimentEstimate)
                        reasonings.append(chunk_result.reasoning)
                        scores += chunk_result.score

                # Calculate how many chunks this document has
                num_chunks = doc_offset[1] - doc_offset[0]

                consolidated_results.append(SentimentEstimate(
                   # Average the sentiment score across all chunks of this document
                   score=scores / num_chunks,
                   # Concatenate all chunk reasonings into a single string for the document
                   # (in production, you might want more sophisticated reasoning aggregation)
                   reasoning=str(reasonings)
                ))
            return consolidated_results
        # --8<-- [end:custom-bridge-sentiment-consolidate]
    # --8<-- [end:custom-bridge-sentiment]


def test_custom_predictive_task_example():
    """Demonstrates custom predictive task wrapper (for docs only)."""
    # --8<-- [start:custom-task-predictive]
    # --8<-- [start:custom-task-predictive-imports]
    from collections.abc import Iterable
    from typing import Any

    import datasets
    import pydantic

    from sieves.data import Doc
    from sieves.model_wrappers import ModelType
    from sieves.serialization import Config
    from sieves.tasks.predictive.core import PredictiveTask
    # --8<-- [end:custom-task-predictive-imports]

    # --8<-- [start:custom-task-predictive-schema]
    # Define the output schema for sentiment estimation
    class SentimentEstimate(pydantic.BaseModel):
       reasoning: str
       score: pydantic.confloat(ge=0, le=1)
    # --8<-- [end:custom-task-predictive-schema]

    # --8<-- [start:custom-task-predictive-bridge-imports]
    # Full bridge implementation (self-contained)
    from functools import cached_property
    from sieves.model_wrappers import ModelWrapperInferenceMode, outlines_
    from sieves.tasks.predictive.bridges import Bridge
    # --8<-- [end:custom-task-predictive-bridge-imports]

    # --8<-- [start:custom-task-predictive-bridge-class]
    class OutlinesSentimentAnalysis(Bridge[SentimentEstimate, SentimentEstimate, outlines_.InferenceMode]):
    # --8<-- [end:custom-task-predictive-bridge-class]
        # --8<-- [start:custom-task-predictive-bridge-prompt]
        # This defines the default prompt template as Jinja2 template string.
        # We include an example block allowing us to include fewshot examples.
        @property
        def _default_prompt_instructions(self) -> str:
            return """
            Estimate the sentiment in this text as a float between 0 and 1. 0 is negative, 1 is positive. Provide your
            reasoning for why you estimate this score before you output the score.

            {% if examples|length > 0 -%}
                Examples:
                ----------
                {%- for example in examples %}
                    Text: "{{ example.text }}":
                    Output:
                        Reasoning: "{{ example.reasoning }}":
                        Sentiment: "{{ example.sentiment }}"
                {% endfor -%}
                ----------
            {% endif -%}

            ========
            Text: {{ text }}
            Output:
            """
        # --8<-- [end:custom-task-predictive-bridge-prompt]

        # --8<-- [start:custom-task-predictive-bridge-properties]
        @property
        def _prompt_example_template(self) -> str | None:
            return None

        @property
        def _prompt_conclusion(self) -> str | None:
            return None

        @property
        def inference_mode(self) -> outlines_.InferenceMode:
            return self._model_settings.inference_mode or outlines_.InferenceMode.json

        # We return our SentimentEstimate as prompt signature.
        @cached_property
        def prompt_signature(self) -> type[pydantic.BaseModel]:
            return SentimentEstimate
        # --8<-- [end:custom-task-predictive-bridge-properties]

        # --8<-- [start:custom-task-predictive-bridge-methods]
        # We copy the result score into our doc's results attribute.
        def integrate(self, results: Sequence[SentimentEstimate], docs: list[Doc]) -> list[Doc]:
            """Integrate results into Doc instances."""
            for doc, result in zip(docs, results):
                assert isinstance(result, SentimentEstimate)
                # doc.results is a dict, with the task ID being the key to store our results under for the corresponding
                # task.
                doc.results[self._task_id] = result.score
            return docs

        # Consolidating multiple chunks for sentiment analysis can be pretty straightforward: we compute the average over
        # all chunks and assume this to be the sentiment score for the doc.
        def consolidate(
            self, results: Sequence[SentimentEstimate], docs_offsets: list[tuple[int, int]]
        ) -> Sequence[SentimentEstimate]:
            """Consolidate results for document chunks into document results."""
            consolidated_results: list[SentimentEstimate] = []

            # Iterate over indices that determine which chunks belong to which documents.
            for doc_offset in docs_offsets:
                # Keep track of all reasonings and the total score.
                reasonings: list[str] = []
                scores = 0.

                # Iterate over chunks' results.
                for chunk_result in results[doc_offset[0] : doc_offset[1]]:
                    # Model wrappers may return None results if they encounter errors and run in permissive mode. We ignore such
                    # results.
                    if chunk_result:
                        assert isinstance(chunk_result, SentimentEstimate)
                        reasonings.append(chunk_result.reasoning)
                        scores += chunk_result.score

                consolidated_results.append(SentimentEstimate(
                   # Average the score.
                   score=scores / (doc_offset[1] - doc_offset[0]),
                   # Concatenate all reasonings.
                   reasoning=str(reasonings)
                ))
            return consolidated_results
        # --8<-- [end:custom-task-predictive-bridge-methods]

    # --8<-- [start:custom-task-predictive-fewshot]
    # We'll define that class we require fewshot examples to be provided in. In our case we can just inherit from our
    # prompt signature class and add a `text` property.
    class FewshotExample(SentimentEstimate):
        text: str
    # --8<-- [end:custom-task-predictive-fewshot]


    # --8<-- [start:custom-task-predictive-task-class]
    class SentimentAnalysis(PredictiveTask[SentimentEstimate, SentimentEstimate, OutlinesSentimentAnalysis]):
    # --8<-- [end:custom-task-predictive-task-class]
        # --8<-- [start:custom-task-predictive-init-supports]
        # For the initialization of the bridge. We raise an error if an model wrapper has been specified that we don't
        # support (due to us not having a bridge implemented that would support this model type).
        def _init_bridge(self, model_type: ModelType) -> OutlinesSentimentAnalysis:
            if model_type == ModelType.outlines:
                return OutlinesSentimentAnalysis(
                    task_id=self._task_id,
                    prompt_instructions=self._custom_prompt_instructions,
                    overwrite=False,
                    model_settings=self._model_settings,
                )
            else:
                raise KeyError(f"Model type {model_type} is not supported by {self.__class__.__name__}.")

        # Represents set of supported model types.
        @property
        def supports(self) -> set[ModelType]:
            return {ModelType.outlines}
        # --8<-- [end:custom-task-predictive-init-supports]

        # --8<-- [start:custom-task-predictive-to-hf-dataset]
        # This implements the conversion of a set of docs to a Hugging Face datasets.Dataset.
        # You can implement this as `raise NotImplementedError` if you're not interested in generating a Hugging Face
        # dataset from your result data.
        def to_hf_dataset(self, docs: Iterable[Doc]) -> datasets.Dataset:
            # Define metadata.
            info = datasets.DatasetInfo(
                description=f"Sentiment estimation dataset. Generated with sieves"
                            f"v{Config.get_version()}.",
                features=datasets.Features({"text": datasets.Value("string"), "score": datasets.Value("float32")}),
            )

            def generate_data() -> Iterable[dict[str, Any]]:
                """Yields results as dicts.
                :return Iterable[dict[str, Any]]: Results as dicts.
                """
                for doc in docs:
                    yield {"text": doc.text, "score": doc.results[self._task_id]}

            # Create dataset.
            return datasets.Dataset.from_generator(generate_data, features=info.features, info=info)
        # --8<-- [end:custom-task-predictive-to-hf-dataset]

        # Distillation not implemented for this example task (not shown in docs)
        def distill(self, base_model_id, framework, data, output_path, val_frac, init_kwargs=None, train_kwargs=None, seed=None):
            raise NotImplementedError("Distillation not implemented for this example task")
    # --8<-- [end:custom-task-predictive]


def test_using_custom_task_example(small_outlines_model):
    """Demonstrates using the custom sentiment analysis task."""
    # First define the classes (from previous tests)
    from collections.abc import Iterable
    from functools import cached_property
    from typing import Any
    import pydantic
    import datasets
    from sieves.data import Doc
    from sieves.model_wrappers import ModelType, ModelWrapperInferenceMode, outlines_
    from sieves.tasks.predictive.bridges import Bridge
    from sieves.tasks.predictive.core import PredictiveTask
    from sieves.serialization import Config

    class SentimentEstimate(pydantic.BaseModel):
       reasoning: str
       score: pydantic.confloat(ge=0, le=1)

    class OutlinesSentimentAnalysis(Bridge[SentimentEstimate, SentimentEstimate, outlines_.InferenceMode]):
        @property
        def _default_prompt_instructions(self) -> str:
            return """
            Estimate the sentiment in this text as a float between 0 and 1. 0 is negative, 1 is positive. Provide your
            reasoning for why you estimate this score before you output the score.

            {% if examples|length > 0 -%}
                Examples:
                ----------
                {%- for example in examples %}
                    Text: "{{ example.text }}":
                    Output:
                        Reasoning: "{{ example.reasoning }}":
                        Sentiment: "{{ example.sentiment }}"
                {% endfor -%}
                ----------
            {% endif -%}

            ========
            Text: {{ text }}
            Output:
            """

        @property
        def _prompt_example_template(self) -> str | None:
            return None

        @property
        def _prompt_conclusion(self) -> str | None:
            return None

        @property
        def inference_mode(self) -> outlines_.InferenceMode:
            return self._model_settings.inference_mode or outlines_.InferenceMode.json

        @cached_property
        def prompt_signature(self) -> type[pydantic.BaseModel]:
            """Return the prompt signature."""
            return SentimentEstimate

        def integrate(self, results: Sequence[SentimentEstimate], docs: list[Doc]) -> list[Doc]:
            """Integrate results into Doc instances."""
            for doc, result in zip(docs, results):
                if result:
                    assert isinstance(result, SentimentEstimate)
                    doc.results[self._task_id] = result.score
            return docs

        def consolidate(
            self, results: Sequence[SentimentEstimate], docs_offsets: list[tuple[int, int]]
        ) -> Sequence[SentimentEstimate]:
            """Consolidate results for document chunks into document results."""
            results = list(results)
            consolidated_results: list[SentimentEstimate] = []
            for doc_offset in docs_offsets:
                reasonings: list[str] = []
                scores = 0.
                for chunk_result in results[doc_offset[0] : doc_offset[1]]:
                    if chunk_result:
                        assert isinstance(chunk_result, SentimentEstimate)
                        reasonings.append(chunk_result.reasoning)
                        scores += chunk_result.score
                consolidated_results.append(SentimentEstimate(
                   score=scores / (doc_offset[1] - doc_offset[0]),
                   reasoning=str(reasonings)
                ))
            return consolidated_results

    class FewshotExample(SentimentEstimate):
        """Few-shot example for sentiment analysis."""
        text: str

    class SentimentAnalysis(PredictiveTask[SentimentEstimate, SentimentEstimate, OutlinesSentimentAnalysis]):
        """Custom sentiment analysis task."""
        def __init__(self, model, task_id: str = "SentimentAnalysis", include_meta: bool = True, batch_size: int = -1,
                     prompt_instructions: str | None = None, fewshot_examples: Any = (),
                     model_settings=None):
            if model_settings is None:
                from sieves.model_wrappers.types import ModelSettings
                model_settings = ModelSettings()
            super().__init__(
                model=model, task_id=task_id, include_meta=include_meta, batch_size=batch_size,
                overwrite=False, prompt_instructions=prompt_instructions, fewshot_examples=fewshot_examples,
                model_settings=model_settings, condition=None
            )

        def _init_bridge(self, model_type: ModelType) -> OutlinesSentimentAnalysis:
            if model_type == ModelType.outlines:
                return OutlinesSentimentAnalysis(
                    task_id=self._task_id,
                    prompt_instructions=self._custom_prompt_instructions,
                    overwrite=False,
                    model_settings=self._model_settings,
                )
            else:
                raise KeyError(f"Model type {model_type} is not supported by {self.__class__.__name__}.")

        @property
        def supports(self) -> set[ModelType]:
            return {ModelType.outlines}

        def to_hf_dataset(self, docs: Iterable[Doc]) -> datasets.Dataset:
            info = datasets.DatasetInfo(
                description=f"Sentiment estimation dataset. Generated with sieves v{Config.get_version()}.",
                features=datasets.Features({"text": datasets.Value("string"), "score": datasets.Value("float32")}),
            )
            def generate_data() -> Iterable[dict[str, Any]]:
                for doc in docs:
                    yield {"text": doc.text, "score": doc.results[self._task_id]}
            return datasets.Dataset.from_generator(generate_data, features=info.features, info=info)

        def distill(self, base_model_id, framework, data, output_path, val_frac, init_kwargs=None, train_kwargs=None, seed=None):
            raise NotImplementedError("Distillation not implemented for this example task")

    # --8<-- [start:custom-task-usage]
    from sieves import Doc, Pipeline
    import outlines

    model = small_outlines_model

    docs = [Doc(text="I'm feeling happy today."), Doc(text="I was sad yesterday.")]
    pipe = Pipeline([SentimentAnalysis(model=model)])

    for doc in pipe(docs):
        print(doc.text, doc.results["SentimentAnalysis"])
    # --8<-- [end:custom-task-usage]
