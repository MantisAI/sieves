# mypy: ignore-errors
import pytest

from sieves import Doc, Pipeline
from sieves.engines import EngineType, GenerationSettings, dspy_, langchain_, outlines_
from sieves.serialization import Config
from sieves.tasks import PredictiveTask
from sieves.tasks.predictive import question_answering


@pytest.mark.parametrize(
    "batch_runtime",
    (
        EngineType.dspy,
        EngineType.glix,
        EngineType.langchain,
        EngineType.outlines,
    ),
    indirect=["batch_runtime"],
)
@pytest.mark.parametrize("fewshot", [True, False])
def test_run(qa_docs, batch_runtime, fewshot):
    fewshot_examples = [
        question_answering.FewshotExample(
            text="""
            Physics is the scientific study of matter, its fundamental constituents, its motion and behavior through
            space and time, and the related entities of energy and force. Physics is one of the most fundamental
            scientific disciplines. A scientist who specializes in the field of physics is called a physicist.
            """,
            reasoning="The text states ad verbatim what a scientist specializing in physics is called.",
            questions=("What's a scientist called who specializes in the field of physics?",),
            answers=("A physicist.",),
        ),
        question_answering.FewshotExample(
            text="""
            A biologist is a scientist who conducts research in biology. Biologists are interested in studying life on
            Earth, whether it is an individual cell, a multicellular organism, or a community of interacting
            populations. They usually specialize in a particular branch (e.g., molecular biology, zoology, and
            evolutionary biology) of biology and have a specific research focus (e.g., studying malaria or cancer).
            """,
            reasoning="The states ad verbatim that biologists are interested in studying life on earth.",
            questions=("What are biologists interested in?",),
            answers=("Studying life on earth.",),
        ),
    ]

    fewshot_args = {"fewshot_examples": fewshot_examples} if fewshot else {}
    pipe = Pipeline(
        [
            question_answering.QuestionAnswering(
                task_id="qa",
                questions=[
                    "What branch of science is this text describing?",
                    "What the goal of the science as described in the text?",
                ],
                model=batch_runtime.model,
                generation_settings=batch_runtime.generation_settings,
                batch_size=batch_runtime.batch_size,
                **fewshot_args,
            ),
        ]
    )
    docs = list(pipe(qa_docs))

    assert len(docs) == 2
    for doc in docs:
        assert doc.text
        assert "qa" in doc.results

    with pytest.raises(NotImplementedError):
        pipe["qa"].distill(None, None, None, None, None, None, None, None)


@pytest.mark.parametrize("batch_runtime", [EngineType.dspy], indirect=["batch_runtime"])
def test_to_hf_dataset(qa_docs, batch_runtime) -> None:
    task = question_answering.QuestionAnswering(
        task_id="qa",
        questions=[
            "What branch of science is this text describing?",
            "What the goal of the science as described in the text?",
        ],
        model=batch_runtime.model,
        generation_settings=batch_runtime.generation_settings,
        batch_size=batch_runtime.batch_size,
    )
    pipe = Pipeline(task)

    assert isinstance(task, PredictiveTask)
    dataset = task.to_hf_dataset(pipe(qa_docs))
    assert all([key in dataset.features for key in ("text", "answers")])
    assert len(dataset) == 2
    dataset_records = list(dataset)
    for rec in dataset_records:
        assert isinstance(rec["text"], str)
        assert isinstance(rec["answers"], list)

    with pytest.raises(KeyError):
        task.to_hf_dataset([Doc(text="This is a dummy text.")])


@pytest.mark.parametrize("batch_runtime", [EngineType.dspy], indirect=["batch_runtime"])
def test_serialization(qa_docs, batch_runtime) -> None:
    pipe = Pipeline(
        [
            question_answering.QuestionAnswering(
                task_id="qa",
                questions=[
                    "What branch of science is this text describing?",
                    "What the goal of the science as described in the text?",
                ],
                model=batch_runtime.model,
                generation_settings=batch_runtime.generation_settings,
                batch_size=batch_runtime.batch_size,
            )
        ]
    )

    config = pipe.serialize()
    assert config.model_dump() == {'cls_name': 'sieves.pipeline.core.Pipeline',
                                   'tasks': {'is_placeholder': False,
                                             'value': [{
                                                           'cls_name': 'sieves.tasks.predictive.question_answering.core.QuestionAnswering',
                                                           'fewshot_examples': {'is_placeholder': False,
                                                                                'value': ()},
                                                           'batch_size': {'is_placeholder': False, "value": -1},
                                                           'generation_settings': {'is_placeholder': False,
                                                                                   'value': {
                                                                                       'config_kwargs': None,
                                                                                       'inference_kwargs': None,
                                                                                       'init_kwargs': None,
                                                                                       'strict_mode': False,
                                                                                       'inference_mode': None}},
                                                           'include_meta': {'is_placeholder': False, 'value': True},
                                                           'model': {'is_placeholder': True,
                                                                     'value': 'dspy.clients.lm.LM'},
                                                           'prompt_instructions': {'is_placeholder': False,
                                                                                   'value': None},
                                                           'questions': {'is_placeholder': False,
                                                                         'value': ['What branch of science is this '
                                                                                   'text describing?',
                                                                                   'What the goal of the science as '
                                                                                   'described in the text?']},
                                                           'task_id': {'is_placeholder': False, 'value': 'qa'},
                                                           'condition': {'is_placeholder': False, 'value': None},
                                                           'version': Config.get_version()}]},
                                   'use_cache': {'is_placeholder': False, 'value': True},
                                   'version': Config.get_version()}

    Pipeline.deserialize(config=config, tasks_kwargs=[{"model": batch_runtime.model}])


@pytest.mark.parametrize(
    "batch_runtime",
    [EngineType.dspy, EngineType.glix, EngineType.langchain, EngineType.outlines],
    indirect=["batch_runtime"],
)
def test_inference_mode_override(batch_runtime) -> None:
    """Test that inference_mode parameter overrides the default value."""
    dummy = "dummy_inference_mode"

    task = question_answering.QuestionAnswering(
        task_id="qa",
        questions=[
            "What branch of science is this text describing?",
            "What the goal of the science as described in the text?",
        ],
        model=batch_runtime.model,
        generation_settings=GenerationSettings(inference_mode=dummy),
        batch_size=batch_runtime.batch_size,
    )

    assert task._bridge.inference_mode == dummy
