# mypy: ignore-errors
import pytest
from flaky import flaky

from sieves import Doc, Pipeline
from sieves.model_wrappers import ModelType, ModelSettings, dspy_, langchain_, outlines_
from sieves.serialization import Config
from sieves.tasks import PredictiveTask, QuestionAnswering
from sieves.tasks.predictive import question_answering


@flaky(max_runs=3, min_passes=1)
@pytest.mark.parametrize(
    "batch_runtime",
    QuestionAnswering.supports(),
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
            questions=("What are biologists interested in?",),
            answers=("Studying life on earth.",),
        ),
    ]

    fewshot_args = {"fewshot_examples": fewshot_examples} if fewshot else {}
    task = question_answering.QuestionAnswering(
        task_id="qa",
        questions=[
            "What branch of science is this text describing?",
            "What the goal of the science as described in the text?",
        ],
        model=batch_runtime.model,
        model_settings=batch_runtime.model_settings,
        batch_size=batch_runtime.batch_size,
        **fewshot_args,
    )
    pipe = Pipeline(task)
    docs = list(pipe(qa_docs))

    assert len(docs) == 2
    for doc in docs:
        assert doc.text
        assert "qa" in doc.results
        assert "qa" in doc.meta
        assert "raw" in doc.meta["qa"]
        assert "usage" in doc.meta["qa"]
        assert "usage" in doc.meta

        print(f"Output: {doc.results['qa']}")
        print(f"Raw output: {doc.meta['qa']['raw']}")
        print(f"Usage: {doc.meta['qa']['usage']}")
        print(f"Total Usage: {doc.meta['usage']}")

    with pytest.raises(NotImplementedError):
        pipe["qa"].distill(None, None, None, None, None, None, None, None)

    if fewshot:
        _to_hf_dataset(task, docs)

def _to_hf_dataset(task: QuestionAnswering, docs: list[Doc]) -> None:
    """Tests whether conversion to HF dataset works as expected.

    :param task: QuestionAnswering task instance.
    :param docs: List of documents to convert.
    """
    dataset = task.to_hf_dataset(docs)
    assert all([key in dataset.features for key in ("text", "answers")])
    assert len(dataset) == 2
    dataset_records = list(dataset)
    for rec in dataset_records:
        assert isinstance(rec["text"], str)
        assert isinstance(rec["answers"], list)

    with pytest.raises(KeyError):
        task.to_hf_dataset([Doc(text="This is a dummy text.")])


@pytest.mark.parametrize("batch_runtime", [ModelType.dspy], indirect=["batch_runtime"])
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
                model_settings=batch_runtime.model_settings,
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
                                                           'model_settings': {'is_placeholder': False,
                                                                                   'value': {
                                                                                       'config_kwargs': None,
                                                                                       'inference_kwargs': None,
                                                                                       'init_kwargs': None,
                                                                                       'strict': True,
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
    QuestionAnswering.supports(),
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
        model_settings=ModelSettings(inference_mode=dummy),
        batch_size=batch_runtime.batch_size,
    )

    assert task._bridge.inference_mode == dummy
