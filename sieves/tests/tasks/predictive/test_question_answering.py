# mypy: ignore-errors
import pytest
from flaky import flaky

from sieves import Doc, Pipeline
from sieves.model_wrappers import ModelType, ModelSettings, dspy_, langchain_, outlines_
from sieves.serialization import Config
from sieves.tasks import PredictiveTask, QuestionAnswering
from sieves.tasks.predictive import question_answering
from sieves.tasks.predictive.schemas.question_answering import Result, QuestionAnswer


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
            questions=["What's a scientist called who specializes in the field of physics?"],
            answers=["A physicist."],
            scores=[1.0],
        ),
        question_answering.FewshotExample(
            text="""
            A biologist is a scientist who conducts research in biology. Biologists are interested in studying life on
            Earth, whether it is an individual cell, a multicellular organism, or a community of interacting
            populations. They usually specialize in a particular branch (e.g., molecular biology, zoology, and
            evolutionary biology) of biology and have a specific research focus (e.g., studying malaria or cancer).
            """,
            questions=["What are biologists interested in?"],
            answers=["Studying life."],
            scores=[1.0],
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

        # Verify unified result types.
        assert isinstance(doc.results["qa"], question_answering.Result)

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
    assert all([key in dataset.features for key in ("text", "answers", "scores")])
    assert len(dataset) == 2
    dataset_records = list(dataset)
    for rec in dataset_records:
        assert isinstance(rec["text"], str)
        assert isinstance(rec["answers"], list)
        assert isinstance(rec["scores"], list)

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


@pytest.mark.parametrize("batch_runtime", [ModelType.dspy], indirect=["batch_runtime"])
def test_evaluation(batch_runtime) -> None:
    """Test evaluation for QA using a real judge."""
    task = question_answering.QuestionAnswering(
        questions=["What is 1+1?"], model=batch_runtime.model, task_id="qa"
    )

    # 1. Full overlap
    doc_full = Doc(text="1+1 equals 2")
    res_full = Result(qa_pairs=[QuestionAnswer(question="What is 1+1?", answer="2")])
    doc_full.results["qa"] = res_full
    doc_full.gold["qa"] = res_full
    report_full = task.evaluate([doc_full], judge=batch_runtime.model)
    assert report_full.metrics[task.metric] > 0.8

    # 2. No overlap
    doc_none = Doc(text="1+1 equals 2")
    doc_none.results["qa"] = Result(qa_pairs=[QuestionAnswer(question="What is 1+1?", answer="It is a sunny day.")])
    doc_none.gold["qa"] = res_full
    report_none = task.evaluate([doc_none], judge=batch_runtime.model)
    assert report_none.metrics[task.metric] < 0.6

    # 3. Partial overlap (one correct, one incorrect)
    task_multi = question_answering.QuestionAnswering(
        questions=["What is 1+1?", "What is 2+2?"], model=batch_runtime.model, task_id="qa_multi"
    )
    doc_partial = Doc(text="1+1 is 2 and 2+2 is 4")
    # Pred: 2 and 5
    doc_partial.results["qa_multi"] = Result(qa_pairs=[
        QuestionAnswer(question="What is 1+1?", answer="2"),
        QuestionAnswer(question="What is 2+2?", answer="5")
    ])
    # Gold: 2 and 4
    doc_partial.gold["qa_multi"] = Result(qa_pairs=[
        QuestionAnswer(question="What is 1+1?", answer="2"),
        QuestionAnswer(question="What is 2+2?", answer="4")
    ])
    report_partial = task_multi.evaluate([doc_partial], judge=batch_runtime.model)
    # Expected to be somewhere in the middle
    assert 0.2 < report_partial.metrics[task_multi.metric] < 0.8
