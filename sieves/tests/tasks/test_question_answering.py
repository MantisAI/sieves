# mypy: ignore-errors
import gliner.multitask
import pytest

from sieves import Doc, Pipeline
from sieves.engines import EngineType, GliX
from sieves.tasks import PredictiveTask
from sieves.tasks.predictive import question_answering


@pytest.mark.parametrize(
    "batch_engine",
    (
        EngineType.dspy,
        EngineType.glix,
        EngineType.instructor,
        EngineType.langchain,
        EngineType.ollama,
        EngineType.outlines,
    ),
    indirect=["batch_engine"],
)
@pytest.mark.parametrize("fewshot", [True, False])
def test_run(qa_docs, batch_engine, fewshot):
    fewshot_examples = [
        question_answering.TaskFewshotExample(
            text="""
            Physics is the scientific study of matter, its fundamental constituents, its motion and behavior through 
            space and time, and the related entities of energy and force. Physics is one of the most fundamental 
            scientific disciplines. A scientist who specializes in the field of physics is called a physicist.
            """,
            reasoning="The text states ad verbatim what a scientist specializing in physics is called.",
            questions=("What's a scientist called who specializes in the field of physics?",),
            answers=("A physicist.",),
        ),
        question_answering.TaskFewshotExample(
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

    # If GliX engine: by default initialized as classifier, change that to QA.
    if isinstance(batch_engine, GliX):
        batch_engine._model = gliner.multitask.GLiNERQuestionAnswerer(model=batch_engine._model.model)

    fewshot_args = {"fewshot_examples": fewshot_examples} if fewshot else {}
    pipe = Pipeline(
        [
            question_answering.QuestionAnswering(
                task_id="qa",
                questions=[
                    "What branch of science is this text describing?",
                    "What the goal of the science as described in the text?",
                ],
                engine=batch_engine,
                **fewshot_args,
            ),
        ]
    )
    docs = list(pipe(qa_docs))

    assert len(docs) == 2
    for doc in docs:
        assert doc.text
        assert "qa" in doc.results


@pytest.mark.parametrize("batch_engine", [EngineType.outlines], indirect=["batch_engine"])
def test_to_dataset(qa_docs, batch_engine) -> None:
    task = question_answering.QuestionAnswering(
        task_id="qa",
        questions=[
            "What branch of science is this text describing?",
            "What the goal of the science as described in the text?",
        ],
        engine=batch_engine,
    )

    assert isinstance(task, PredictiveTask)
    dataset = task.to_dataset(task(qa_docs))
    assert all([key in dataset.features for key in ("text", "answers")])
    assert len(dataset) == 2
    dataset_records = list(dataset)
    for rec in dataset_records:
        assert isinstance(rec["text"], str)
        assert isinstance(rec["answers"], list)

    with pytest.raises(KeyError):
        task.to_dataset([Doc(text="This is a dummy text.")])


@pytest.mark.parametrize("batch_engine", [EngineType.outlines], indirect=["batch_engine"])
def test_serialization(qa_docs, batch_engine) -> None:
    pipe = Pipeline(
        [
            question_answering.QuestionAnswering(
                task_id="qa",
                questions=[
                    "What branch of science is this text describing?",
                    "What the goal of the science as described in the text?",
                ],
                engine=batch_engine,
            )
        ]
    )
    list(pipe(qa_docs))

    config = pipe.serialize()
    assert config.model_dump() == {
        "cls_name": "sieves.pipeline.core.Pipeline",
        "tasks": {
            "is_placeholder": False,
            "value": [
                {
                    "cls_name": "sieves.tasks.predictive.question_answering.core.QuestionAnswering",
                    "engine": {
                        "is_placeholder": False,
                        "value": {
                            "cls_name": "sieves.engines.outlines_.Outlines",
                            "inference_kwargs": {"is_placeholder": False, "value": {}},
                            "init_kwargs": {"is_placeholder": False, "value": {}},
                            "model": {"is_placeholder": True, "value": "outlines.models.transformers.Transformers"},
                            "version": "0.7.0",
                        },
                    },
                    "fewshot_examples": {"is_placeholder": False, "value": ()},
                    "include_meta": {"is_placeholder": False, "value": True},
                    "prompt_signature_desc": {"is_placeholder": False, "value": None},
                    "prompt_template": {"is_placeholder": False, "value": None},
                    "questions": {
                        "is_placeholder": False,
                        "value": [
                            "What branch of science is this " "text describing?",
                            "What the goal of the science as " "described in the text?",
                        ],
                    },
                    "show_progress": {"is_placeholder": False, "value": True},
                    "task_id": {"is_placeholder": False, "value": "qa"},
                    "version": "0.7.0",
                }
            ],
        },
        "version": "0.7.0",
    }

    Pipeline.deserialize(config=config, tasks_kwargs=[{"engine": {"model": batch_engine.model}}])
