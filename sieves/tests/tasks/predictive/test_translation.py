# mypy: ignore-errors
import pytest

from sieves import Doc, Pipeline
from sieves.engines import EngineType
from sieves.serialization import Config
from sieves.tasks import PredictiveTask
from sieves.tasks.predictive import translation


@pytest.mark.parametrize(
    "batch_engine",
    (
        EngineType.instructor,
        EngineType.langchain,
        EngineType.ollama,
        EngineType.outlines,
        # EngineType.vllm
    ),
    indirect=["batch_engine"],
)
@pytest.mark.parametrize("fewshot", [True, False])
def test_run(translation_docs, batch_engine, fewshot) -> None:
    fewshot_examples = [
        translation.FewshotExample(
            text="The sun is shining today.",
            to="Spanish",
            translation="El sol brilla hoy.",
        ),
        translation.FewshotExample(
            text="There's a lot of fog today",
            to="Spanish",
            translation="Hay mucha niebla hoy.",
        ),
    ]

    fewshot_args = {"fewshot_examples": fewshot_examples} if fewshot else {}
    pipe = Pipeline([translation.Translation(to="Spanish", engine=batch_engine, **fewshot_args)])
    docs = list(pipe(translation_docs))

    assert len(docs) == 2
    for doc in docs:
        assert doc.text
        assert "Translation" in doc.results

    with pytest.raises(NotImplementedError):
        pipe["Translation"].distill(None, None, None, None, None, None, None, None)


@pytest.mark.parametrize("batch_engine", [EngineType.dspy], indirect=["batch_engine"])
def test_to_hf_dataset(translation_docs, batch_engine) -> None:
    task = translation.Translation(to="Spanish", engine=batch_engine)
    docs = task(translation_docs)

    assert isinstance(task, PredictiveTask)
    dataset = task.to_hf_dataset(docs)
    assert all([key in dataset.features for key in ("text", "translation")])
    assert len(dataset) == 2
    records = list(dataset)
    assert records[0]["text"] == "It is rainy today."
    assert records[1]["text"] == "It is cloudy today."
    for record in records:
        assert isinstance(record["translation"], str)

    with pytest.raises(KeyError):
        task.to_hf_dataset([Doc(text="This is a dummy text.")])


@pytest.mark.parametrize("batch_engine", [EngineType.dspy], indirect=["batch_engine"])
def test_serialization(translation_docs, batch_engine) -> None:
    pipe = Pipeline([translation.Translation(to="Spanish", engine=batch_engine)])

    config = pipe.serialize()
    assert config.model_dump() == {
        "cls_name": "sieves.pipeline.core.Pipeline",
        "use_cache": {"is_placeholder": False, "value": True},
        "tasks": {
            "is_placeholder": False,
            "value": [
                {
                    "cls_name": "sieves.tasks.predictive.translation.core.Translation",
                    "engine": {
                        "is_placeholder": False,
                        "value": {
                            "cls_name": "sieves.engines.wrapper.Engine",
                            "strict_mode": {"is_placeholder": False, "value": False},
                            "inference_kwargs": {"is_placeholder": False, "value": {}},
                            "init_kwargs": {"is_placeholder": False, "value": {}},
                            "model": {"is_placeholder": True, "value": "dspy.clients.lm.LM"},
                            "batch_size": {"is_placeholder": False, "value": -1},
                            "version": Config.get_version(),
                        },
                    },
                    "fewshot_examples": {"is_placeholder": False, "value": ()},
                    "include_meta": {"is_placeholder": False, "value": True},
                    "prompt_signature_desc": {"is_placeholder": False, "value": None},
                    "prompt_template": {"is_placeholder": False, "value": None},
                    "task_id": {"is_placeholder": False, "value": "Translation"},
                    "to": {"is_placeholder": False, "value": "Spanish"},
                    "version": Config.get_version(),
                }
            ],
        },
        "version": Config.get_version(),
    }

    Pipeline.deserialize(config=config, tasks_kwargs=[{"engine": {"model": batch_engine.model}}])
