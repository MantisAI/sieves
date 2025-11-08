# mypy: ignore-errors
import pytest

from sieves import Doc, Pipeline
from sieves.engines import EngineType, GenerationSettings
from sieves.serialization import Config
from sieves.tasks import PredictiveTask
from sieves.tasks.predictive import ner
from sieves.tasks.predictive.ner.core import Entity


@pytest.mark.parametrize(
    "batch_runtime",
    (
        EngineType.dspy,
        EngineType.langchain,
        EngineType.outlines,
        EngineType.glix,
    ),
    indirect=["batch_runtime"],
)
@pytest.mark.parametrize("fewshot", [True, False])
def test_run(ner_docs, batch_runtime, fewshot) -> None:
    fewshot_examples = [
        ner.FewshotExample(
            text="John studied data science in Barcelona and lives with Jaume",
            entities=[
                Entity(text="John", context="John studied data", entity_type="PERSON"),
                Entity(text="Barcelona", context="science in Barcelona", entity_type="LOCATION"),
                Entity(text="Jaume", context="lives with Jaume", entity_type="PERSON"),
            ],
        ),
        ner.FewshotExample(
            text="Maria studied computer engineering in Madrid and works with Carlos",
            entities=[
                Entity(text="Maria", context="Maria studied computer", entity_type="PERSON"),
                Entity(text="Madrid", context="engineering in Madrid and works", entity_type="LOCATION"),
                Entity(text="Carlos", context="works with Carlos", entity_type="PERSON"),
            ],
        ),
    ]

    fewshot_args = {"fewshot_examples": fewshot_examples} if fewshot else {}
    pipe = Pipeline(
        ner.NER(
            entities=["PERSON", "LOCATION", "COMPANY"],
            model=batch_runtime.model,
            generation_settings=batch_runtime.generation_settings,
            batch_size=batch_runtime.batch_size,
            **fewshot_args
        )
    )
    docs = list(pipe(ner_docs))

    assert len(docs) == 2
    for doc in docs:
        assert "NER" in doc.results

    with pytest.raises(NotImplementedError):
        pipe["NER"].distill(None, None, None, None, None, None, None, None)


@pytest.mark.parametrize("batch_runtime", [EngineType.dspy], indirect=["batch_runtime"])
def test_serialization(ner_docs, batch_runtime) -> None:
    pipe = Pipeline(
        ner.NER(
            entities=["PERSON", "LOCATION", "COMPANY"],
            model=batch_runtime.model,
            generation_settings=batch_runtime.generation_settings,
            batch_size=batch_runtime.batch_size,
        )
    )

    config = pipe.serialize()
    assert config.model_dump() == {'cls_name': 'sieves.pipeline.core.Pipeline',
 'tasks': {'is_placeholder': False,
           'value': [{'cls_name': 'sieves.tasks.predictive.ner.core.NER',
                      'entities': {'is_placeholder': False,
                                   'value': ['PERSON', 'LOCATION', 'COMPANY']},
                      'fewshot_examples': {'is_placeholder': False,
                                           'value': ()},
                      'batch_size': {'is_placeholder': False, "value": -1},
                      'generation_settings': {'is_placeholder': False,
                                              'value': {
                                                        'config_kwargs': None,
                                                        'inference_kwargs': None,
                                                        'init_kwargs': None,
                                                        'strict_mode': False,
                                                        'inference_mode': None,}},
                      'include_meta': {'is_placeholder': False, 'value': True},
                      'model': {'is_placeholder': True,
                                'value': 'dspy.clients.lm.LM'},
                      'prompt_instructions': {'is_placeholder': False,
                                          'value': None},
                      'task_id': {'is_placeholder': False, 'value': 'NER'},
                      'condition': {'is_placeholder': False, 'value': None},
                      'version': Config.get_version()}]},
 'use_cache': {'is_placeholder': False, 'value': True},
 'version': Config.get_version()}
    Pipeline.deserialize(
        config=config,
        tasks_kwargs=[{"model": batch_runtime.model}],
    )


@pytest.mark.parametrize("batch_runtime", [EngineType.glix], indirect=["batch_runtime"])
def test_to_hf_dataset(ner_docs, batch_runtime) -> None:
    task = ner.NER(
        entities=["PERSON", "LOCATION", "COMPANY"],
        model=batch_runtime.model,
        generation_settings=batch_runtime.generation_settings,
        batch_size=batch_runtime.batch_size,
    )
    pipe = Pipeline(task)

    assert isinstance(task, PredictiveTask)
    dataset = task.to_hf_dataset(pipe(ner_docs))
    assert all([key in dataset.features for key in ("text", "entities")])
    assert len(dataset) == 2
    dataset_records = list(dataset)
    for rec in dataset_records:
        assert isinstance(rec["entities"], dict)
        assert (
            len(rec["entities"]["entity_type"])
            == len(rec["entities"]["start"])
            == len(rec["entities"]["end"])
            == len(rec["entities"]["text"])
        )
        assert isinstance(rec["text"], str)

    with pytest.raises(KeyError):
        task.to_hf_dataset([Doc(text="This is a dummy text.")])


@pytest.mark.parametrize(
    "batch_runtime",
    [EngineType.dspy, EngineType.langchain, EngineType.outlines, EngineType.glix],
    indirect=["batch_runtime"],
)
def test_inference_mode_override(batch_runtime) -> None:
    """Test that inference_mode parameter overrides the default value."""
    dummy = "dummy_inference_mode"

    task = ner.NER(
        entities=["PERSON", "LOCATION", "COMPANY"],
        model=batch_runtime.model,
        generation_settings=GenerationSettings(inference_mode=dummy),
        batch_size=batch_runtime.batch_size,
    )

    assert task._bridge.inference_mode == dummy
