# mypy: ignore-errors
import pytest

from sieves import Doc, Pipeline, tasks
from sieves.engines import EngineType
from sieves.serialization import Config
from sieves.tasks import PredictiveTask
from sieves.tasks.predictive import pii_masking


@pytest.mark.parametrize(
    "batch_runtime",
    (
        EngineType.dspy,
        EngineType.instructor,
        EngineType.langchain,
        EngineType.ollama,
        EngineType.outlines,
        # EngineType.vllm,
    ),
    indirect=["batch_runtime"],
)
@pytest.mark.parametrize("fewshot", [True, False])
def test_run(pii_masking_docs, batch_runtime, fewshot) -> None:
    fewshot_examples = [
        pii_masking.FewshotExample(
            text="Jane Smith works at NASA.",
            reasoning="Jane Smith is a person's name and should be masked.",
            masked_text="[MASKED] works at NASA.",
            pii_entities=[pii_masking.PIIEntity(entity_type="PERSON", text="Jane Smith")],
        ),
        pii_masking.FewshotExample(
            text="He lives at Diagon Alley 37.",
            reasoning="Diagon Alley 37 is a residential address and should be masked.",
            masked_text="He lives at [MASKED].",
            pii_entities=[pii_masking.PIIEntity(entity_type="ADDRESS", text="Diagon Alley 37")],
        ),
    ]

    fewshot_args = {"fewshot_examples": fewshot_examples} if fewshot else {}
    pipe = Pipeline([
        tasks.predictive.PIIMasking(
            model=batch_runtime.model,
            generation_settings=batch_runtime.generation_settings,
            **fewshot_args,
        )
    ])
    docs = list(pipe(pii_masking_docs))

    assert len(docs) == 2
    for doc in docs:
        assert doc.text
        assert "PIIMasking" in doc.results

    with pytest.raises(NotImplementedError):
        pipe["PIIMasking"].distill(None, None, None, None, None, None, None, None)


@pytest.mark.parametrize("batch_runtime", [EngineType.dspy], indirect=["batch_runtime"])
def test_to_hf_dataset(pii_masking_docs, batch_runtime) -> None:
    task = tasks.predictive.PIIMasking(
        model=batch_runtime.model,
        generation_settings=batch_runtime.generation_settings,
    )
    docs = task(pii_masking_docs)

    assert isinstance(task, PredictiveTask)
    dataset = task.to_hf_dataset(docs)
    assert all([key in dataset.features for key in ("text", "masked_text")])
    assert len(dataset) == 2
    records = list(dataset)
    assert records[0]["text"] == "Her SSN is 222-333-444. Her credit card number is 1234 5678."
    assert records[1]["text"] == "You can reach Michael at michael.michaels@gmail.com."

    with pytest.raises(KeyError):
        task.to_hf_dataset([Doc(text="This is a dummy text.")])


@pytest.mark.parametrize("batch_runtime", [EngineType.dspy], indirect=["batch_runtime"])
def test_serialization(pii_masking_docs, batch_runtime) -> None:
    pipe = Pipeline([
        tasks.predictive.PIIMasking(
            model=batch_runtime.model,
            generation_settings=batch_runtime.generation_settings,
        )
    ])

    config = pipe.serialize()
    assert config.model_dump() == {'cls_name': 'sieves.pipeline.core.Pipeline',
 'tasks': {'is_placeholder': False,
           'value': [{'cls_name': 'sieves.tasks.predictive.pii_masking.core.PIIMasking',
                      'fewshot_examples': {'is_placeholder': False,
                                           'value': ()},
                      'generation_settings': {'is_placeholder': False,
                                              'value': {'batch_size': -1,
                                                        'config_kwargs': None,
                                                        'inference_kwargs': None,
                                                        'init_kwargs': None,
                                                        'strict_mode': False}},
                      'include_meta': {'is_placeholder': False, 'value': True},
                      'mask_placeholder': {'is_placeholder': False,
                                           'value': '[MASKED]'},
                      'model': {'is_placeholder': True,
                                'value': 'dspy.clients.lm.LM'},
                      'pii_types': {'is_placeholder': False, 'value': None},
                      'prompt_signature_desc': {'is_placeholder': False,
                                                'value': None},
                      'prompt_template': {'is_placeholder': False,
                                          'value': None},
                      'task_id': {'is_placeholder': False,
                                  'value': 'PIIMasking'},
                      'version': Config.get_version()}]},
 'use_cache': {'is_placeholder': False, 'value': True},
 'version': Config.get_version()}

    Pipeline.deserialize(config=config, tasks_kwargs=[{"model": batch_runtime.model}])
