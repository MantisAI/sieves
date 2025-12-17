# mypy: ignore-errors
import pydantic
import pytest

from sieves import Doc, Pipeline
from sieves.model_wrappers import ModelType, GenerationSettings
from sieves.serialization import Config
from sieves.tasks import Classification
from sieves.tasks.predictive import classification
from sieves.tests.conftest import Runtime


def _run(
    runtime: Runtime, docs: list[Doc], fewshot: bool, multilabel: bool = True, test_hf_conversion: bool = False
) -> None:
    """Tests whether the classification task works as expected."""
    if multilabel:
        fewshot_examples = [
            classification.FewshotExampleMultiLabel(
                text="On the properties of hydrogen atoms and red dwarfs.",
                confidence_per_label={"science": 1.0, "politics": 0.0},
            ),
            classification.FewshotExampleMultiLabel(
                text="A parliament is elected by casting votes.",
                confidence_per_label={"science": 0, "politics": 1.0},
            ),
        ]
    else:
        fewshot_examples = [
            classification.FewshotExampleSingleLabel(
                text="On the properties of hydrogen atoms and red dwarfs.",
                label="science",
                confidence=1.0,
            ),
            classification.FewshotExampleSingleLabel(
                text="A parliament is elected by casting votes.",
                label="politics",
                confidence=1.0,
            ),
        ]

    fewshot_args = {"fewshot_examples": fewshot_examples} if fewshot else {}
    labels = {
        "science": "Topics related to scientific disciplines and research",
        "politics": "Topics related to government, elections, and political systems",
    }

    task = classification.Classification(
        task_id="classifier",
        labels=labels,
        model=runtime.model,
        generation_settings=runtime.generation_settings,
        batch_size=runtime.batch_size,
        multi_label=multilabel,
        **fewshot_args,
    )
    pipe = Pipeline(task)
    docs = list(pipe(docs))

    assert len(docs) == 2
    for doc in docs:
        assert doc.text
        assert doc
        assert "classifier" in doc.results

    if test_hf_conversion:
        _to_hf_dataset(task, docs, multilabel)

@pytest.mark.parametrize("batch_runtime", Classification.supports(), indirect=["batch_runtime"])
@pytest.mark.parametrize("fewshot", [True, False])
@pytest.mark.parametrize("multilabel", [True, False])
def test_run(classification_docs, batch_runtime, fewshot, multilabel):
    _run(batch_runtime, classification_docs, fewshot, multilabel, test_hf_conversion=fewshot is True)

@pytest.mark.parametrize("runtime", Classification.supports(), indirect=["runtime"])
@pytest.mark.parametrize("fewshot", [True, False])
def test_run_nonbatched(classification_docs, runtime, fewshot):
    _run(runtime, classification_docs, fewshot, test_hf_conversion=False)


def _to_hf_dataset(task: Classification, docs: list[Doc], multi_label: bool) -> None:
    """Tests whether conversion to HF dataset works as expected.

    :param task: Classification task instance.
    :param docs: List of documents to convert.
    :param multi_label: Whether the task is multi-label.
    """
    dataset = task.to_hf_dataset(docs)
    assert all([key in dataset.features for key in ("text", "labels")])
    assert len(dataset) == 2
    dataset_records = list(dataset)
    for rec in dataset_records:
        if multi_label:
            assert isinstance(rec["labels"], list)
            assert all(
                [isinstance(v, int) for v in rec["labels"]]
            ), "Labels should be integers for multi-label classification"
            for v in rec["labels"]:
                assert isinstance(v, int)
        else:
            assert isinstance(rec["labels"], int)
        assert isinstance(rec["text"], str)

    with pytest.raises(KeyError):
        task.to_hf_dataset([Doc(text="This is a dummy text.")])


@pytest.mark.parametrize("batch_runtime", [ModelType.huggingface], indirect=["batch_runtime"])
def test_serialization(classification_docs, batch_runtime) -> None:
    labels = {
        "science": "Topics related to scientific disciplines and research",
        "politics": "Topics related to government, elections, and political systems",
    }

    pipe = Pipeline(
        classification.Classification(
            task_id="classifier",
            labels=labels,
            model=batch_runtime.model,
            generation_settings=batch_runtime.generation_settings,
            batch_size=batch_runtime.batch_size,
        )
    )

    config = pipe.serialize()
    assert config.model_dump() == {'cls_name': 'sieves.pipeline.core.Pipeline',
                                   'tasks': {'is_placeholder': False,
                                             'value': [{
                                                           'cls_name': 'sieves.tasks.predictive.classification.core.Classification',
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
                                                           'labels': {'is_placeholder': False,
                                                                      'value': {'politics': 'Topics '
                                                                                                        'related to '
                                                                                                        'government, '
                                                                                                        'elections, '
                                                                                                        'and '
                                                                                                        'political '
                                                                                                        'systems',
                                                                                            'science': 'Topics '
                                                                                                       'related to '
                                                                                                       'scientific '
                                                                                                       'disciplines '
                                                                                                       'and '
                                                                                                       'research'}},
                                                           'model': {'is_placeholder': True,
                                                                     'value': 'transformers.pipelines.zero_shot_classification.ZeroShotClassificationPipeline'},
                                                           'prompt_instructions': {'is_placeholder': False,
                                                                                   'value': None},
                                                           'task_id': {'is_placeholder': False,
                                                                       'value': 'classifier'},
                                                           'condition': {'is_placeholder': False, 'value': None},
                                                           'version': Config.get_version()}]},
                                   'use_cache': {'is_placeholder': False, 'value': True},
                                   'version': Config.get_version()}

    Pipeline.deserialize(config=config, tasks_kwargs=[{"model": batch_runtime.model}])


@pytest.mark.parametrize("batch_runtime", [ModelType.huggingface], indirect=["batch_runtime"])
def test_labels_validation(batch_runtime) -> None:
    """Test that labels parameter accepts both list and dict formats."""
    # Valid case - list format (no descriptions)
    classification.Classification(
        labels=["science", "politics"],
        model=batch_runtime.model,
        generation_settings=batch_runtime.generation_settings,
        batch_size=batch_runtime.batch_size,
    )

    # Valid case - dict format with all labels having descriptions
    labels_with_descriptions = {"science": "Science related", "politics": "Politics related"}
    classification.Classification(
        labels=labels_with_descriptions,
        model=batch_runtime.model,
        generation_settings=batch_runtime.generation_settings,
        batch_size=batch_runtime.batch_size,
    )

    # Valid case - dict format with some labels having descriptions (empty strings)
    partial_descriptions = {"science": "Science related", "politics": ""}
    classification.Classification(
        labels=partial_descriptions,
        model=batch_runtime.model,
        generation_settings=batch_runtime.generation_settings,
        batch_size=batch_runtime.batch_size,
    )


def test_fewshot_example_singlelabel_confidence() -> None:
    """Test that the confidence of a fewshot example is correctly validated."""
    classification.FewshotExampleSingleLabel(
        text="...",
        label="science",
        confidence=1.0,
    )

    with pytest.raises(ValueError):
        classification.FewshotExampleSingleLabel(
            text="...",
            label="science",
            confidence=2.0,
        )

    with pytest.raises(ValueError):
        classification.FewshotExampleSingleLabel(
            text="...",
            label="science",
            confidence=-2.0,
        )


def test_result_to_scores() -> None:
    """Test that the result to scores method works as expected."""
    # 1) list of (label, score) pairs
    res = [("science", 0.9), ("politics", 0.1)]
    scores = classification.Classification._result_to_scores(res)
    assert scores == {"science": 0.9, "politics": 0.1}

    # 2) single (label, score) tuple
    res = ("science", 0.8)
    scores = classification.Classification._result_to_scores(res)
    assert scores == {"science": 0.8}

    # 3) plain label string -> assumes score 1.0
    res = "science"
    scores = classification.Classification._result_to_scores(res)
    assert scores == {"science": 1.0}

    # 4) Pydantic model with label and score
    class PResWithScore(pydantic.BaseModel):  # type: ignore[attr-defined]
        label: str
        score: float

    pyd_res = PResWithScore(label="science", score=0.55)
    scores = classification.Classification._result_to_scores(pyd_res)
    assert scores == {"science": 0.55}

    # 5) Pydantic model with only label (defaults to 1.0)
    class PResLabelOnly(pydantic.BaseModel):  # type: ignore[attr-defined]
        label: str

    pyd_res2 = PResLabelOnly(label="politics")
    scores = classification.Classification._result_to_scores(pyd_res2)
    assert scores == {"politics": 1.0}

    # 6) Unsupported types raise TypeError
    with pytest.raises(TypeError):
        classification.Classification._result_to_scores({"science": 0.9})

    with pytest.raises(TypeError):
        classification.Classification._result_to_scores([("science", 0.9, 123)])

    with pytest.raises(TypeError):

        class BadPRes(pydantic.BaseModel):  # type: ignore[attr-defined]
            not_label: str

        classification.Classification._result_to_scores(BadPRes(not_label="x"))


@pytest.mark.parametrize("batch_runtime", Classification.supports(), indirect=["batch_runtime"])
def test_inference_mode_override(batch_runtime) -> None:
    """Test that inference_mode parameter overrides the default value."""
    dummy = "dummy_inference_mode"

    task = classification.Classification(
        task_id="classifier",
        labels=["science", "politics"],
        model=batch_runtime.model,
        generation_settings=GenerationSettings(inference_mode=dummy),
        batch_size=batch_runtime.batch_size,
    )

    assert task._bridge.inference_mode == dummy
