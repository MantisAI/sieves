# mypy: ignore-errors
import pytest

from sieves import Pipeline, Doc
from sieves.model_wrappers import ModelType, ModelSettings
from sieves.serialization import Config
from sieves.tasks.predictive import relation_extraction
from sieves.tasks.predictive.schemas.relation_extraction import RelationEntity, RelationTriplet, Result
from sieves.tests.conftest import _make_runtime


@pytest.mark.parametrize(
    "batch_runtime",
    relation_extraction.RelationExtraction.supports(),
    indirect=["batch_runtime"],
)
@pytest.mark.parametrize("fewshot", [True, False])
def test_run(relation_extraction_docs, batch_runtime, fewshot) -> None:
    # --8<-- [start:re-usage]
    relations = {
        "works_for": "A person works for a company or organization.",
        "located_in": "A place or organization is located in a city, country, or region.",
        "founded": "A person founded a company or organization.",
    }

    fewshot_examples = [
        relation_extraction.FewshotExample(
            text="Henri Dunant founded the Red Cross in Geneva.",
            triplets=[
                RelationTriplet(
                    head=RelationEntity(text="Henri Dunant", entity_type="PERSON"),
                    relation="founded",
                    tail=RelationEntity(text="Red Cross", entity_type="ORGANIZATION"),
                    score=1.0,
                ),
                RelationTriplet(
                    head=RelationEntity(text="Red Cross", entity_type="ORGANIZATION"),
                    relation="located_in",
                    tail=RelationEntity(text="Geneva", entity_type="LOCATION"),
                    score=1.0,
                ),
            ],
        ),
        relation_extraction.FewshotExample(
            text="Eglantyne Jebb founded Save the Children in London.",
            triplets=[
                RelationTriplet(
                    head=RelationEntity(text="Eglantyne Jebb", entity_type="PERSON"),
                    relation="founded",
                    tail=RelationEntity(text="Save the Children", entity_type="ORGANIZATION"),
                    score=1.0,
                ),
                RelationTriplet(
                    head=RelationEntity(text="Save the Children", entity_type="ORGANIZATION"),
                    relation="located_in",
                    tail=RelationEntity(text="London", entity_type="LOCATION"),
                    score=1.0,
                ),
            ],
        ),
    ]

    fewshot_args = {"fewshot_examples": fewshot_examples} if fewshot else {}

    task = relation_extraction.RelationExtraction(
        relations=relations,
        model=batch_runtime.model,
        model_settings=batch_runtime.model_settings,
        batch_size=batch_runtime.batch_size,
        entity_types=["PERSON", "ORGANIZATION", "LOCATION"],
        **fewshot_args
    )

    pipe = Pipeline(task)
    docs = list(pipe(relation_extraction_docs))
    # --8<-- [end:re-usage]

    assert len(docs) == 2
    for doc in docs:
        assert "RelationExtraction" in doc.results
        res = doc.results["RelationExtraction"]
        assert isinstance(res, relation_extraction.Result)
        assert len(res.triplets)

        assert "RelationExtraction" in doc.meta
        assert "raw" in doc.meta["RelationExtraction"]
        assert "usage" in doc.meta["RelationExtraction"]
        assert "usage" in doc.meta

        print(f"Output: {doc.results['RelationExtraction']}")
        print(f"Raw output: {doc.meta['RelationExtraction']['raw']}")
        print(f"Usage: {doc.meta['RelationExtraction']['usage']}")
        print(f"Total Usage: {doc.meta['usage']}")

        # Verify schema.
        for triplet in res.triplets:
            assert isinstance(triplet, relation_extraction.RelationTriplet)
            assert isinstance(triplet.head, relation_extraction.RelationEntity)
            assert isinstance(triplet.tail, relation_extraction.RelationEntity)
            assert isinstance(triplet.relation, str)
            assert triplet.relation.lower() in ["works_for", "located_in", "founded"]

            # Verify content.
            assert triplet.head.text in doc.text
            assert triplet.tail.text in doc.text

    with pytest.raises(NotImplementedError):
        pipe["RelationExtraction"].distill(None, None, None, None, None, None, None, None)

    if fewshot:
        _to_hf_dataset(task, docs)


@pytest.mark.parametrize("batch_runtime", [ModelType.dspy], indirect=["batch_runtime"])
def test_serialization(relation_extraction_docs, batch_runtime) -> None:
    pipe = Pipeline(
        relation_extraction.RelationExtraction(
            relations=["works_for", "located_in", "founded"],
            model=batch_runtime.model,
            model_settings=batch_runtime.model_settings,
            batch_size=batch_runtime.batch_size,
            entity_types=["PERSON", "ORGANIZATION", "LOCATION"]
        )
    )

    config = pipe.serialize()
    assert config.model_dump() == {
        'cls_name': 'sieves.pipeline.core.Pipeline',
        'tasks': {
            'is_placeholder': False,
            'value': [
                {
                    'cls_name': 'sieves.tasks.predictive.relation_extraction.core.RelationExtraction',
                    'relations': {'is_placeholder': False, 'value': ['works_for', 'located_in', 'founded']},
                    'entity_types': {'is_placeholder': False, 'value': ['PERSON', 'ORGANIZATION', 'LOCATION']},
                    'fewshot_examples': {'is_placeholder': False, 'value': ()},
                    'batch_size': {'is_placeholder': False, 'value': -1},
                    'model_settings': {
                        'is_placeholder': False,
                        'value': {
                            'config_kwargs': None,
                            'inference_kwargs': None,
                            'init_kwargs': None,
                            'strict': True,
                            'inference_mode': None,
                        }
                    },
                    'include_meta': {'is_placeholder': False, 'value': True},
                    'model': {'is_placeholder': True, 'value': 'dspy.clients.lm.LM'},
                    'prompt_instructions': {'is_placeholder': False,
                                          'value': None},
                    'task_id': {'is_placeholder': False, 'value': 'RelationExtraction'},
                    'condition': {'is_placeholder': False, 'value': None},
                    'version': Config.get_version()
                }
            ]
        },
        'use_cache': {'is_placeholder': False, 'value': True},
        'version': Config.get_version()
    }

    Pipeline.deserialize(
        config=config,
        tasks_kwargs=[{"model": batch_runtime.model}],
    )


def _to_hf_dataset(task: relation_extraction.RelationExtraction, docs: list[Doc]) -> None:
    """Tests whether conversion to HF dataset works as expected.

    :param task: RelationExtraction task instance.
    :param docs: List of documents to convert.
    """
    dataset = task.to_hf_dataset(docs)
    assert all([key in dataset.features for key in ("text", "triplets")])
    assert len(dataset) == 2
    dataset_records = list(dataset)
    for rec in dataset_records:
        # Hugging Face datasets stores sequence of dicts as dict of lists (column-oriented)
        triplets = rec["triplets"]
        assert isinstance(triplets, dict)
        assert "head" in triplets
        assert "relation" in triplets
        assert "tail" in triplets

        # Verify consistent lengths
        assert len(triplets["head"]) == len(triplets["relation"]) == len(triplets["tail"])

        if len(triplets["head"]) > 0:
            # Check structure of the first item
            head = triplets["head"][0]
            assert isinstance(head, dict)
            assert "text" in head
            assert "entity_type" in head
            assert isinstance(triplets["relation"][0], str)

        assert isinstance(rec["text"], str)

    with pytest.raises(KeyError):
        task.to_hf_dataset([Doc(text="This is a dummy text.")])


@pytest.mark.parametrize(
    "batch_runtime",
    [ModelType.dspy, ModelType.langchain, ModelType.outlines, ModelType.gliner],
    indirect=["batch_runtime"],
)
def test_inference_mode_override(batch_runtime) -> None:
    """Test that inference_mode parameter overrides the default value."""
    dummy = "dummy_inference_mode"

    task = relation_extraction.RelationExtraction(
        relations=["works_for"],
        model=batch_runtime.model,
        model_settings=ModelSettings(inference_mode=dummy),
        batch_size=batch_runtime.batch_size,
    )

    assert task._bridge.inference_mode == dummy


@pytest.mark.parametrize(
    "batch_runtime",
    relation_extraction.RelationExtraction.supports(),
    indirect=["batch_runtime"],
)
def test_run_with_list_relations(relation_extraction_docs, batch_runtime) -> None:
    """Test RelationExtraction with list format relations."""
    relations_list = ["works_for", "located_in"]

    pipe = Pipeline(
        relation_extraction.RelationExtraction(
            relations=relations_list,
            model=batch_runtime.model,
            model_settings=batch_runtime.model_settings,
            batch_size=batch_runtime.batch_size,
            entity_types=["PERSON", "ORGANIZATION", "LOCATION"]
        )
    )
    docs = list(pipe(relation_extraction_docs))

    assert len(docs) == 2
    for doc in docs:
        assert "RelationExtraction" in doc.results


def test_gliner_warning() -> None:
    """Test that GliNER2 issues a warning when entity_types are provided."""
    batch_runtime_gliner = _make_runtime(model_type=ModelType.gliner, batch_size=-1)
    relations = ["works_for"]

    with pytest.warns(UserWarning, match="GliNER2 backend does not support entity type constraints"):
        relation_extraction.RelationExtraction(
            relations=relations,
            model=batch_runtime_gliner.model,
            entity_types=["PERSON"],
            task_id="re",
        )


@pytest.mark.parametrize("batch_runtime", [ModelType.gliner], indirect=["batch_runtime"])
def test_evaluation(batch_runtime) -> None:
    """Test evaluation for relation extraction without running pipeline."""
    task = relation_extraction.RelationExtraction(
        relations=["founded", "located_in"], model=batch_runtime.model, task_id="re"
    )

    # 1. Full overlap
    doc_full = Doc(text="Henri founded Red Cross in Geneva.")
    res_full = Result(triplets=[
        RelationTriplet(
            head=RelationEntity(text="Henri", entity_type="PERSON"),
            relation="founded",
            tail=RelationEntity(text="Red Cross", entity_type="ORGANIZATION")
        ),
        RelationTriplet(
            head=RelationEntity(text="Red Cross", entity_type="ORGANIZATION"),
            relation="located_in",
            tail=RelationEntity(text="Geneva", entity_type="LOCATION")
        )
    ])
    doc_full.results["re"] = res_full
    doc_full.gold["re"] = res_full
    report_full = task.evaluate([doc_full])
    assert report_full.metrics[task.metric] == 1.0

    # 2. Partial overlap
    doc_partial = Doc(text="Henri founded Red Cross in Geneva.")
    # Pred has both
    res_pred = res_full
    # Gold has only one
    res_gold = Result(triplets=[
        RelationTriplet(
            head=RelationEntity(text="Henri", entity_type="PERSON"),
            relation="founded",
            tail=RelationEntity(text="Red Cross", entity_type="ORGANIZATION")
        )
    ])
    doc_partial.results["re"] = res_pred
    doc_partial.gold["re"] = res_gold
    report_partial = task.evaluate([doc_partial])
    # TP=1 (founded), FP=1 (located_in), FN=0 -> Precision=0.5, Recall=1.0 -> F1=0.666...
    assert 0.6 < report_partial.metrics[task.metric] < 0.7

    # 3. No overlap
    doc_none = Doc(text="Henri founded Red Cross.")
    res_none_pred = Result(triplets=[
        RelationTriplet(
            head=RelationEntity(text="Henri", entity_type="PERSON"),
            relation="founded",
            tail=RelationEntity(text="Red Cross", entity_type="ORGANIZATION")
        )
    ])
    res_none_gold = Result(triplets=[
        RelationTriplet(
            head=RelationEntity(text="Eglantyne", entity_type="PERSON"),
            relation="founded",
            tail=RelationEntity(text="Save the Children", entity_type="ORGANIZATION")
        )
    ])
    doc_none.results["re"] = res_none_pred
    doc_none.gold["re"] = res_none_gold
    report_none = task.evaluate([doc_none])
    assert report_none.metrics[task.metric] == 0.0
