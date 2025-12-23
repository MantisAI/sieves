# mypy: ignore-errors
import pytest
from sieves import Doc, Pipeline
from sieves.model_wrappers import ModelType
from sieves.tasks.predictive import relation_extraction

@pytest.mark.parametrize(
    "batch_runtime",
    relation_extraction.RelationExtraction.supports(),
    indirect=["batch_runtime"],
)
def test_run(batch_runtime) -> None:
    # --8<-- [start:re-usage]
    from sieves import Doc, Pipeline, tasks

    relations = {
        "works_for": "A person works for a company or organization.",
        "located_in": "A place or organization is located in a city, country, or region.",
    }

    task = tasks.RelationExtraction(
        relations=relations,
        model=batch_runtime.model,
        model_settings=batch_runtime.model_settings,
        batch_size=batch_runtime.batch_size,
        entity_types=["PERSON", "COMPANY", "LOCATION"]
    )

    pipe = Pipeline(task)
    docs = [
        Doc(text="Bill Gates founded Microsoft in Albuquerque, New Mexico."),
        Doc(text="Steve Jobs worked for Apple in Cupertino."),
    ]
    results = list(pipe(docs))
    # --8<-- [end:re-usage]

    assert len(results) == 2
    for doc in results:
        assert "RelationExtraction" in doc.results
        res = doc.results["RelationExtraction"]
        assert isinstance(res, relation_extraction.Result)

        # Verify schema.
        for triplet in res.triplets:
            assert isinstance(triplet, relation_extraction.RelationTriplet)
            assert isinstance(triplet.head, relation_extraction.RelationEntity)
            assert isinstance(triplet.tail, relation_extraction.RelationEntity)
            assert isinstance(triplet.relation, str)
            assert triplet.relation in ["works_for", "located_in", "founded"]

            # Verify spans (if found).
            if triplet.head.start != -1:
                assert doc.text[triplet.head.start:triplet.head.end].lower() == triplet.head.text.lower()
            if triplet.tail.start != -1:
                assert doc.text[triplet.tail.start:triplet.tail.end].lower() == triplet.tail.text.lower()

def test_gliner_warning() -> None:
    """Test that GliNER2 issues a warning when entity_types are provided."""
    from sieves.tests.conftest import _make_runtime
    batch_runtime_gliner = _make_runtime(model_type=ModelType.gliner, batch_size=-1)
    relations = ["works_for"]

    with pytest.warns(UserWarning, match="GliNER2 backend does not support entity type constraints"):
        relation_extraction.RelationExtraction(
            relations=relations,
            model=batch_runtime_gliner.model,
            entity_types=["PERSON"],
            task_id="re"
        )
