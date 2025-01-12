# mypy: ignore-errors
import gliclass
import transformers

from sieves import Pipeline, engines, tasks


def test_simple_run(dummy_docs) -> None:
    pipeline = gliclass.ZeroShotClassificationPipeline(
        gliclass.GLiClassModel.from_pretrained("knowledgator/gliclass-small-v1.0"),
        transformers.AutoTokenizer.from_pretrained("knowledgator/gliclass-small-v1.0"),
        classification_type="multi-label",
        device="cpu",
    )

    engine = engines.glix_.GliX(model=pipeline)
    pipe = Pipeline(
        [
            tasks.predictive.Classification(
                task_id="classifier", labels=["scientific paper", "newspaper article"], engine=engine
            ),
        ]
    )
    docs = list(pipe(dummy_docs))

    assert len(docs) == 2
    for doc in docs:
        assert doc.text
        assert doc.results["classifier"]
        assert "classifier" in doc.results
