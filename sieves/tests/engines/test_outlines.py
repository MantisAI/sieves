# mypy: ignore-errors

import outlines

from sieves import Pipeline, engines, tasks


def test_simple_run(dummy_docs) -> None:
    model_name = "gpt2"
    engine = engines.outlines_.Outlines(model=outlines.models.transformers(model_name))
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
