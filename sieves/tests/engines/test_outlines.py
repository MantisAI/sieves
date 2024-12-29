# mypy: ignore-errors

import outlines

from sieves import Pipeline, engines, tasks


def test_simple_run(dummy_docs) -> None:
    outlines_model_name = "gpt2"
    engine_outlines = engines.outlines_.Outlines(model=outlines.models.transformers(outlines_model_name))
    pipe = Pipeline(
        [
            tasks.predictive.Classification(
                task_id="classifier", labels=["scientific paper", "newspaper article"], engine=engine_outlines
            ),
        ]
    )
    docs = list(pipe(dummy_docs))

    assert len(docs) == 1
    assert docs[0].text
    assert "classifier" in docs[0].results
