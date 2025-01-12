# mypy: ignore-errors

import transformers

from sieves import Pipeline, engines, tasks


def test_simple_run(dummy_docs) -> None:
    model = transformers.pipeline(
        "zero-shot-classification", model="MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33"
    )
    engine = engines.huggingface_.HuggingFace(model=model)
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
