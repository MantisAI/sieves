# mypy: ignore-errors
import os

import dspy

from sieves import Pipeline, engines, tasks


def test_simple_run(dummy_docs) -> None:
    engine_dspy = engines.dspy_.DSPy(model=dspy.LM("claude-3-haiku-20240307", api_key=os.environ["ANTHROPIC_API_KEY"]))
    pipe = Pipeline(
        [
            tasks.predictive.Classification(
                task_id="classifier", labels=["scientific paper", "newspaper article"], engine=engine_dspy
            ),
        ]
    )
    docs = list(pipe(dummy_docs))

    assert len(docs) == 1
    assert docs[0].text
    assert "classifier" in docs[0].results
