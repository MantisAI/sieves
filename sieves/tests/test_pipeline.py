# mypy: ignore-errors
import os

import chonkie
import dspy
import outlines
import pytest
from tokenizers import tokenizers

from sieves import Doc, Pipeline, engines, tasks


@pytest.fixture(scope="session")
def preprocessed_docs() -> list[Doc]:
    resources = [Doc(uri="https://arxiv.org/pdf/2408.09869")]
    pipe = Pipeline(
        tasks=[
            tasks.parsers.Docling(),
            tasks.chunkers.Chonkie(chonkie.TokenChunker(tokenizers.Tokenizer.from_pretrained("gpt2"))),
        ]
    )
    return list(pipe(resources))


def test_pipeline_outlines(preprocessed_docs) -> None:
    outlines_model_name = "gpt2"
    engine_outlines = engines.outlines_engine.Outlines(model=outlines.models.transformers(outlines_model_name))
    pipe = Pipeline(
        [
            tasks.predictive.Classification(
                task_id="classifier_outlines", labels=["scientific paper", "newspaper article"], engine=engine_outlines
            ),
        ]
    )
    docs = list(pipe(preprocessed_docs))

    assert len(docs) == 1
    assert docs[0].text
    assert docs[0].chunks
    assert "classifier_outlines" in docs[0].results


def test_pipeline_dspy(preprocessed_docs) -> None:
    engine_dspy = engines.dspy_engine.DSPy(
        model=dspy.LM("claude-3-haiku-20240307", api_key=os.environ["ANTHROPIC_API_KEY"])
    )
    pipe = Pipeline(
        [
            tasks.predictive.Classification(
                task_id="classifier_dspy", labels=["scientific paper", "newspaper article"], engine=engine_dspy
            ),
        ]
    )
    docs = list(pipe(preprocessed_docs))

    assert len(docs) == 1
    assert docs[0].text
    assert docs[0].chunks
    assert "classifier_dspy" in docs[0].results
