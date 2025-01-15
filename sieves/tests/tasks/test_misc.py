# mypy: ignore-errors
import os

import chonkie
import dspy
import pytest
import tokenizers

from sieves import Doc, Pipeline, engines, tasks


def test_custom_prompt_template():
    prompt_template = "This is a different prompt template."
    engine = engines.dspy_.DSPy(model=dspy.LM("claude-3-haiku-20240307", api_key=os.environ["ANTHROPIC_API_KEY"]))
    task = tasks.predictive.Classification(
        task_id="classifier",
        labels=["science", "politics"],
        engine=engine,
        prompt_template=prompt_template,
    )
    assert task.prompt_template == prompt_template


@pytest.mark.slow
@pytest.mark.parametrize(
    "engine",
    [engines.EngineType.outlines],
    indirect=True,
)
def test_run_readme_example_short(engine):
    # Define documents by text or URI.
    docs = [Doc(text="Special relativity applies to all physical phenomena in the absence of gravity.")]

    # Create pipeline with tasks.
    pipe = Pipeline(
        [
            # Run classification on provided document.
            tasks.predictive.Classification(labels=["science", "politics"], engine=engine),
        ]
    )

    # Run pipe and output results.
    docs = list(pipe(docs))
    print(docs[0].results["Classification"])


@pytest.mark.slow
@pytest.mark.parametrize(
    "engine",
    [engines.EngineType.glix],
    indirect=True,
)
def test_run_readme_example_long(engine):
    # Define documents by text or URI.
    docs = [Doc(uri="https://arxiv.org/pdf/2408.09869")]

    # Create engine responsible for generating structured output.
    model_name = "knowledgator/gliclass-small-v1.0"

    # Create pipeline with tasks.
    pipe = Pipeline(
        [
            # Add document parsing task.
            tasks.parsers.Docling(),
            # Add chunking task to ensure we don't exceed our model's context window.
            tasks.chunkers.Chonkie(chonkie.TokenChunker(tokenizers.Tokenizer.from_pretrained(model_name))),
            # Run classification on provided document.
            tasks.predictive.Classification(labels=["science", "politics"], engine=engine),
        ]
    )

    # Run pipe and output results.
    docs = list(pipe(docs))
    print(docs[0].results["Classification"])
