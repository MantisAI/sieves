# mypy: ignore-errors
import os
import pickle
import tempfile

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


def test_custom_prompt_signature_desc():
    prompt_sig_desc = "This is a different prompt signature description."
    engine = engines.dspy_.DSPy(model=dspy.LM("claude-3-haiku-20240307", api_key=os.environ["ANTHROPIC_API_KEY"]))
    task = tasks.predictive.Classification(
        task_id="classifier",
        labels=["science", "politics"],
        engine=engine,
        prompt_signature_desc=prompt_sig_desc,
    )
    assert task.prompt_signature_description == prompt_sig_desc


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

    # Serialize pipeline and docs.
    with tempfile.NamedTemporaryFile(suffix=".yml") as tmp_pipeline_file:
        with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp_docs_file:
            pipe.dump(tmp_pipeline_file.name)
            with open(tmp_docs_file.name, "wb") as f:
                pickle.dump(docs, f)

            # To load a pipeline and docs from disk:
            loaded_pipe = Pipeline.load(
                tmp_pipeline_file.name,
                (
                    {"doc_converter": None},
                    {"chunker": chonkie.TokenChunker(tokenizers.Tokenizer.from_pretrained("gpt2"))},
                    {"engine": {"model": engine.model}},
                ),
            )

            pipe_config = pipe.serialize().model_dump()
            assert pipe_config["tasks"]["value"][2]["fewshot_examples"]["value"] == ()
            pipe_config["tasks"]["value"][2]["fewshot_examples"]["value"] = []
            assert loaded_pipe.serialize().model_dump() == pipe_config

            with open(tmp_docs_file.name, "rb") as f:
                loaded_docs = pickle.load(f)
            assert len(loaded_docs) == len(docs)
            assert all([(ld == d for ld, d in zip(loaded_docs, docs))])
