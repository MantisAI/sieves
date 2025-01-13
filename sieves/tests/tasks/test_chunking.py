# mypy: ignore-errors
import chonkie
import tokenizers
import transformers

from sieves import Doc, Pipeline, engines, tasks


def test_chonkie() -> None:
    resources = [Doc(text="This is a text " * 100)]
    pipe = Pipeline(tasks=[tasks.chunkers.Chonkie(chonkie.TokenChunker(tokenizers.Tokenizer.from_pretrained("gpt2")))])
    docs = list(pipe(resources))

    assert len(docs) == 1
    assert docs[0].text
    assert docs[0].chunks


def test_task_chunking(dummy_docs) -> None:
    """Tests whether chunking mechanism in PredictiveTask works as expected."""
    model = transformers.pipeline(
        "zero-shot-classification", model="MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33"
    )
    engine = engines.huggingface_.HuggingFace(model=model)

    chunk_interval = 5
    pipe = Pipeline(
        [
            tasks.chunkers.NaiveChunker(interval=chunk_interval),
            tasks.predictive.Classification(task_id="classifier", labels=["science", "politics"], engine=engine),
        ]
    )
    docs = list(pipe(dummy_docs))

    assert len(docs) == 2
    for doc in docs:
        assert doc.text
        assert doc.results["classifier"]
        assert len(doc.chunks) == 2
        assert "classifier" in doc.results
