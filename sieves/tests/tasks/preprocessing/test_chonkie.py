# mypy: ignore-errors
import chonkie

from sieves import Doc, Pipeline, tasks


def test_chonkie(tokenizer) -> None:
    resources = [Doc(text="This is a text " * 100)]
    pipe = Pipeline(tasks=[tasks.preprocessing.Chonkie(chonkie.TokenChunker(tokenizer))])
    docs = list(pipe(resources))

    assert len(docs) == 1
    assert docs[0].text
    assert docs[0].chunks
