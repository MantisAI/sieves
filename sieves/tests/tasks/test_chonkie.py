import chonkie
import tokenizers

from sieves import Doc, Pipeline, tasks


def test_chunking() -> None:
    resources = [Doc(text="This is a text " * 100)]
    pipe = Pipeline(tasks=[tasks.chunkers.Chonkie(chonkie.TokenChunker(tokenizers.Tokenizer.from_pretrained("gpt2")))])
    docs = list(pipe(resources))

    assert len(docs) == 1
    assert docs[0].text
    assert docs[0].chunks
