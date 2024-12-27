import chonkie
import tokenizers

from sieves import Pipeline, tasks
from sieves.data import Doc


def test_pipeline() -> None:
    all_tasks = [
        tasks.parsers.Docling(),
        tasks.chunkers.Chonkie(chonkie.TokenChunker(tokenizers.Tokenizer.from_pretrained("gpt2"))),
    ]

    resources = [Doc(uri="https://arxiv.org/pdf/2408.09869")]
    pipe = Pipeline(all_tasks)
    docs = list(pipe(resources))

    assert len(docs) == 1
    assert docs[0].text
    assert docs[0].chunks


if __name__ == "__main__":
    test_pipeline()
