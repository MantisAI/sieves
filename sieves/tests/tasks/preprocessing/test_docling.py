import chonkie
import tokenizers

from sieves import Doc, Pipeline, tasks


def test_docling() -> None:
    resources = [Doc(uri="https://arxiv.org/pdf/2408.09869")]
    pipe = Pipeline(
        tasks=[
            tasks.preprocessing.Docling(),
            tasks.preprocessing.Chonkie(
                chonkie.TokenChunker(tokenizers.Tokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct"))
            ),
        ]
    )
    docs = list(pipe(resources))

    assert len(docs) == 1
    assert docs[0].text
