import chonkie
import outlines
import tokenizers

from sieves import Doc, Pipeline, engines, tasks


def test_pipeline() -> None:
    engine = engines.outlines_engine.Outlines(outlines.models.transformers("gpt2"))
    all_tasks = [
        tasks.parsers.Docling(),
        tasks.chunkers.Chonkie(chonkie.TokenChunker(tokenizers.Tokenizer.from_pretrained("gpt2"))),
        tasks.predictive.Classification(labels=["scientific paper", "newspaper article"], engine=engine),
    ]

    resources = [Doc(uri="https://arxiv.org/pdf/2408.09869")]
    pipe = Pipeline(all_tasks)
    docs = list(pipe(resources))

    assert len(docs) == 1
    assert docs[0].text
    assert docs[0].chunks


if __name__ == "__main__":
    test_pipeline()
