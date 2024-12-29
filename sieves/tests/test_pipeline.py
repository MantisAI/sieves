import os

import chonkie
import dspy
import outlines
from tokenizers import tokenizers

from sieves import Doc, Pipeline, engines, tasks


def test_pipeline() -> None:
    engine_outlines = engines.outlines_engine.Outlines(
        model=outlines.models.transformers("HuggingFaceTB/SmolLM-135M-Instruct")
    )
    engine_dspy = engines.dspy_engine.DSPy(
        model=dspy.LM("claude-3-haiku-20240307", api_key=os.environ["ANTHROPIC_API_KEY"])
    )

    all_tasks = [
        tasks.parsers.Docling(),
        tasks.chunkers.Chonkie(chonkie.TokenChunker(tokenizers.Tokenizer.from_pretrained("gpt2"))),
        tasks.predictive.Classification(
            task_id="classifier_outlines", labels=["scientific paper", "newspaper article"], engine=engine_outlines
        ),
        tasks.predictive.Classification(
            task_id="classifier_dspy", labels=["scientific paper", "newspaper article"], engine=engine_dspy
        ),
    ]

    resources = [Doc(uri="https://arxiv.org/pdf/2408.09869")]
    pipe = Pipeline(all_tasks)
    docs = list(pipe(resources))

    assert len(docs) == 1
    assert docs[0].text
    assert docs[0].chunks


if __name__ == "__main__":
    test_pipeline()
