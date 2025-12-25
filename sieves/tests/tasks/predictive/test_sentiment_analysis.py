# mypy: ignore-errors
import pytest

from sieves.tasks.predictive import SentimentAnalysis, sentiment_analysis


@pytest.mark.parametrize(
    "batch_runtime",
    SentimentAnalysis.supports(),
    indirect=["batch_runtime"],
)
@pytest.mark.parametrize("fewshot", [True, False])
def test_run(sentiment_analysis_docs, batch_runtime, fewshot):
    fewshot_examples = [
        sentiment_analysis.FewshotExample(
            text="Beautiful dishes, haven't eaten so well in a long time.",
            sentiment_per_aspect={"overall": 1.0, "food": 1.0, "service": 0.5},
            score=1.0,
        ),
        sentiment_analysis.FewshotExample(
            text="Horrible place. Service is unfriendly, food overpriced and bland.",
            sentiment_per_aspect={"overall": 0.0, "food": 0.0, "service": 0.0},
            score=1.0,
        ),
    ]

    task = SentimentAnalysis(
        aspects=["food", "service", "overall"],
        model=batch_runtime.model,
        model_settings=batch_runtime.model_settings,
        fewshot_examples=fewshot_examples if fewshot else [],
        batch_size=batch_runtime.batch_size,
    )

    results = list(task(sentiment_analysis_docs))

    assert len(results) == 2
    for doc in results:
        assert "SentimentAnalysis" in doc.results
        res = doc.results["SentimentAnalysis"]

        assert isinstance(res, sentiment_analysis.Result)
        assert len(res.sentiment_per_aspect) == 3
        for aspect, score in res.sentiment_per_aspect.items():
            assert aspect in ["food", "service", "overall"]
            assert isinstance(score, float)

        print(f"Output: {doc.results['SentimentAnalysis']}")
        print(f"Raw output: {doc.meta['SentimentAnalysis']['raw']}")
        print(f"Usage: {doc.meta['SentimentAnalysis']['usage']}")
        print(f"Total Usage: {doc.meta['usage']}")
