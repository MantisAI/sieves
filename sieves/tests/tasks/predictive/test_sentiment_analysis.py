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
            text="The food was perfect, the service only ok.",
            sentiment_per_aspect={"food": 1.0, "service": 0.5, "overall": 0.8},
        ),
        sentiment_analysis.FewshotExample(
            text="The service was amazing - they take excellent care of their customers. The food was despicable "
            "though, I strongly recommend not to go.",
            sentiment_per_aspect={"food": 0.1, "service": 1.0, "overall": 0.3},
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
