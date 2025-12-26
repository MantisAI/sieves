# mypy: ignore-errors
import pytest

from sieves import Doc
from sieves.model_wrappers import ModelType
from sieves.tasks.predictive import SentimentAnalysis, sentiment_analysis
from sieves.tasks.predictive.schemas.sentiment_analysis import Result


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


@pytest.mark.parametrize("batch_runtime", [ModelType.outlines], indirect=["batch_runtime"])
def test_evaluation(batch_runtime) -> None:
    """Test evaluation for sentiment analysis without running pipeline."""
    task = SentimentAnalysis(aspects=["food", "service"], model=batch_runtime.model, task_id="sent")

    # 1. Full overlap
    doc_full = Doc(text="Great food and service.")
    res_full = Result(sentiment_per_aspect={"food": 1.0, "service": 1.0, "overall": 1.0})
    doc_full.results["sent"] = res_full
    doc_full.gold["sent"] = res_full
    report_full = task.evaluate([doc_full])
    assert report_full.metrics[task.metric] == 1.0

    # 2. No overlap (opposite sentiments)
    doc_none = Doc(text="Great food and service.")
    res_none_pred = Result(sentiment_per_aspect={"food": 0.0, "service": 0.0, "overall": 0.0})
    doc_none.results["sent"] = res_none_pred
    doc_none.gold["sent"] = res_full
    report_none = task.evaluate([doc_none])
    assert report_none.metrics[task.metric] == 0.0

    # 3. Partial overlap
    doc_partial = Doc(text="Great food. Bad service.")
    # food is correct (1.0), service is wrong (0.0 vs 1.0)
    # F1 (Macro):
    # food: 1.0 (TP) -> F1 1.0
    # service: 0.0 (FN) -> F1 0.0
    # overall: 1.0 (TP) -> F1 1.0
    # Average: (1.0 + 0.0 + 1.0) / 3 = 0.66...
    res_partial_pred = Result(sentiment_per_aspect={"food": 1.0, "service": 0.0, "overall": 1.0})
    doc_partial.results["sent"] = res_partial_pred
    doc_partial.gold["sent"] = res_full
    report_partial = task.evaluate([doc_partial])
    assert 0.6 < report_partial.metrics[task.metric] < 0.7
