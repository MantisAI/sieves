"""Test engine-related edge cases, e.g. specific model-task combinations or engine behavior under certain conditions."""
import os

import openai
import outlines

from sieves import tasks, GenerationSettings, Pipeline, Doc


def test_openai_outlines_batching() -> None:
    """Test Outlines batching fallback uses async batching."""
    model = "gpt-5-nano"
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    model = outlines.from_openai(client, model_name=model)

    classifier = tasks.Classification(
        task_id="procedures_classifier",
        labels=["Fruit", "Vegetable"],
        model=model,
        generation_settings=GenerationSettings(
            batch_size=10,
            strict_mode=False,
            inference_kwargs={"max_tokens": 200},
        ),
        multi_label=True,
    )

    docs = [Doc(text="Apple"), Doc(text="Carrot"), Doc(text="Watermelon")]
    list(Pipeline(classifier)(docs))
