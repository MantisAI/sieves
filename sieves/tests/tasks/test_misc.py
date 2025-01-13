# mypy: ignore-errors
import os

import dspy

from sieves import engines, tasks


def test_custom_prompt_template():
    prompt_template = "This is a different prompt template."
    engine = engines.dspy_.DSPy(model=dspy.LM("claude-3-haiku-20240307", api_key=os.environ["ANTHROPIC_API_KEY"]))
    task = tasks.predictive.Classification(
        task_id="classifier",
        labels=["scientific paper", "newspaper article"],
        engine=engine,
        prompt_template=prompt_template,
    )
    assert task.prompt_template == prompt_template
