"""Tests for task optimization."""
import os

import dspy
from loguru import logger

from sieves import GenerationSettings
from sieves.engines import EngineType
from sieves.tasks.optimization import Optimizer
from sieves.tasks.predictive import classification


if __name__ == '__main__':
    # model = dspy.LM(
    #     f"openrouter/z-ai/glm-4.5-air",
    #     api_base="https://openrouter.ai/api/v1/",
    #     api_key=os.environ['OPENROUTER_API_KEY']
    # )
    model = dspy.LM("claude-3-haiku-20240307", api_key=os.environ["ANTHROPIC_API_KEY"])

    rf = 'Is a fruit.'
    rv = 'Is a vegetable.'
    rm = 'Is a mushroom.'
    examples = [
        classification.FewshotExampleSingleLabel(text='Apple', reasoning=rf, label='fruit', confidence=1.),
        classification.FewshotExampleSingleLabel(text='Broccoli', reasoning=rv, label='vegetable', confidence=1.),
        classification.FewshotExampleSingleLabel(text='Shiitake', reasoning=rm, label='mushroom', confidence=1.),
        classification.FewshotExampleSingleLabel(text='Melon', reasoning=rf, label='fruit', confidence=1.),
        classification.FewshotExampleSingleLabel(text='Carrot', reasoning=rv, label='vegetable', confidence=1.),
        classification.FewshotExampleSingleLabel(text='Lion\'s Mane', reasoning=rm, label='mushroom', confidence=1.),
        classification.FewshotExampleSingleLabel(text='Tomato', reasoning=rf, label='fruit', confidence=1.),
        classification.FewshotExampleSingleLabel(text='Pepper', reasoning=rv, label='vegetable',
                                                 confidence=1.),
        classification.FewshotExampleSingleLabel(text='Champignon', reasoning=rm, label='mushroom', confidence=1.),
        classification.FewshotExampleSingleLabel(text='Kiwi', reasoning=rf, label='fruit', confidence=1.),
        classification.FewshotExampleSingleLabel(text='Cucumber', reasoning=rv, label='vegetable',
                                                 confidence=1.),
        classification.FewshotExampleSingleLabel(text='Chanterelle', reasoning=rm, label='mushroom', confidence=1.),
        classification.FewshotExampleSingleLabel(text='Pear', reasoning=rf, label='fruit', confidence=1.),
        classification.FewshotExampleSingleLabel(text='Onion', reasoning=rv, label='vegetable',
                                                 confidence=1.),
        classification.FewshotExampleSingleLabel(text='Porcini', reasoning=rm, label='mushroom', confidence=1.),
    ]

    task = classification.Classification(
        multi_label=False,
        labels=["fruit", "vegetable", "mushroom"],
        fewshot_examples=examples,
        model=model,
        generation_settings=GenerationSettings(),
    )

    optimizer = Optimizer(
        model,
        val_frac=.25,
        shuffle=True,
        # dspy_init_kwargs=dict(auto=None, num_candidates=2, max_errors=0),
        dspy_init_kwargs=dict(max_errors=3),
        # dspy_compile_kwargs=dict(
        #     num_trials=1, minibatch=False
        # )
    )
    task.optimize(optimizer)
    print(task._custom_prompt_instructions)
    for ex in task._fewshot_examples:
        print(ex)
    x = 3
