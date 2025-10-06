import os

import dspy

from sieves import GenerationSettings
from sieves.tasks.optimization import Optimizer
from sieves.tasks.predictive import classification

if __name__ == '__main__':
    model = dspy.LM("claude-3-haiku-20240307", api_key=os.environ["ANTHROPIC_API_KEY"])

    rf = 'Is a fruit.'
    rv = 'Is a vegetable.'
    examples = [
        classification.FewshotExampleSingleLabel(text='Apple', reasoning=rf, label='fruit', confidence=1.),
        classification.FewshotExampleSingleLabel(text='Broccoli', reasoning=rv, label='vegetable', confidence=1.),
        classification.FewshotExampleSingleLabel(text='Melon', reasoning=rf, label='fruit', confidence=1.),
        classification.FewshotExampleSingleLabel(text='Carrot', reasoning=rv, label='vegetable', confidence=1.),
        classification.FewshotExampleSingleLabel(text='Tomato', reasoning=rf, label='fruit', confidence=1.),
        classification.FewshotExampleSingleLabel(text='Pepper', reasoning=rv, label='vegetable',
                                                 confidence=1.),
        classification.FewshotExampleSingleLabel(text='Kiwi', reasoning=rf, label='fruit', confidence=1.),
        classification.FewshotExampleSingleLabel(text='Cucumber', reasoning=rv, label='vegetable',
                                                 confidence=1.),
        classification.FewshotExampleSingleLabel(text='Pear', reasoning=rf, label='fruit', confidence=1.),
        classification.FewshotExampleSingleLabel(text='Onion', reasoning=rv, label='vegetable',
                                                 confidence=1.),
    ]

    task = classification.Classification(
        multi_label=False,
        labels=["fruit", "vegetable"],
        fewshot_examples=examples,
        model=model,
        generation_settings=GenerationSettings(),
    )

    optimizer = Optimizer(
        model,
        val_frac=.25,
        shuffle=True,
        dspy_init_kwargs=dict(auto=None, num_candidates=2, max_errors=0),
        dspy_compile_kwargs=dict(
            num_trials=1, minibatch=False, requires_permission_to_run=False
        )
    )
    task.optimize(optimizer)
