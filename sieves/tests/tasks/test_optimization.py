"""Tests for task optimization."""
import os
from functools import cache

import dspy
from loguru import logger

from sieves import GenerationSettings
from sieves.engines import EngineType
from sieves.tasks.optimization import Optimizer
from sieves.tasks.predictive import classification


@cache
def _model() -> dspy.LM:
    """Return model to use for optimization.

    :return dspy.LM: Model to use for optimization.
    """
    model = dspy.LM(
        f"openai/gpt-4.1-nano",
        api_base="https://openrouter.ai/api/v1/",
        api_key=os.environ['OPENROUTER_API_KEY']
    )
    # model = dspy.LM("claude-3-haiku-20240307", api_key=os.environ["ANTHROPIC_API_KEY"])

    return model


def test_optimization_classification() -> None:
    """Tests optimization for classification tasks."""
    model = _model()

    rf = 'Is a fruit.'
    rv = 'Is a vegetable.'
    examples_single_label = [
        classification.FewshotExampleSingleLabel(text='Apple', reasoning=rf, label='fruit', confidence=1.),
        classification.FewshotExampleSingleLabel(text='Broccoli', reasoning=rv, label='vegetable', confidence=1.),
        classification.FewshotExampleSingleLabel(text='Melon', reasoning=rf, label='fruit', confidence=1.),
        classification.FewshotExampleSingleLabel(text='Carrot', reasoning=rv, label='vegetable', confidence=1.),
        classification.FewshotExampleSingleLabel(text='Tomato', reasoning=rf, label='fruit', confidence=1.),
        classification.FewshotExampleSingleLabel(text='Pepper', reasoning=rv, label='vegetable',
                                                 confidence=1.),
        classification.FewshotExampleSingleLabel(text='Kiwi', reasoning=rf, label='fruit', confidence=1.),
        classification.FewshotExampleSingleLabel(text='Onion', reasoning=rv, label='vegetable',
                                                 confidence=1.),
    ]
    examples_multi_label = [
        classification.FewshotExampleMultiLabel(
            text='Ghostbusters',
            reasoning='A group of scientists battle paranormal entities with humor and sci-fi equipment.',
            confidence_per_label={'comedy': 0.9, 'scifi': 0.8}
        ),
        classification.FewshotExampleMultiLabel(
            text='The Martian',
            reasoning='A stranded astronaut uses science and wit to survive on Mars with comedic moments.',
            confidence_per_label={'comedy': 0.4, 'scifi': 1.0}
        ),
        classification.FewshotExampleMultiLabel(
            text='Galaxy Quest',
            reasoning='A comedy about actors from a sci-fi show who encounter real aliens.',
            confidence_per_label={'comedy': 1.0, 'scifi': 0.9}
        ),
        classification.FewshotExampleMultiLabel(
            text='Back to the Future',
            reasoning='Time travel adventure with comedic family dynamics and sci-fi concepts.',
            confidence_per_label={'comedy': 0.8, 'scifi': 0.9}
        ),
        classification.FewshotExampleMultiLabel(
            text='Superbad',
            reasoning='A pure comedy about high school friends, no science fiction elements.',
            confidence_per_label={'comedy': 1.0, 'scifi': 0.0}
        ),
        classification.FewshotExampleMultiLabel(
            text='Blade Runner 2049',
            reasoning='A serious dystopian sci-fi film with minimal comedic elements.',
            confidence_per_label={'comedy': 0.05, 'scifi': 1.0}
        ),
        classification.FewshotExampleMultiLabel(
            text='Guardians of the Galaxy',
            reasoning='Space adventure with humor, aliens, and sci-fi action.',
            confidence_per_label={'comedy': 0.75, 'scifi': 0.9}
        ),
        classification.FewshotExampleMultiLabel(
            text='Interstellar',
            reasoning='Hard science fiction about space exploration with emotional but not comedic tone.',
            confidence_per_label={'comedy': 0.05, 'scifi': 1.0}
        ),
    ]

    task_single_label = classification.Classification(
        multi_label=False,
        labels=["fruit", "vegetable"],
        fewshot_examples=examples_single_label,
        model=model,
        generation_settings=GenerationSettings(),
    )
    task_multi_label = classification.Classification(
        multi_label=True,
        labels=["comedy", "scifi"],
        fewshot_examples=examples_multi_label,
        model=model,
        generation_settings=GenerationSettings(),
    )

    optimizer = Optimizer(
        model,
        val_frac=.25,
        shuffle=True,
        dspy_init_kwargs=dict(auto=None, num_candidates=2, max_errors=3),
        dspy_compile_kwargs=dict(num_trials=1, minibatch=False)
    )

    # Test evaluation.
    assert task_single_label._evaluate_optimization_example(
        truth=dspy.Example(text="", reasoning="", label="fruit", confidence=.7),
        pred=dspy.Prediction(text="", reasoning="", label="fruit", confidence=.1),
    ) == .4
    assert task_single_label._evaluate_optimization_example(
        truth=dspy.Example(text="", reasoning="", label="fruit", confidence=.7),
        pred=dspy.Prediction(text="", reasoning="", label="vegetable", confidence=.1),
    ) == 0
    assert task_multi_label._evaluate_optimization_example(
        truth=dspy.Example(text="", reasoning="", confidence_per_label={"comedy": .4, "scifi": .2}),
        pred=dspy.Prediction(text="", reasoning="", confidence_per_label={"comedy": .1, "scifi": .3}),
    ) == .8
    assert task_multi_label._evaluate_optimization_example(
        truth=dspy.Example(text="", reasoning="", confidence_per_label={"comedy": .4, "scifi": .2}),
        pred=dspy.Prediction(text="", reasoning="", confidence_per_label={"comedy": .4, "scifi": .2}),
    ) == 1

    # Smoke-test optimization.
    best_prompt, best_examples = task_single_label.optimize(optimizer, verbose=False)
    assert task_single_label._custom_prompt_instructions == best_prompt
    assert task_single_label._bridge._prompt_instructions == best_prompt
    assert isinstance(task_single_label._fewshot_examples, list)

    best_prompt, best_examples = task_multi_label.optimize(optimizer, verbose=False)
    assert task_multi_label._custom_prompt_instructions == best_prompt
    assert task_multi_label._bridge._prompt_instructions == best_prompt
    assert isinstance(task_multi_label._fewshot_examples, list)
