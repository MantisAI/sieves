"""
Test file containing examples for the Optimization guide.

These code blocks are referenced in docs/guides/optimization.md using snippet injection.

IMPORTANT: These tests use MINIMAL settings to reduce cost:
- num_trials=1 (instead of default 30)
- num_candidates=2 (instead of default 10)
- Only 4 training examples (instead of 20+)
"""

import pytest


def test_basic_optimization_example(small_dspy_model):
    """Test basic classification optimization with minimal settings."""
    model = small_dspy_model

    # --8<-- [start:optimization-classification-basic]
    # --8<-- [start:optimization-imports]
    import dspy
    from sieves import tasks, Doc
    from sieves.model_wrappers.utils import ModelSettings
    from sieves.tasks import Optimizer
    from sieves.tasks.predictive.classification import FewshotExampleSingleLabel
    # --8<-- [end:optimization-imports]

    # --8<-- [start:optimization-training-data]
    # 1. Create minimal training data (only 4 examples for speed)
    examples = [
        FewshotExampleSingleLabel(
            text="New smartphone released",
            label="technology",
            confidence=1.0
        ),
        FewshotExampleSingleLabel(
            text="Senate votes on bill",
            label="politics",
            confidence=1.0
        ),
        FewshotExampleSingleLabel(
            text="Football match results",
            label="sports",
            confidence=1.0
        ),
        FewshotExampleSingleLabel(
            text="Software update available",
            label="technology",
            confidence=1.0
        ),
    ]
    # --8<-- [end:optimization-training-data]

    # --8<-- [start:optimization-task-setup]
    # 2. Define task with few-shot examples
    task = tasks.Classification(
        labels={
            "technology": "Technology news, AI, software, and digital innovations",
            "politics": "Political events, elections, and government affairs",
            "sports": "Sports news, games, athletes, and competitions"
        },
        model=model,
        fewshot_examples=examples,
        multi_label=False,
        model_settings=ModelSettings(),
    )
    # --8<-- [end:optimization-task-setup]

    # --8<-- [start:optimization-optimizer-config]
    # 3. Create optimizer with MINIMAL settings for cost efficiency
    optimizer = Optimizer(
        model=model,
        val_frac=0.25,              # Use 25% for validation (1 example)
        seed=42,
        shuffle=True,
        dspy_init_kwargs=dict(
            auto=None,                  # Disable auto mode to use manual settings
            num_candidates=2,           # Minimal candidates (instead of 10)
            max_bootstrapped_demos=1,   # Minimal bootstrapped demos
            max_labeled_demos=1,        # Minimal labeled demos
            max_errors=10,              # Max errors before stopping
            num_threads=1               # Single thread
        ),
        dspy_compile_kwargs=dict(
            num_trials=1,               # Only 1 trial (instead of 30)
            minibatch=False
        )
    )
    # --8<-- [end:optimization-optimizer-config]

    # --8<-- [start:optimization-run]
    # 4. Run optimization
    best_prompt, best_examples = task.optimize(optimizer, verbose=True)

    print(f"Optimized prompt: {best_prompt}")
    print(f"Number of selected examples: {len(best_examples)}")
    # --8<-- [end:optimization-run]
    # --8<-- [end:optimization-classification-basic]

    assert best_prompt is not None
