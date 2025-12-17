"""
Test file containing examples for the Distillation guide.

These code blocks are referenced in docs/guides/distillation.md using snippet injection.

IMPORTANT: These tests use MINIMAL settings to reduce cost and time:
- num_epochs=1 (instead of default 3-5)
- Only 9 training examples (instead of 100+)
- Minimal batch sizes
"""

import pytest
from tempfile import TemporaryDirectory
from pathlib import Path


def test_basic_setfit_distillation(small_dspy_model):
    """Test basic SetFit distillation with minimal settings."""
    model = small_dspy_model

    # --8<-- [start:distillation-setfit-basic]
    # --8<-- [start:distillation-setfit-imports]
    import dspy
    from sieves import tasks, Doc, Pipeline
    from sieves.tasks import DistillationFramework
    from setfit import SetFitModel
    # --8<-- [end:distillation-setfit-imports]

    # --8<-- [start:distillation-setfit-data]
    # 1. Create MINIMAL training data (9 examples - 3 per label)
    docs = [
        Doc(text="New AI model released", meta={"label": "technology"}),
        Doc(text="Software update announced", meta={"label": "technology"}),
        Doc(text="Smartphone innovation unveiled", meta={"label": "technology"}),
        Doc(text="Election results in", meta={"label": "politics"}),
        Doc(text="Parliament votes on bill", meta={"label": "politics"}),
        Doc(text="Political debate tonight", meta={"label": "politics"}),
        Doc(text="Team wins game", meta={"label": "sports"}),
        Doc(text="Championship final scores", meta={"label": "sports"}),
        Doc(text="Olympic athlete breaks record", meta={"label": "sports"}),
    ]
    # --8<-- [end:distillation-setfit-data]

    # --8<-- [start:distillation-setfit-teacher]
    # 2. Define teacher task
    teacher = tasks.Classification(
        labels=["technology", "politics", "sports"],
        model=model,
    )

    # 3. Process documents to generate labels
    pipeline = Pipeline([teacher])
    labeled_docs = list(pipeline(docs))
    # --8<-- [end:distillation-setfit-teacher]

    # --8<-- [start:distillation-setfit-distill]
    # 4. Distill with MINIMAL training settings
    with TemporaryDirectory() as tmp_dir:
        teacher.distill(
            base_model_id="sentence-transformers/paraphrase-albert-small-v2",
            framework=DistillationFramework.setfit,
            data=labeled_docs,
            output_path=tmp_dir,
            val_frac=0.3,
            seed=42,
            train_kwargs={
                "num_iterations": 1,    # Min. iterations.
                "num_epochs": 1,        # Minimal epochs (instead of 3-5)
                "batch_size": 8,
            },
        )
        # --8<-- [end:distillation-setfit-distill]

        # --8<-- [start:distillation-setfit-load]
        # 5. Load and use distilled model
        distilled_model = SetFitModel.from_pretrained(tmp_dir)
        predictions = distilled_model.predict(["Smartphone announcement"])
        print(f"Predictions: {predictions}")
        # --8<-- [end:distillation-setfit-load]
    # --8<-- [end:distillation-setfit-basic]

        assert predictions is not None


def test_to_hf_dataset_export(small_dspy_model):
    """Test exporting task results to HuggingFace dataset."""
    model = small_dspy_model

    # --8<-- [start:distillation-to-hf-dataset]
    import dspy
    from sieves import tasks, Doc, Pipeline

    # 1. Create minimal training data
    docs = [
        Doc(text="AI breakthrough announced", meta={"label": "technology"}),
        Doc(text="Election campaign begins", meta={"label": "politics"}),
        Doc(text="Sports team victory", meta={"label": "sports"}),
    ]

    # 2. Run classification task
    task = tasks.Classification(
        labels=["technology", "politics", "sports"],
        model=model,
    )

    pipeline = Pipeline([task])
    labeled_docs = list(pipeline(docs))

    # 3. Export to HuggingFace dataset
    hf_dataset = task.to_hf_dataset(labeled_docs)

    print(f"Dataset features: {hf_dataset.features}")
    print(f"Dataset size: {len(hf_dataset)}")
    # --8<-- [end:distillation-to-hf-dataset]

    assert len(hf_dataset) == 3
    assert "text" in hf_dataset.features
    assert "labels" in hf_dataset.features
