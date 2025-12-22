# mypy: ignore-errors
import pytest
from sieves import Doc, Pipeline
from sieves.model_wrappers import ModelType
from sieves.tasks import Classification

@pytest.mark.parametrize("include_meta", [True, False])
@pytest.mark.parametrize("runtime", [ModelType.gliner], indirect=["runtime"])
def test_include_meta_toggle(classification_docs, include_meta, runtime):
    """Test whether the include_meta flag correctly toggles raw result population."""
    docs = [Doc(text=d.text) for d in classification_docs]
    task_id = "toggle_task"

    task = Classification(
        labels=["science", "politics"],
        task_id=task_id,
        model=runtime.model,
        model_settings=runtime.model_settings,
        include_meta=include_meta,
        batch_size=runtime.batch_size,
    )

    pipe = Pipeline(task)
    results = list(pipe(docs))

    for doc in results:
        print(f"Output: {doc.results[task_id]}")
        if include_meta:
            assert task_id in doc.meta
            assert "raw" in doc.meta[task_id]
            assert isinstance(doc.meta[task_id]["raw"], list)
            assert len(doc.meta[task_id]["raw"]) > 0
            print(f"Raw output: {doc.meta[task_id]['raw']}")
            print(f"Task Usage: {doc.meta[task_id]['usage']}")
            print(f"Total Usage: {doc.meta['usage']}")
        else:
            assert task_id not in doc.meta
