# mypy: ignore-errors
from pathlib import Path
from tempfile import TemporaryDirectory

import datasets
import pytest
import setfit

from sieves import Pipeline
from sieves.engines import EngineType
from sieves.tasks import Distillation
from sieves.tasks.postprocessing.distillation.types import DistillationFramework
from sieves.tasks.predictive import classification


@pytest.mark.parametrize("batch_engine", (EngineType.dspy,), indirect=["batch_engine"])
# @pytest.mark.parametrize("distillation_framework", DistillationFramework.all(), indirect=["batch_engine"])
@pytest.mark.parametrize("distillation_framework", (DistillationFramework.setfit,))
def test_run(classification_docs, batch_engine, distillation_framework) -> None:
    label_descriptions = {
        "science": "Topics related to scientific disciplines and research",
        "politics": "Topics related to government, elections, and political systems",
    }
    seed = 42

    with TemporaryDirectory() as temp_dir:
        classifier = classification.Classification(
            task_id="classifier",
            labels=["science", "politics"],
            engine=batch_engine,
            label_descriptions=label_descriptions,
        )
        pipe = Pipeline(
            [
                classifier,
                Distillation(
                    target_task_id="classifier",
                    # "minishlab/potion-base-32M"
                    base_model_id="sentence-transformers/paraphrase-mpnet-base-v2",
                    framework=distillation_framework,
                    train_kwargs={},
                    output_path=Path(temp_dir),
                    seed=seed,
                ),
            ]
        )
        docs = list(pipe(classification_docs * 4))
        hf_dataset = classifier._split_dataset(classifier.to_hf_dataset(docs), (0.7, 0.15, 0.15), seed)

        match distillation_framework:
            case DistillationFramework.setfit:
                model = setfit.SetFitModel.from_pretrained(temp_dir)
                preds = model.predict(
                    ["This is about the galaxy and nuclear fusion.", "This is about political election and lobbying."],
                    as_numpy=True,
                )
                assert preds.shape == (2, 2)

                hf_dataset_loaded = datasets.DatasetDict.load_from_disk(Path(temp_dir) / "data")
                for split in ("train", "val", "test"):
                    assert hf_dataset_loaded[split].info == hf_dataset[split].info
                    assert hf_dataset_loaded[split]["text"] == hf_dataset[split]["text"]
                    assert hf_dataset_loaded[split]["labels"] == hf_dataset[split]["labels"]

            case DistillationFramework.sentence_transformers:
                raise NotImplementedError

            case DistillationFramework.model2vec:
                raise NotImplementedError
