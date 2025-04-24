# mypy: ignore-errors
from pathlib import Path
from tempfile import TemporaryDirectory

import datasets
import model2vec
import model2vec.inference
import model2vec.train
import numpy as np
import pytest
import setfit

from sieves import Pipeline
from sieves.engines import EngineType
from sieves.tasks import Distillation
from sieves.tasks.postprocessing.distillation.types import DistillationFramework
from sieves.tasks.predictive import classification


@pytest.mark.parametrize("batch_engine", (EngineType.dspy,), indirect=["batch_engine"])
# @pytest.mark.parametrize("distillation_framework", DistillationFramework.all(), indirect=["batch_engine"])
@pytest.mark.parametrize("distillation_framework", (DistillationFramework.model2vec,))
def test_run(classification_docs, batch_engine, distillation_framework) -> None:
    label_descriptions = {
        "science": "Topics related to scientific disciplines and research",
        "politics": "Topics related to government, elections, and political systems",
    }
    seed = 42

    with TemporaryDirectory() as tmp_dir:
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
                    # base_model_id="sentence-transformers/paraphrase-mpnet-base-v2",
                    base_model_id="minishlab/potion-base-32M",
                    framework=distillation_framework,
                    train_kwargs={},
                    output_path=Path(tmp_dir),
                    seed=seed,
                ),
            ]
        )
        docs = list(pipe(classification_docs * 4))

        # Ensure equality of saved with original dataset.
        hf_dataset = classifier._split_dataset(classifier.to_hf_dataset(docs), (0.7, 0.15, 0.15), seed)
        hf_dataset_loaded = datasets.DatasetDict.load_from_disk(Path(tmp_dir) / "data")
        for split in ("train", "val", "test"):
            assert hf_dataset_loaded[split].info == hf_dataset[split].info
            assert hf_dataset_loaded[split]["text"] == hf_dataset[split]["text"]
            assert hf_dataset_loaded[split]["labels"] == hf_dataset[split]["labels"]

        test_sents = ["This is about the galaxy and laws of nature.", "This is about political election and lobbying."]

        match distillation_framework:
            case DistillationFramework.setfit:
                model = setfit.SetFitModel.from_pretrained(tmp_dir)
                preds = model.predict(test_sents, as_numpy=True)
                assert preds.shape == (2, 2)

            case DistillationFramework.sentence_transformers:
                raise NotImplementedError

            case DistillationFramework.model2vec:
                model = model2vec.inference.StaticModelPipeline.from_pretrained(tmp_dir)
                preds = model.predict(test_sents)
                assert set(np.unique(preds).tolist()) <= {"science", "politics"}
                assert preds.shape[0] == 2
                assert preds.shape[1] in (1, 2)
