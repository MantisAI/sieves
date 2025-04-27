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

from sieves import Doc, Pipeline
from sieves.engines import EngineType
from sieves.tasks import Distillation, DistillationFramework
from sieves.tasks.predictive import classification


@pytest.mark.parametrize("batch_engine", (EngineType.huggingface,), indirect=["batch_engine"])
@pytest.mark.parametrize("distillation_framework", DistillationFramework.all())
def test_run(batch_engine, distillation_framework) -> None:
    label_descriptions = {
        "science": "Topics related to scientific disciplines and research",
        "politics": "Topics related to government, elections, and political systems",
    }
    seed = 42
    base_model_id = "sentence-transformers/paraphrase-mpnet-base-v2"
    if distillation_framework == DistillationFramework.model2vec:
        base_model_id = "minishlab/potion-base-32M"

    science_text = (
        "Scientists report that plasma is a state of matter. They published an academic paper. This is about science -"
        " scientists, papers, experiments, laws of nature."
    )
    politics_text = (
        "A new law has been passed. The opposition doesn't support it, but parliament has voted on it. This is about "
        "politics - parliament, laws, parties, politicians."
    )
    docs = [
        *[Doc(text=f"{i}. {science_text}") for i in range(5)],
        *[Doc(text=f"{i}. {politics_text}") for i in range(5)],
    ]

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
                    base_model_id=base_model_id,
                    framework=distillation_framework,
                    output_path=Path(tmp_dir),
                    train_frac=0.5,
                    val_frac=0.5,
                    seed=seed,
                ),
            ],
        )

        if distillation_framework == DistillationFramework.sentence_transformers:
            with pytest.raises(NotImplementedError):
                list(pipe(docs))
            return
        else:
            docs = list(pipe(docs))

        # Ensure equality of saved with original dataset.
        hf_dataset = classifier.to_hf_dataset(docs)
        hf_dataset = classifier._split_dataset(hf_dataset, 0.5, 0.5, seed)
        hf_dataset_loaded = datasets.DatasetDict.load_from_disk(Path(tmp_dir) / "data")
        for split in ("train", "val"):
            assert hf_dataset_loaded[split].info == hf_dataset[split].info
            assert hf_dataset_loaded[split]["text"] == hf_dataset[split]["text"]
            assert hf_dataset_loaded[split]["labels"] == hf_dataset[split]["labels"]

        # Assert predictions of distilled models look as expected.

        test_sents = ["This is about the galaxy and laws of nature.", "This is about political election and lobbying."]

        match distillation_framework:
            case DistillationFramework.setfit:
                model = setfit.SetFitModel.from_pretrained(tmp_dir)
                preds = model.predict(test_sents, as_numpy=True)
                assert preds.shape == (2, 2)

            case DistillationFramework.model2vec:
                model = model2vec.inference.StaticModelPipeline.from_pretrained(tmp_dir)
                preds = model.predict(test_sents)
                assert set(np.unique(preds).tolist()) <= {"science", "politics"}
                assert preds.shape[0] == 2
                assert preds.shape[1] in (1, 2)
