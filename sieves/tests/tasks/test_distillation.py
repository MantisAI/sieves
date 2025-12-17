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
from sieves.model_wrappers import ModelType
from sieves.serialization import Config
from sieves.tasks import DistillationFramework
from sieves.tasks.predictive import classification


def _get_docs() -> list[Doc]:
    science_text = (
        "Scientists report that plasma is a state of matter. They published an academic paper. This is about science -"
        " scientists, papers, experiments, laws of nature."
    )
    politics_text = (
        "A new law has been passed. The opposition doesn't support it, but parliament has voted on it. This is about "
        "politics - parliament, laws, parties, politicians."
    )

    return [
        *[Doc(text=f"{i}. {science_text}") for i in range(5)],
        *[Doc(text=f"{i}. {politics_text}") for i in range(5)],
    ]


@pytest.mark.parametrize("batch_runtime", (ModelType.huggingface,), indirect=["batch_runtime"])
@pytest.mark.parametrize("distillation_framework", DistillationFramework.all())
def test_distillation_classification(batch_runtime, distillation_framework) -> None:
    seed = 42
    base_model_id = "sentence-transformers/paraphrase-mpnet-base-v2"
    if distillation_framework == DistillationFramework.model2vec:
        base_model_id = "minishlab/potion-base-32M"

    docs = _get_docs()

    with TemporaryDirectory() as tmp_dir:
        classifier = classification.Classification(
            task_id="classifier",
            model=batch_runtime.model,
            model_settings=batch_runtime.model_settings,
            batch_size=batch_runtime.batch_size,
            labels={
                "science": "Topics related to scientific disciplines and research",
                "politics": "Topics related to government, elections, and political systems",
            },
        )
        pipe = Pipeline([classifier])
        docs = list(pipe(docs))

        if distillation_framework == DistillationFramework.sentence_transformers:
            with pytest.raises(NotImplementedError):
                classifier.distill(
                    base_model_id=base_model_id,
                    framework=distillation_framework,
                    output_path=Path(tmp_dir),
                    val_frac=0.5,
                    seed=seed,
                    data=docs
                )
        else:
            classifier.distill(
                base_model_id=base_model_id,
                framework=distillation_framework,
                output_path=Path(tmp_dir),
                val_frac=0.5,
                seed=seed,
                data=docs
            )


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


@pytest.mark.parametrize("batch_runtime", [ModelType.huggingface], indirect=["batch_runtime"])
def test_serialization(batch_runtime) -> None:
    seed = 42
    dir_path: str | None
    docs = _get_docs()

    with TemporaryDirectory() as tmp_dir:
        classifier = classification.Classification(
            task_id="classifier",
            model=batch_runtime.model,
            model_settings=batch_runtime.model_settings,
            batch_size=batch_runtime.batch_size,
            labels={
                "science": "Topics related to scientific disciplines and research",
                "politics": "Topics related to government, elections, and political systems",
            },
        )
        pipe = Pipeline(classifier)
        docs = list(pipe(docs))

        classifier.distill(
            base_model_id="sentence-transformers/paraphrase-mpnet-base-v2",
            framework=DistillationFramework.setfit,
            output_path=Path(tmp_dir),
            val_frac=.5,
            seed=seed,
            data=docs
        )

    config = pipe.serialize()
    assert config.model_dump() == {'cls_name': 'sieves.pipeline.core.Pipeline',
 'tasks': {'is_placeholder': False,
           'value': [{'cls_name': 'sieves.tasks.predictive.classification.core.Classification',
                      'fewshot_examples': {'is_placeholder': False,
                                           'value': ()},
                      'batch_size': {'is_placeholder': False, 'value': -1},
                      'condition': {'is_placeholder': False, 'value': None},
                      'model_settings': {'is_placeholder': False,
                                              'value': {
                                                        'config_kwargs': None,
                                                        'inference_kwargs': None,
                                                        'init_kwargs': None,
                                                        'strict': True,
                                                        'inference_mode': None,}},
                      'include_meta': {'is_placeholder': False, 'value': True},
                      'labels': {'is_placeholder': False,
                                             'value': {'politics': 'Topics '
                                                                   'related to '
                                                                   'government, '
                                                                   'elections, '
                                                                   'and '
                                                                   'political '
                                                                   'systems',
                                                       'science': 'Topics '
                                                                  'related to '
                                                                  'scientific '
                                                                  'disciplines '
                                                                  'and '
                                                                  'research'}},
                      'model': {'is_placeholder': True,
                                'value': 'transformers.pipelines.zero_shot_classification.ZeroShotClassificationPipeline'},
                      'prompt_instructions': {'is_placeholder': False,
                                          'value': None},
                      'task_id': {'is_placeholder': False,
                                  'value': 'classifier'},
                      'version': Config.get_version()},
                     ]},
 'use_cache': {'is_placeholder': False, 'value': True},
 'version': Config.get_version()}

    Pipeline.deserialize(config=config, tasks_kwargs=[{"model": batch_runtime.model}])
