# mypy: ignore-errors
import os
import pickle
import tempfile
from pathlib import Path

import chonkie
import dspy
import pytest

from sieves import Pipeline
from sieves.model_wrappers import ModelType
from sieves.serialization import Config
from sieves.tasks import preprocessing
from sieves.tasks.predictive import classification
from sieves.tests.conftest import make_model


@pytest.mark.parametrize(
    "batch_runtime",
    [ModelType.dspy],
    indirect=["batch_runtime"],
)
def test_serialization_pipeline(dummy_docs, batch_runtime, tokenizer):
    """Tests serialization and deserialization of pipeline to files and config objects."""
    pipe = Pipeline(
        [
            preprocessing.Chunking(chonkie.TokenChunker(tokenizer)),
            classification.Classification(
                task_id="classifier",
                labels={"science": "Everything about science.", "politics": "Everything about politics."},
                model=batch_runtime.model,
                model_settings=batch_runtime.model_settings,
                batch_size=batch_runtime.batch_size,
            ),
        ]
    )

    # Get config, assert values are correct.
    config = pipe.serialize()
    config_model_dump = config.model_dump()
    version = Config.get_version()
    assert config_model_dump == {
        "cls_name": "sieves.pipeline.core.Pipeline",
        "use_cache": {"is_placeholder": False, "value": True},
        "tasks": {
            "is_placeholder": False,
            "value": [
                {
                    "chunker": {"is_placeholder": True, "value": "chonkie.chunker.token.TokenChunker"},
                    'batch_size': {'is_placeholder': False, 'value': -1},
                    "cls_name": "sieves.tasks.preprocessing.chunking.core.Chunking",
                    "include_meta": {"is_placeholder": False, "value": False},
                    "task_id": {"is_placeholder": False, "value": "Chunking"},
                    'condition': {'is_placeholder': False, 'value': None},
                    "version": version,
                },
                {
                    "cls_name": "sieves.tasks.predictive.classification.core.Classification",
                    'model_settings': {
                        'is_placeholder': False,
                        'value': {
                            'config_kwargs': None,
                            'inference_kwargs': None,
                            'init_kwargs': None,
                            'strict': True,
                            'inference_mode': None,
                        }
                    },
                    'condition': {'is_placeholder': False, 'value': None},
                    "fewshot_examples": {"is_placeholder": False, "value": ()},
                    "include_meta": {"is_placeholder": False, "value": True},
                    'batch_size': {'is_placeholder': False, 'value': -1},
                    "labels": {
                        "is_placeholder": False,
                        "value": {"science": "Everything about science.", "politics": "Everything about politics."},
                    },
                    'model': {'is_placeholder': True, 'value': 'dspy.clients.lm.LM'},
                    "prompt_instructions": {"is_placeholder": False, "value": None},
                    "task_id": {"is_placeholder": False, "value": "classifier"},
                    "version": version,
                },
            ],
        },
        "version": version,
    }

    # Save config to temporary file
    with tempfile.NamedTemporaryFile(suffix=".yml") as tmp_file:
        tmp_path = Path(tmp_file.name)
        config.dump(tmp_path)

        # Load config from file and verify it matches
        loaded_config = Config.load(tmp_path)
        # For some reason empty tuple is stored as list, which is fine for our purposes.
        assert config_model_dump["tasks"]["value"][1]["fewshot_examples"]["value"] == ()
        config_model_dump["tasks"]["value"][1]["fewshot_examples"]["value"] = []
        assert loaded_config.model_dump() == config_model_dump

        # Restore pipeline from config.
        loaded_pipe = Pipeline.load(
            tmp_path,
            (
                {"chunker": chonkie.TokenChunker(tokenizer)},
                {"model": make_model(ModelType.dspy)},
            ),
        )

        # Run restored pipeline.
        docs = list(loaded_pipe(dummy_docs))
        assert len(docs) == 2
        assert len(docs[0].results["classifier"])

        # Compare loaded pipe config with original one.
        assert loaded_pipe.serialize().model_dump() == config_model_dump


def test_serialization_docs(dummy_docs):
    """Test serializition of docs by saving to and loading from pickle objects."""
    # Create a temporary file for pickle serialization.
    with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp_file:
        tmp_path = Path(tmp_file.name)

        # Pickle the dummy_docs to file.
        with open(tmp_path, "wb") as f:
            pickle.dump(dummy_docs, f)

        # Load the docs back from file.
        with open(tmp_path, "rb") as f:
            loaded_docs = pickle.load(f)

        # Assert the loaded docs are identical to original
        assert len(loaded_docs) == len(dummy_docs)
        assert all([orig_doc == loaded_doc for orig_doc, loaded_doc in zip(dummy_docs, loaded_docs)])

        # Test that comparing Doc with int raises NotImplementedError
        with pytest.raises(NotImplementedError):
            loaded_docs[0] == 42
