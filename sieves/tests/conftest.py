# mypy: ignore-errors
import os

import dspy
import gliclass
import ollama
import outlines
import pytest
import transformers

from sieves import Doc, engines


@pytest.fixture(scope="session")
def engine(request) -> engines.Engine:
    """Initializes engine."""
    match request.param:
        case engines.EngineType.dspy:
            return engines.dspy_.DSPy(model=dspy.LM("claude-3-haiku-20240307", api_key=os.environ["ANTHROPIC_API_KEY"]))
        case engines.EngineType.glix:
            pipeline = gliclass.ZeroShotClassificationPipeline(
                gliclass.GLiClassModel.from_pretrained("knowledgator/gliclass-small-v1.0"),
                transformers.AutoTokenizer.from_pretrained("knowledgator/gliclass-small-v1.0"),
                classification_type="multi-label",
                device="cpu",
            )
            return engines.glix_.GliX(model=pipeline)
        case engines.EngineType.huggingface:
            model = transformers.pipeline(
                "zero-shot-classification", model="MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33"
            )
            return engines.huggingface_.HuggingFace(model=model)
        case engines.EngineType.ollama:
            model = engines.ollama_.Model(client=ollama.Client(host="http://localhost:11434"), name="smollm:135m")
            return engines.ollama_.Ollama(model=model)
        case engines.EngineType.outlines:
            model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
            return engines.outlines_.Outlines(model=outlines.models.transformers(model_name))
        case _:
            raise ValueError(f"Unsupported engine type {request.param}.")


@pytest.fixture(scope="session")
def dummy_docs() -> list[Doc]:
    return [Doc(text="This is about politics stuff. " * 10), Doc(text="This is about science stuff. " * 10)]


@pytest.fixture(scope="session")
def information_extraction_docs() -> list[Doc]:
    return [
        Doc(text="Mahatma Ghandi lived to 79 years old. Bugs Bunny is at least 85 years old."),
        Doc(text="Marie Curie passed away with 67 years. Marie Curie was 67 years old."),
    ]
