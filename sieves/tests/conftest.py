# mypy: ignore-errors
import os

import anthropic
import dspy
import gliner.multitask
import instructor
import langchain_anthropic
import outlines
import pytest
import tokenizers
import transformers

from sieves import Doc, engines


@pytest.fixture(scope="session")
def tokenizer() -> tokenizers.Tokenizer:
    return tokenizers.Tokenizer.from_pretrained("gpt2")


def _make_engine(engine_type: engines.EngineType, batch_size: int):
    """Create engine.
    :param engine_type: Engine type.
    :param batch_size: Batch size to use in engine.
    :returns Engine: Enstantiated engine.
    """
    match engine_type:
        case engines.EngineType.dspy:
            return engines.dspy_.DSPy(model=dspy.LM("claude-3-haiku-20240307", api_key=os.environ["ANTHROPIC_API_KEY"]))

        case engines.EngineType.glix:
            model_id = "knowledgator/gliner-multitask-v1.0"
            return engines.glix_.GliX(
                model=gliner.multitask.GLiNERClassifier(model=gliner.GLiNER.from_pretrained(model_id)),
                batch_size=batch_size,
            )

        case engines.EngineType.langchain:
            model = langchain_anthropic.ChatAnthropic(
                model="claude-3-haiku-20240307", api_key=os.environ["ANTHROPIC_API_KEY"]
            )
            return engines.langchain_.LangChain(model=model, batch_size=batch_size)

        case engines.EngineType.instructor:
            model = engines.instructor_.Model(
                name="claude-3-haiku-20240307",
                client=instructor.from_anthropic(anthropic.AsyncClient()),
            )
            return engines.instructor_.Instructor(model=model, batch_size=batch_size)

        case engines.EngineType.huggingface:
            model = transformers.pipeline(
                "zero-shot-classification", model="MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33"
            )
            return engines.huggingface_.HuggingFace(model=model, batch_size=batch_size)

        case engines.EngineType.ollama:
            model = engines.ollama_.Model(client_mode="async", host="http://localhost:11434", name="smollm:135m")
            return engines.ollama_.Ollama(model=model, batch_size=batch_size)

        case engines.EngineType.outlines:
            model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
            return engines.outlines_.Outlines(model=outlines.models.transformers(model_name), batch_size=batch_size)


@pytest.fixture(scope="session")
def batch_engine(request) -> engines.Engine:
    """Initializes engine with batching."""
    assert isinstance(request.param, engines.EngineType)
    return _make_engine(engine_type=request.param, batch_size=-1)


@pytest.fixture(scope="session")
def engine(request) -> engines.Engine:
    """Initializes engine without batching."""
    assert isinstance(request.param, engines.EngineType)
    return _make_engine(engine_type=request.param, batch_size=1)


@pytest.fixture(scope="session")
def dummy_docs() -> list[Doc]:
    return [Doc(text="This is about politics stuff. " * 10), Doc(text="This is about science stuff. " * 10)]


@pytest.fixture(scope="session")
def information_extraction_docs() -> list[Doc]:
    return [
        Doc(text="Mahatma Ghandi lived to 79 years old. Bugs Bunny is at least 85 years old."),
        Doc(text="Marie Curie passed away at the age of 67 years. Marie Curie was 67 years old."),
    ]
