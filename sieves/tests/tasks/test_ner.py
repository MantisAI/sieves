# mypy: ignore-errors
import pytest
import os
from sieves import Doc, Pipeline, engines
from sieves.tasks.predictive.ner.core import NER
import dspy
import outlines
import langchain_anthropic
import instructor
from dotenv import load_dotenv
import openai

load_dotenv()

@pytest.fixture
def ner_docs():
    return [
        Doc(text="John Smith works at Microsoft in Seattle."),
        Doc(text="Sarah visited Paris last summer with her friend Alex.")
    ]


# This is a helper function, not a test
def run_ner_test(ner_docs, engine, engine_type):
    pipe = Pipeline(
        [
            NER(
                entities=["PERSON", "LOCATION", "ORGANIZATION"],
                engine=engine,
                task_id="NER",
            ),
        ]
    )
    docs = list(pipe(ner_docs))
    
    assert len(docs) == 2, f"Engine type: {engine_type}"
    for doc in docs:
        assert doc.text, f"Engine type: {engine_type}"
        assert "NER" in doc.results, f"Engine type: {engine_type}"
        assert hasattr(doc.results["NER"], "entities"), f"Engine type: {engine_type}"
        assert isinstance(doc.results["NER"].entities, list), f"Engine type: {engine_type}"


# Create a test generator function that will create individual tests for each engine
def create_engine_test(engine_name, engine_factory):
    def test_function(ner_docs):
        engine = engine_factory()
        run_ner_test(ner_docs, engine, engine_name)
    # Set the function name and docstring for better pytest reporting
    test_function.__name__ = f"test_{engine_name}_engine"
    test_function.__doc__ = f"Test the {engine_name} engine for NER task"
    return test_function

# Define engine factories to create fresh instances for each test
def dspy_engine_factory():
    return engines.DSPy(
        model=dspy.LM('openai/gpt-4o', api_key=os.getenv('OPENAI_API_KEY')),
        config_kwargs={"max_tokens": 1000},
        inference_kwargs={"temperature": 0.2}
    )

def outlines_engine_factory():
    return engines.Outlines(
        model=outlines.models.transformers("HuggingFaceTB/SmolLM-135M-Instruct")
    )

def langchain_engine_factory():
    return engines.LangChain(
        model=langchain_anthropic.ChatAnthropic(
            model="claude-3-haiku-20240307", api_key=os.environ["ANTHROPIC_API_KEY"]
        ),
        batch_size=1
    )

def instructor_engine_factory():
    return engines.Instructor(
        model=engines.instructor_.Model(
            name="gpt-4o",
            client=instructor.from_openai(openai.AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY')))
        ),
        batch_size=1
    )

# Create individual test functions for each engine
test_dspy_engine = create_engine_test("dspy", dspy_engine_factory)
test_outlines_engine = create_engine_test("outlines", outlines_engine_factory)
test_langchain_engine = create_engine_test("langchain", langchain_engine_factory)
test_instructor_engine = create_engine_test("instructor", instructor_engine_factory)

# Keep the original combined test for backward compatibility or if you want to run all engines in one test
def test_all_engines(ner_docs):
    engines_to_test = {
        "dspy": dspy_engine_factory(),
        "outlines": outlines_engine_factory(),
        "langchain": langchain_engine_factory(),
        "instructor": instructor_engine_factory()
    }
    for engine_name, engine in engines_to_test.items():
        run_ner_test(ner_docs, engine, engine_name)

if __name__ == "__main__":
    # This allows running the test directly without pytest
    docs = ner_docs()
    engines_dict = {
        "dspy": engines.DSPy(
            model=dspy.LM('openai/gpt-4o', api_key=os.getenv('OPENAI_API_KEY')),
            config_kwargs={"max_tokens": 1000},
            inference_kwargs={"temperature": 0.2}
        ),
        "outlines": engines.Outlines(
            model=outlines.models.transformers("HuggingFaceTB/SmolLM-135M-Instruct")
        ),
        "langchain": engines.LangChain(
            model=langchain_anthropic.ChatAnthropic(
                model="claude-3-haiku-20240307", api_key=os.environ["ANTHROPIC_API_KEY"]
            ),
            batch_size=1
        ),
        "instructor": engines.Instructor(
            model=engines.instructor_.Model(
                name="gpt-4o",
                client=instructor.from_openai(openai.AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY')))
            ),
            batch_size=1
        )
    }
    for engine_name, engine in engines_dict.items():
        print(f"\nTesting engine: {engine_name}")
        run_ner_test(docs, engine, engine_name)
    print("\nAll tests passed!")