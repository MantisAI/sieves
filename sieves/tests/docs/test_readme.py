def test_simple_example() -> None:
    import outlines
    import transformers
    from sieves import Pipeline, tasks, Doc

    # Set up model.
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    model = outlines.models.from_transformers(
        transformers.AutoModelForCausalLM.from_pretrained(model_name),
        transformers.AutoTokenizer.from_pretrained(model_name)
    )

    # Define task.
    task = tasks.Classification(
        labels=["science", "politics"], mode='single', model=model
    )

    # Define pipeline with the classification task.
    pipeline = Pipeline(task)

    # Define documents to analyze.
    doc = Doc(text="The new telescope captures images of distant galaxies.")

    # Run pipeline and print results.
    docs = list(pipeline([doc]))

    # The `results` field contains the structured task output as a unified Pydantic model.
    print(docs[0].results["Classification"])
    # -> ResultSingleLabel(label='science', score=1.0)
    # The `meta` field contains more information helpful for observability and debugging, such as raw model output and token count information.
    print(docs[0].meta)
    # -> {'Classification': {
    #       'raw': ['{ "label": "science" }'], 'usage': {'input_tokens': 83, 'output_tokens': 8, 'chunks': [{'input_tokens': 83, 'output_tokens': 8}]}}, 'usage': {'input_tokens': 83, 'output_tokens': 8},
    #       'cached': False
    #    }

def test_advanced_example() -> None:
    import dspy
    import os
    import pydantic
    import chonkie
    import tokenizers
    from sieves import tasks, Doc

    # Define which schema of entity to extract.
    class Equation(pydantic.BaseModel, frozen=True):
        id: str = pydantic.Field(description="ID/index of equation in paper.")
        equation: str = pydantic.Field(description="Equation as shown in paper.")

    # Setup DSPy model.
    model = dspy.LM(
        "openrouter/google/gemini-3-flash-preview",
        api_base="https://openrouter.ai/api/v1/",
        api_key=os.environ["OPENROUTER_API_KEY"]
    )

    # Build pipeline: ingest -> chunk -> extract.
    pipeline = (
        tasks.Ingestion() +
        tasks.Chunking(chonkie.TokenChunker(tokenizers.Tokenizer.from_pretrained("gpt2"))) +
        tasks.InformationExtraction(entity_type=Equation, model=model)
    )

    # Define docs to analyze.
    doc = Doc(uri="https://arxiv.org/pdf/1204.0162")

    # Run pipeline.
    results = list(pipeline([doc]))

    # Print results.
    for equation in results[0].results["InformationExtraction"].entities:
        print(equation)
