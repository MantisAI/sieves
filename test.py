import transformers

from sieves import Doc, Engine, Pipeline
from sieves.tasks import Classification

if __name__ == "__main__":
    model = transformers.pipeline(
        "zero-shot-classification", model="MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33"
    )
    engine = Engine(model=model)
    pipe = Pipeline(
        Classification(
            labels=["science", "politics"],
            engine=engine,
        )
    )
    docs = [Doc(text="This is about atoms and stars.")]
    docs = list(pipe(docs))

    for doc in docs:
        print(doc.results)
