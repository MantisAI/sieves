import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Case Study: Structured Crisis Tweet Analysis

    This notebook demonstrates how to use `sieves` to build a structured information extraction pipeline. We'll use a **toy example** involving crisis-related tweets to show how `sieves` can help sift through unstructured text to identify relevant events and extract key entities.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The goal is to build a multi-stage filter (a sieve, if you will) that:
    1. **Classifies** if a tweet is relevant.
    2. **Extracts** the crisis type and location **only if** it's relevant.

    This conditional orchestration allows for efficient processing and reduced noise.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Downloading the Data


    We'll first download a [dataset](https://crisisnlp.qcri.org/data/lrec2016/labeled_cf/CrisisNLP_labeled_data_crowdflower_v2.zip) from https://crisisnlp.qcri.org/.
    """)
    return


@app.cell
def _():
    import urllib.request
    import zipfile
    from pathlib import Path
    import shutil

    url = "https://crisisnlp.qcri.org/data/lrec2016/labeled_cf/CrisisNLP_labeled_data_crowdflower_v2.zip"
    zip_path = Path("CrisisNLP_dataset.zip")
    data_dir = Path("CrisisNLP")

    if not zip_path.exists():
        urllib.request.urlretrieve(url, str(zip_path))

    with zipfile.ZipFile(str(zip_path), 'r') as z:
        z.extractall(".")

    extracted = Path("CrisisNLP_labeled_data_crowdflower")
    data_dir.mkdir(exist_ok=True)
    for item in extracted.iterdir():
        target = data_dir / item.name
        if not target.exists():
            item.rename(target)
    shutil.rmtree(extracted)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next we'll load the data into memory.
    """)
    return


@app.cell
def _():
    import pandas as pd
    import os

    data: pd.DataFrame = pd.DataFrame()

    for file in os.listdir("CrisisNLP"):
        if os.path.isdir(os.path.join("CrisisNLP", file)):
            for file_ in os.listdir(os.path.join("CrisisNLP", file)):

                if file_.endswith(".tsv"):
                    data_ = pd.read_csv(os.path.join("CrisisNLP", file, file_), sep="\t")
                    if len(data_) == 0:
                        continue

                    data_["dataset"] = file.split("_")[1]
                    crisis_type = file.split("_")[2]
                    if crisis_type == 'eq':
                        crisis_type = 'Earthquake'
                    elif crisis_type in ('Odile', 'Pam', 'Typhoon'):
                        crisis_type = 'Hurrican'
                    elif crisis_type in ('ebola', 'cf', 'East'):
                        crisis_type = 'Diseases'
                    elif crisis_type == 'floods':
                        crisis_type = 'Floods'
                    data_["crisis_type"] = crisis_type
                    data = pd.concat([data, data_])
    return data, os, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Data Exploration

    We'll load the CrisisNLP dataset, which contains tweets labeled by humans across different disaster types. This "gold" data allows us to evaluate our automated pipeline later.
    """)
    return


@app.cell
def _(data: pd.DataFrame, mo):
    mo.ui.table(data.head(100), label="CrisisNLP Dataset Preview")
    return


@app.cell
def _(data: pd.DataFrame):
    # Quick summary of the dataset
    summary = {
        "Total Tweets": len(data),
        "Crisis Types": data.crisis_type.unique().tolist(),
        "Labels": data.label.unique().tolist()
    }
    summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we'll convert the data into `sieves` documents, which will allow us to process them. We'll also sample down the data to cut down the time we need for processing the data.
    """)
    return


@app.cell
def _(data: pd.DataFrame):
    from sieves import Doc
    import random

    # We down-sample our dataset to avoid long processing times.
    data_sampled = data.sample(n=100)

    docs = [
        Doc(
            uri=f"tweet_{row['tweet_id']}",
            text=row['tweet_text'],
            meta={
                'tweet_id': row['tweet_id'],
                'gold_label': row['label'],
                'dataset': row['dataset'],
                'crisis_type': row['crisis_type']
            }
        )
        for idx, row in data_sampled.iterrows()
    ]

    print(f"Created {len(docs)} docs.")
    print(f"\nSample doc:")
    print(f"Tweet: {docs[0].text[:100]}...")
    print(f"Metadata: {docs[0].meta}")
    return Doc, data_sampled, docs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setting up the Pipeline

    A `sieves` pipeline is composed of tasks. We'll use:
    1. **ClassificationTask**: To filter for relevance and identify crisis types.
    2. **InformationExtractionTask**: To extract structured entities (Locations) using Pydantic models.

    ### Conditional Orchestration
    We only want to run the expensive extraction tasks on tweets that are actually related to a crisis. `sieves` allows us to define a `condition` function that acts as a gatekeeper.
    """)
    return


@app.cell
def _(os):
    import dspy

    # We use a lightweight model for this demonstration
    model = dspy.LM(
        "openrouter/google/gemini-2.5-flash-lite-preview-09-2025",
        api_base="https://openrouter.ai/api/v1",
        api_key=os.environ['OPENROUTER_API_KEY'],
    )

    # Batching improves throughput by processing multiple docs in a single prompt
    batch_size = 10
    return batch_size, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's create a task predicting tweet labels, one of which indicates whether the tweet is being relevant to any crisis:
    """)
    return


@app.cell
def _(batch_size, data_sampled, model):
    from sieves import tasks

    crisis_label_classifier = tasks.Classification(
        task_id="crisis_label_classifier",
        labels=data_sampled.label.unique(),
        mode='single',
        model=model,
        batch_size=batch_size,
    )
    return crisis_label_classifier, tasks


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Defining the "Gatekeeper"
    The `related_to_crisis` function checks the output of the first classifier. If the tweet isn't relevant or the confidence is too low, subsequent tasks in the pipeline will be skipped for that document.
    """)
    return


@app.cell
def _(Doc, batch_size, data_sampled, model, tasks):
    def related_to_crisis(doc: Doc) -> bool:
        """Checks if the tweet is relevant enough to proceed with further extraction."""
        result = doc.results.get("crisis_label_classifier")
        if not result:
            return False
        return result.label != 'not_related_or_irrelevant' and result.score >= .6


    crisis_type_classifier = tasks.Classification(
        task_id="crisis_type_classifier",
        labels=data_sampled.crisis_type.unique(),
        mode='single',
        model=model,
        condition=related_to_crisis,
        batch_size=batch_size,
    )
    return crisis_type_classifier, related_to_crisis


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Schema-Driven Information Extraction
    By using Pydantic models, we ensure that the LLM output is parsed into a structured object. This makes the data immediately useful for downstream applications.
    """)
    return


@app.cell
def _():
    import pydantic
    from typing import Literal


    class Country(pydantic.BaseModel, frozen=True):
        name: str | None = pydantic.Field(
            description="The name of the country mentioned in the tweet, if any."
        )
    return (Country,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Having the entity types defined, we can proceed to define our tasks. We'll reuse the conditional check from before that allows us to skip irrelevant tweets.
    """)
    return


@app.cell
def _(Country, batch_size, model, related_to_crisis, tasks):
    location_extractor = tasks.InformationExtraction(
        task_id="location_extractor",
        entity_type=Country,
        model=model,
        mode='single',
        batch_size=batch_size,
        condition=related_to_crisis
    )
    return (location_extractor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally we can combine four tasks into a pipeline:
    """)
    return


@app.cell
def _(crisis_label_classifier, crisis_type_classifier, location_extractor):
    pipeline = (
        crisis_label_classifier +
        crisis_type_classifier +
        location_extractor
    )
    return (pipeline,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluating Results

    ### Running the Pipeline

    Everything is in place, so we run our pipeline and collect results:
    """)
    return


@app.cell
def _(docs, pipeline):
    results = list(pipeline(docs))
    return (results,)


@app.cell
def _(results):
    from typing import Any

    # Flatten results for display in a table
    display_results: list[dict[str, Any]] = []
    for doc in results:
        display_results.append({
            "Tweet": doc.text[:100] + "...",
            "Label": doc.results.get('crisis_label_classifier').label if doc.results.get('crisis_label_classifier') else "N/A",
            "Type": doc.results.get('crisis_type_classifier').label if doc.results.get('crisis_type_classifier') else "Skipped",
            "Location": doc.results.get('location_extractor').entity.name if doc.results.get('location_extractor') and doc.results.get('location_extractor').entity else "None/Skipped"
        })
    return (display_results,)


@app.cell
def _(display_results: list[dict[str, Any]], mo):
    mo.ui.table(display_results)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Evaluating Pipeline Reliability

    In a real-world scenario, we need to know if we can trust our automated extraction. By comparing our pipeline's "Predicted" results against the "Gold" human labels, we can calculate metrics like Accuracy and F1 Score.

    This evaluation helps us tune our `condition` thresholds and improve our task prompts.
    """)
    return


@app.cell
def _(pd, results):
    evaluation_data: list[dict] = []
    for res_doc in results:
        crisis_label_result = res_doc.results.get('crisis_label_classifier')
        crisis_type_result = res_doc.results.get('crisis_type_classifier')
        location_result = res_doc.results.get('location_extractor')

        location = location_result.entity.name if location_result and location_result.entity else None

        evaluation_data.append({
            'tweet_id': res_doc.meta.get('tweet_id', ''),
            'text': res_doc.text[:100],
            'label_gold': res_doc.meta.get('gold_label', 'unknown'),
            'label_predicted': crisis_label_result.label if crisis_label_result else None,
            'crisis_type_gold': res_doc.meta.get('crisis_type', 'unknown'),
            'crisis_type_predicted': crisis_type_result.label if crisis_type_result else None,
            'location_predicted': location,
        })

    df_eval = pd.DataFrame(evaluation_data)
    return (df_eval,)


@app.cell
def _(df_eval, mo):
    from sklearn.metrics import f1_score, accuracy_score, recall_score
    from pprint import pprint

    metrics_summary: list[dict[str, float]] = []
    errors: list[dict[str, str | float]] = []

    for category in ["label", "crisis_type"]:
        y_true = df_eval[f"{category}_gold"].astype(str)
        y_pred = df_eval[f"{category}_predicted"].fillna("skipped").astype(str)

        metrics_summary.append({
            "Task": category,
            "Metric": 'Accuracy',
            "Value": accuracy_score(y_true, y_pred),
        })
        metrics_summary.append({
            "Task": category,
            "Metric": 'F1 (Macro)',
            "Value": f1_score(y_true, y_pred, average="macro"),
        })
        metrics_summary.append({
            "Task": category,
            "Metric": 'Recall (Macro)',
            "Value": recall_score(y_true, y_pred, average="macro"),
        })

        wrong_mask = df_eval[f"{category}_gold"] != df_eval[f"{category}_predicted"].fillna("skipped")
        for _, row in df_eval[wrong_mask].iterrows():
            errors.append({
                "Task": category,
                "Text": row["text"],
                "Gold": row[f"{category}_gold"],
                "Predicted": row[f"{category}_predicted"],
            })

    mo.ui.table(metrics_summary)
    return (errors,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's inspect the errors - hopefully we can learn from to improve our pipeline (or maybe even find mistakes in the gold data!):
    """)
    return


@app.cell
def _(errors: list[dict[str, str | float]], mo):
    mo.ui.table(errors)
    return


@app.cell(hide_code=True)
def _():
    return


if __name__ == "__main__":
    app.run()
