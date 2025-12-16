"""Chat Control demo for Marimo."""
import marimo

__generated_with = "0.18.3"
app = marimo.App(
    width="full",
    layout_file="layouts/demo_chatcontrol.slides.json",
)

with app.setup:
    # Initialization code that runs before all other cells
    pass


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # OSA Community Demo: Sieves

    2025-12-12
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Chat Control is a hot topic in European politics. In order to get up to date on current developments, we'll analyze a recent article of the Electronic Frontier Foundation on this topic.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Building a dataset

    We define a single document referencing a recent EFF article on this:
    """)
    return


@app.cell
def _():
    from sieves import Doc

    docs = [
        Doc(
            uri="https://www.eff.org/deeplinks/2025/12/after-years-controversy-eus-chat-control-nears-its-final-hurdle-what-know"
        )
    ]
    return (docs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Before we run anything, we'll need to define our model. In this case we go with DSPy (which uses LiteLLM underneath), but other libraries
    are supported:
    - Hugging Face zero-shot pipelines
    - GliNER2
    - Outlines
    - LangChain
    """)
    return


@app.cell
def _():
    import os

    import dspy

    openrouter_api_base = "https://openrouter.ai/api/v1/"
    openrouter_model_id = "anthropic/claude-haiku-4.5"

    model = dspy.LM(
        f"openrouter/{openrouter_model_id}", api_base=openrouter_api_base, api_key=os.environ["OPENROUTER_API_KEY"]
    )
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Questions...
    - What's the summary of the article?
    - What are possible risks of introducing ChatControl?
    - Why is ChatControl an issue _again_ after being rejected already?
    - How positive or negative is the author's stance on chat control?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's define a pipeline that reflects our requirements. As first task we add an _ingestion_ task to convert the document into plain text, then we define the tasks that address our questions.
    """)
    return


@app.cell
def _(model):
    from sieves import tasks

    pipe = (
        tasks.Ingestion()
        + tasks.QuestionAnswering(
            questions=[
                "What are the risks of introducing ChatControl?",
                "I thought ChatControl was rejected. Why is this an issue again?",
            ],
            model=model,
        )
        + tasks.Summarization(n_words=20, model=model)
        + tasks.SentimentAnalysis(aspects=["digital rights organisation", "chat control"], model=model)
    )
    return pipe, tasks


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ...(changing our mind)...
    Hold on - maybe we want to read further literature on this?

    To find related articles, we first must define what we're looking for, so that the model can pick up on it.
    """)
    return


@app.cell
def _():
    import pydantic

    class RelatedLiterature(pydantic.BaseModel, frozen=True):
        """Another article on the topic of ChatControl."""

        topic: str = pydantic.Field(description="Topic or title of the article on this.")
        link: str = pydantic.Field(description="Link to the article.")

    return (RelatedLiterature,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we can extend our pipeline:
    """)
    return


@app.cell
def _(RelatedLiterature, model, pipe, tasks):
    full_pipe = pipe + tasks.InformationExtraction(entity_type=RelatedLiterature, model=model)
    return (full_pipe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ...and answers!

    We're finally ready to extract results:
    """)
    return


@app.cell
def _(docs, full_pipe):
    doc_with_results = [result for result in full_pipe(docs)][0]
    result = doc_with_results.results
    return (result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What's the summary of the article?**
    """)
    return


@app.cell
def _(result):
    from pprint import pprint

    pprint(result["Summarization"])
    return (pprint,)


@app.cell
def _(mo):
    mo.md(r"""
    **What are possible risks of introducing ChatControl?**
    """)
    return


@app.cell
def _(pprint, result):
    pprint(result["QuestionAnswering"][0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Why is ChatControl an issue again after being rejected already?**
    """)
    return


@app.cell
def _(pprint, result):
    pprint(result["QuestionAnswering"][1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **How positive or negative is the author's stance on chat control?**
    """)
    return


@app.cell(hide_code=True)
def _(pprint, result):
    pprint(result["SentimentAnalysis"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Further EFF literature on this topic:**
    """)
    return


@app.cell
def _(result):
    for article in result["InformationExtraction"]:
        print(article.topic)
        print(f"{article.link}")
        print("#######")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
