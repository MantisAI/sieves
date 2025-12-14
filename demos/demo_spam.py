import marimo

__generated_with = "0.18.3"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # OSA Community Demo: Sieves

    2025-12-12
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## NLP 101, contemporary: building a spam filter
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A classic first example in NLP is building a spam filter, often done using bag-of-words/tf-idf classification.

    How does a zero-shot version of this look like?

    We'll first define a bunch of documents. Half of them is spam, the other half is not.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Building a dataset
    """)
    return


@app.cell
def _():
    from sieves import Doc

    docs = [
        Doc(
            text="""
        Dear User,
        We detected unusual activity on your account. To avoid immediate suspension, please verify your information by clicking the secure link below within 24 hours. Failure to comply may result in permanent closure.
        """
        ),
        Doc(
            text="""
        Hi Team,
        I wanted to share a quick update on the project. The initial phase is complete, and we’re on track to begin the next steps early next week. I’ll follow up with a detailed timeline shortly.
        """
        ),
        Doc(
            text="""
        Hello,
        You are the lucky winner of an exclusive reward valued at $5,000. This offer is time-sensitive. Confirm your eligibility now to claim your prize before it expires.
        """
        ),
        Doc(
            text="""
        Hello,
        Thank you for the productive discussion earlier today. As agreed, I’ll send over the revised proposal by Friday. Please let me know if you have any questions in the meantime.
        """
        ),
        Doc(
            text="""
        Dear Customer,
        Your recent invoice remains unpaid. Please review the attached document and settle the balance immediately to avoid additional fees. Contact support if you believe this is an error.
        """
        ),
        Doc(
            text="""
        Hi,
        Could you please review the attached document and share your feedback by the end of the week? Your input will help us finalize the draft before submission.
        """
        ),
    ]
    return (docs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First, we'll define our model. In this case we go with DSPy (which uses LiteLLM underneath), but other libraries
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

    import anthropic
    import dspy
    import outlines
    import transformers

    openrouter_api_base = "https://openrouter.ai/api/v1/"
    openrouter_model_id = "google/gemini-2.5-flash-lite-preview-09-2025"

    model = dspy.LM(
        f"openrouter/{openrouter_model_id}", api_base=openrouter_api_base, api_key=os.environ["OPENROUTER_API_KEY"]
    )
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Building a pipeline

    We'll want to **classify** our data into spam and not spam. We'd also like a **summary** of our mails.
    """)
    return


@app.cell
def _(model):
    from sieves import tasks

    classifier = tasks.Classification(
        labels=["spam", "not spam"],
        multi_label=False,
        model=model,
    )
    summarizer = tasks.Summarization(n_words=10, model=model)
    pipe = classifier + summarizer
    return classifier, pipe, tasks


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Spam or not spam?

    Let's give this a spin:
    """)
    return


@app.cell
def _(docs, pipe):
    from pprint import pprint

    for doc in pipe(docs):
        print(doc.text)
        pprint(doc.results["Classification"])
        pprint(doc.results["Summarization"])
    return (pprint,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Looks reasonable! But wait, we're not interested in summarizing spam emails - we just want to ignore them.

    ## Enter conditionals
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Conditionals are functions that tell the pipeline which documents to skip for a task. For example, we'll want to skip summarizing spam emails, as this is wasted time and money:
    """)
    return


@app.cell
def _(classifier, docs, model, pprint, tasks):
    conditional_summarizer = tasks.Summarization(
        n_words=10, model=model, condition=lambda doc: doc.results["Classification"][0] == "not spam"
    )
    improved_pipe = classifier + conditional_summarizer

    for doc2 in improved_pipe(docs):
        print(doc2.text)
        pprint(doc2.results["Classification"])
        pprint(doc2.results["Summarization"])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
