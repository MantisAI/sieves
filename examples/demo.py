"""Demo for PyData Amsterdam 2025."""

import os
from pprint import pprint
from typing import Literal

import chonkie
import openai
import outlines
import pydantic

from sieves import Doc, Engine, Pipeline, tasks

docs = [
    Doc(
        uri="https://www.euronews.com/my-europe/2025/09/11/fact-check-is-the-eu-about-to-start-scanning-your-text-messages"
    )
]
engine = Engine(model=outlines.from_openai(openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"]), model_name="gpt-5"))


class Country(pydantic.BaseModel, frozen=True):
    """Information to look for in document."""

    name: str
    in_eu: bool
    stance_on_chat_message_scanning_proposal: Literal["pro", "undecided", "contra", "unknown"]


pipe = Pipeline(
    [
        # Add document ingestion.
        tasks.Ingestion(export_format="markdown"),
        # Chunk just in case text is too long to fit into model context window.
        tasks.Chunking(chonkie.TokenChunker()),
        # Define information extraction task.
        tasks.InformationExtraction(entity_type=Country, engine=engine),
    ]
)

for doc in pipe(docs):
    results = (
        (res.name, res.stance_on_chat_message_scanning_proposal)
        for res in doc.results["InformationExtraction"]
        if res.in_eu
    )
    pprint(sorted(results, key=lambda item: item[1]))
