"""Demo for PyData Amsterdam 2025.

Required additional dependencies:
- openai
- outlines
"""

import os
from collections import defaultdict
from pprint import pprint
from typing import Literal

import chonkie
import openai
import outlines
import pydantic

from sieves import Doc, Engine, Pipeline, tasks


class Country(pydantic.BaseModel, frozen=True):
    """Information to look for in document."""

    name: str
    in_eu: bool
    stance_on_chat_message_scanning_proposal: Literal["pro", "undecided", "contra", "unknown"]


if __name__ == '__main__':
    docs = [
        Doc(
            uri="https://www.techradar.com/computing/cyber-security/chat-control-the-list-of-countries-opposing-the-law-grows-but-support-remains-strong"
        )
    ]
    engine = Engine(model=outlines.from_openai(openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"]), model_name="gpt-5"))

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
        results = defaultdict(list)
        for res in doc.results["InformationExtraction"]:
            if res.in_eu:
                results[res.stance_on_chat_message_scanning_proposal].append(res.name)

        pprint(results)
