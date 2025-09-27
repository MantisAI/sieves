"""Demo for PyData Amsterdam 2025.

Required additional dependencies:
- openai
- outlines
"""

import os
from collections import defaultdict
from pprint import pprint
from typing import Literal

import openai
import outlines
import pydantic

from sieves import Doc, Engine, tasks


class Country(pydantic.BaseModel, frozen=True):
    """Describes a country and it's stance on the chat control proposal."""

    name: str
    in_eu: bool
    stance_on_chat_control_proposal: Literal["pro", "undecided", "contra", "unknown"]


if __name__ == '__main__':
    docs = [
        Doc(
            uri="https://www.techradar.com/computing/cyber-security/chat-control-the-list-of-countries-opposing-the-"
                "law-grows-but-support-remains-strong"
        )
    ]

    model = outlines.from_openai(
        openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
        model_name="gpt-5-mini"
    )

    pipe = tasks.Ingestion() + tasks.InformationExtraction(entity_type=Country, model=model)

    for doc in pipe(docs):
        countries = defaultdict(list)
        for country in doc.results["InformationExtraction"]:
            assert isinstance(country, Country)
            if country.in_eu:
                countries[country.stance_on_chat_control_proposal].append(country.name)

        pprint(countries)
