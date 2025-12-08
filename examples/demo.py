"""OSA demo."""
import os
from pprint import pprint
from typing import Literal

import dspy
import pydantic

from sieves import Pipeline, tasks, Doc

openrouter_api_base = "https://openrouter.ai/api/v1/"
openrouter_model_id = "google/gemini-2.5-flash-lite-preview-09-2025"

# Define documents by text or URI.
docs = [
  Doc(
    uri="https://www.eff.org/deeplinks/2025/12/after-years-controversy-eus-chat-control-nears-its-final-hurdle-what-know"
  )
]

# Define model.
model = dspy.LM(
  f"openrouter/{openrouter_model_id}",
  api_base=openrouter_api_base,
  api_key=os.environ['OPENROUTER_API_KEY']
)


class Concern(pydantic.BaseModel):
  """A concern against the introduction of ChatControl."""
  argument: str = pydantic.Field(description="Describes the core of a concern about ChatControl.")


# Create pipeline with tasks.
pipe =  tasks.Ingestion() + \
        tasks.QuestionAnswering(
          questions=[
            "What's the current situation around ChatControl?",
            "I thought ChatControl was rejected?",
          ],
          model=model
        ) + \
        tasks.Summarization(n_words=20, model=model) + \
        tasks.InformationExtraction(entity_type=Concern,model=model)

# Run pipe and output results.
for doc in pipe(docs):
  pprint(doc.results)
