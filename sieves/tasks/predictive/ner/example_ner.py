from sieves.tasks.predictive.ner.core import NER
from sieves import Pipeline, engines, Doc
import outlines
import dspy
from sieves.engines.dspy_ import DSPy
import openai
from dotenv import load_dotenv
import os
load_dotenv()

import anthropic
import langchain_anthropic

import instructor

############ DOCUMENTS
docs = [Doc(text="John studied data science in Barcelona and lives with James"),
        Doc(text="Maria studied computer engineering in Madrid and works with Carlos")]

############ DSPY
lm = dspy.LM('openai/gpt-4o', api_key=os.getenv('OPENAI_API_KEY'))
dspy.configure(lm=lm)
engine_dspy = engines.DSPy(model=lm,
                          config_kwargs={"max_tokens": 1000},  # Adjust depending on the model
                          inference_kwargs={"temperature": 0.2})

############ OUTLINES
model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
engine_outlines = engines.Outlines(model=outlines.models.transformers(model_name))

############ LANGCHAIN
model = langchain_anthropic.ChatAnthropic(
                model="claude-3-haiku-20240307", api_key=os.environ["ANTHROPIC_API_KEY"]
            )
engine_langchain = engines.LangChain(model=model, batch_size=1)

############ INSTRUCTOR
model = engines.instructor_.Model(
                name="gpt-4o",
                client=instructor.from_openai(openai.AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))),
            )
engine_instructor = engines.Instructor(model=model, batch_size=1)

############ OLLAMA
# model = engines.ollama_.Model(client_mode="async", host="http://localhost:11434", name="llama3.1:8b")
# engine_ollama = engines.ollama_.Ollama(model=model, batch_size=1)

############ PIPELINE
pipe = Pipeline(
    [
        NER(entities=["PERSON", "LOCATION", "ORGANIZATION"], engine=engine_instructor, task_id="NER"),
    ]
)

############ RUN PIPELINE
results = pipe(docs)
for doc in results:
  print(f"Results: {doc.results}")
