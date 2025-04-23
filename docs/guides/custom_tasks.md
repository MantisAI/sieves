# Creating Custom Tasks

This guide explains how to create custom tasks. `sieves` distinguishes two types of tasks:
1. Ordinary tasks inherit from `Task`. Pretty much their only requirement is to process a bunch of documents and output
   the same set of documents with their modifications. 
2. Predictive tasks inherit from `PredictiveTask` (which inherits from `Task`). Those are for tasks using engines (i.e. 
   zero-shot models). They are more complex, as they need to implement the required interface to integrate with at least 
   one engines. 

While there are a bunch of pre-built tasks available for you to use, you might want to write your own to match your 
use-case. This guide describes how to do that.

If you feel like your task might be useful for others, we'd happy to see you submit a PR! 

## Tasks

Inherit from `Task` whenever you want to implement something that doesn't require interacting with engines.
That can be document pre- or postprocessing, or something completely different - you could e.g. run an agent following 
instructions provided in `docs`, and then follow this up with a subsequent task in your pipeline analyzing and 
structuring those results.

To create a basic custom task, inherit from the `Task` class and implement the required abstract methods. In this case
we'll implement a dummy task that counts how many characters are in the document's text and stores that as a result. 

```python
from typing import Iterable
from sieves.tasks.core import Task
from sieves.data import Doc

class CharCountTask(Task):    
    def __call__(self, docs: Iterable[Doc]) -> Iterable[Doc]:
        """Counts characters in doc.text.
        
        :param docs: Documents to process.
        :return Iterable[Doc]: Processed documents.
        """
        for doc in docs:
            doc.results[self.id] = len(doc.text)
            yield doc
```

That's it! You can customize this, of course. You might also want to extend `__init__()` to allow for initializing what 
you need.

## Predictive Tasks

Inherit from `PredictiveTask` whenever you want to make use of the structured generation capabilities in `sieves`.
`PredictiveTask` requires you to implement a few methods that define how your task expects results to be structured, how 
few-shot examples are expected to look like, which prompt to use etc.

We'll break down how to create a predictive task step by step. For this example, let's implement a sentiment analysis 
task using `outlines`.

### 1. Implement a `Bridge`

A `Bridge` defines how to solve a task for a certain engine. We decided to go with `outlines` as our engine (you can 
allow multiple engines for a task by implementing corresponding bridges, but for simplicity's sake we'll stick with 
DSPy only here).

A `Bridge` requires you to implement/specify the following:
- A _prompt template_ (optional depending on the engine used).
- A _prompt signature description_ (optional depending on the engine used).
- A _prompt signature_ describing how results have to be structured.
- Which _inference mode_ to use. This defines how to engine queries the model and parses the results.
- How to _integrate_ results into docs.
- How to _consolidate_ results from multiple doc chunks into one result per doc.  

We'll save this in `sentiment_analysis_bridges.py`.

```python
from collections.abc import Iterable
from functools import cached_property

import pydantic

from sieves.data import Doc
from sieves.engines import EngineInferenceMode, outlines_
from sieves.tasks.predictive.bridges import Bridge


# This is how we require our response to look like - we require not just the score, but also a reasoning/justification 
# for why this model assigns this score. We also force the score to be between 0 and 1.
class SentimentEstimate(pydantic.BaseModel):
   reasoning: str
   score: pydantic.confloat(ge=0, le=1)
   

# This is the bridge class.
class OutlinesSentimentAnalysis(Bridge[SentimentEstimate, SentimentEstimate, EngineInferenceMode]):
    # This defines the default prompt template as Jinja2 template string.
    # We include an example block allowing us to include fewshot examples.
    @property
    def _prompt_template(self) -> str | None:
        return """
        Estimate the sentiment in this text as a float between 0 and 1. 0 is negative, 1 is positive. Provide your 
        reasoning for why you estimate this score before you output the score.

        {% if examples|length > 0 -%}
            Examples:
            ----------
            {%- for example in examples %}
                Text: "{{ example.text }}":
                Output:
                    Reasoning: "{{ example.reasoning }}":
                    Sentiment: "{{ example.sentiment }}"
            {% endfor -%}
            ----------
        {% endif -%}

        ========
        Text: {{ text }}
        Output: 
        """

    # Outlines doesn't make use of a prompt signature description, hence we return None here. 
    @property
    def _prompt_signature_description(self) -> str | None:
        return None

    # We return our SentimentEstimate as prompt signature. 
    @cached_property
    def prompt_signature(self) -> type[pydantic.BaseModel]:
        return SentimentEstimate

    # Outlines as a JSON mode, which allows complex return objects to be specified and parsed as Pydantic objects.
    @property
    def inference_mode(self) -> outlines_.InferenceMode:
        return outlines_.InferenceMode.json
    
    # We copy the result score into our doc's results attribute.
    def integrate(self, results: Iterable[SentimentEstimate], docs: Iterable[Doc]) -> Iterable[Doc]:
        for doc, result in zip(docs, results):
            assert isinstance(result, SentimentEstimate)
            # doc.results is a dict, with the task ID being the key to store our results under for the corresponding 
            # task.
            doc.results[self._task_id] = result.score
        return docs

    # Consolidating multiple chunks for sentiment analysis can be pretty straightforward: we compute the average over 
    # all chunks and assume this to be the sentiment score for the doc.
    def consolidate(
        self, results: Iterable[SentimentEstimate], docs_offsets: list[tuple[int, int]]
    ) -> Iterable[SentimentEstimate]:
        results = list(results)
        
        # Iterate over indices that determine which chunks belong to which documents.
        for doc_offset in docs_offsets:
            # Keep track of all reasonings and the total score.
            reasonings: list[str] = []
            scores = 0.

            # Iterate over chunks' results.
            for chunk_result in results[doc_offset[0] : doc_offset[1]]:
                # Engines may return None results if they encounter errors and run in permissive mode. We ignore such 
                # results.
                if chunk_result:
                    assert isinstance(chunk_result, SentimentEstimate)
                    reasonings.append(chunk_result.reasoning)
                    scores += chunk_result.score
            
            yield SentimentEstimate(
               # Average the score.
               score=scores / (doc_offset[1] - doc_offset[0]),
               # Concatenate all reasonings.
               reasoning=str(reasonings)
            )
```

Our bridge takes care of most of the heavy lifting: it defines how we expect our results to look like,
it consolidates the results we're getting back from the engine, and integrates them into our docs.

### 2. Build a `SentimentAnalysisTask`

The task class itself is mostly glue code: we instantiate our bridge(s) and provide other auxiliary, engine-agnostic 
functionality. We'll save this in `sentiment_analysis_task.py`

```python
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import datasets

from sieves.data import Doc
from sieves.engines import EngineType
from sieves.serialization import Config
from .sentiment_analysis_bridges import OutlinesSentimentAnalysis, SentimentEstimate
from sieves.tasks.predictive.core import PredictiveTask


# We'll define that class we require fewshot examples to be provided in. In our case we can just inherit from our 
# prompt signature class and add a `text` property.
class FewshotExample(SentimentEstimate):
    text: str


class SentimentAnalysis(PredictiveTask[SentimentEstimate, SentimentEstimate, OutlinesSentimentAnalysis]):
    # For the initialization of the bridge. We raise an error if an engine has been specified that we don't support (due 
    # to us not having a bridge implemented that would support this engine type).
    def _init_bridge(self, engine_type: EngineType) -> OutlinesSentimentAnalysis:
       if engine_type == EngineType.outlines:
           return OutlinesSentimentAnalysis(
               task_id=self._task_id,
               prompt_template=self._custom_prompt_template,
               prompt_signature_desc=self._custom_prompt_signature_desc,
           )
       else:
          raise KeyError(f"Engine type {engine_type} is not supported by {self.__class__.__name__}.")
        
    # Represents set of supported engine types.
    @property
    def supports(self) -> set[EngineType]:
        return {EngineType.outlines}

    # This implements the conversion of a set of docs to a Hugging Face datasets.Dataset.
    # You can implement this as `raise NotImplementedError` if you're not interested in generating a Hugging Face 
    # dataset from your result data. 
    def to_hf_dataset(self, docs: Iterable[Doc]) -> datasets.Dataset:
        # Define metadata.
        info = datasets.DatasetInfo(
            description=f"Sentiment estimation dataset. Generated with sieves"
            f"v{Config.get_version()}.",
            features=datasets.Features({"text": datasets.Value("string"), "score": datasets.Value("float32")}),
        )

        def generate_data() -> Iterable[dict[str, Any]]:
            """Yields results as dicts.
            :return Iterable[dict[str, Any]]: Results as dicts.
            """
            for doc in docs:
                yield {"text": doc.text, "score": doc.results[self._task_id]}

        # Create dataset.
        return datasets.Dataset.from_generator(generate_data, features=info.features, info=info)
```

And that's it! Our sentiment analysis task is finished.

### 3. Running our task

We can now use our sentiment analysis task like every built-in task:

```python
from sieves import Doc, Pipeline, Engine
import outlines
from .sentiment_analysis_task import SentimentAnalysis

model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
engine = Engine(model=outlines.models.transformers(model_name))
            
docs = [Doc(text="I'm feeling happy today."), Doc(text="I was sad yesterday.")]
pipe = Pipeline([SentimentAnalysis(engine=engine)])

for doc in pipe(docs):
    print(doc.text, doc.results["SentimentAnalysis"])
```