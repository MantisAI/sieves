"""Schemas for question answering task."""

from __future__ import annotations

import dspy
import pydantic

from sieves.model_wrappers import dspy_, langchain_, outlines_
from sieves.tasks.predictive.schemas.core import FewshotExample as BaseFewshotExample


class QuestionAnswer(pydantic.BaseModel):
    """A pair consisting of a question, its generated answer, and a confidence score.

    Attributes:
        question: Question asked.
        answer: Answer to the question.
        score: Confidence score.
    """

    question: str = pydantic.Field(
        description="The question being asked. This should exactly match one of the provided questions."
    )
    answer: str = pydantic.Field(description="The generated answer to the question based on the document.")
    score: float | None = pydantic.Field(
        default=None, description="Provide a confidence score for the generated answer, between 0 and 1."
    )


class FewshotExample(BaseFewshotExample):
    """Few-shot example with questions and answers for a context.

    Attributes:
        text: Input text.
        questions: Questions asked.
        answers: Expected answers.
        scores: Confidence scores for answers.
    """

    text: str
    questions: list[str]
    answers: list[str]
    scores: list[float] | None = None

    @property
    def input_fields(self) -> tuple[str, ...]:
        """Return input fields.

        :return: Input fields.
        """
        return ("text", "questions")

    @property
    def target_fields(self) -> tuple[str, ...]:
        """Return target fields.

        :return: Target fields.
        """
        return ("answers", "scores")

    def to_dspy(self) -> dspy.Example:
        """Convert to `dspy.Example` with qa_pairs.

        :returns: Example as `dspy.Example`.
        """
        scores = self.scores or [None] * len(self.answers)
        qa_pairs = [
            QuestionAnswer(
                question=q,
                answer=a,
                score=s,
            )
            for q, a, s in zip(self.questions, self.answers, scores)
        ]

        return dspy.Example(text=self.text, questions=self.questions, qa_pairs=qa_pairs).with_inputs(*self.input_fields)


# --8<-- [start:Result]
class Result(pydantic.BaseModel):
    """Result of a question-answering task. Contains a list of question-answer pairs.

    Attributes:
        qa_pairs: List of question-answer pairs.
    """

    qa_pairs: list[QuestionAnswer] = pydantic.Field(
        description="A list of question-answer pairs corresponding to the provided questions."
    )


# --8<-- [end:Result]


TaskModel = dspy_.Model | langchain_.Model | outlines_.Model
TaskPromptSignature = type[dspy.Signature] | type[pydantic.BaseModel]
TaskResult = Result
