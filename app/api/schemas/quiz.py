


from typing import Literal

from pydantic import BaseModel, Field

# Re-export QuizOutput from the pipeline — single source of truth.
from app.pipeline.quiz.formatter import QuizOutput  # noqa: F401
from app.pipeline.quiz.grader import GradeResult  # noqa: F401


class QuizRequest(BaseModel):

    type: Literal["lesson", "course"]
    course_id: str
    lesson_id: str | None = None
    file_id: str | None = None
    keywords: list[str] = Field(default_factory=list)
    difficulty: Literal["easy", "medium", "hard"] = Field(default="medium")
    limit_chunks: int = Field(default=20, ge=5, le=100)


class GradeRequest(BaseModel):
    """Request body for the /quiz/grade endpoint."""

    question: str
    reference_answer: str
    user_answer: str


class GradeResponse(BaseModel):
    """Response body for the /quiz/grade endpoint."""

    is_correct: bool
    score: float = Field(ge=0.0, le=1.0)
    explanation: str = ""
