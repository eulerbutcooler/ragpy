from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single turn in the conversation history."""

    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):

    course_ids: list[str]
    query: str = Field(min_length=1, max_length=2000)
    stream: bool = True
    history: list[ChatMessage] = Field(
        default_factory=list,
        description="Previous conversation turns for context. Last N messages.",
        max_length=20,
    )


class CitationItem(BaseModel):

    file_name: str
    file_id: str
    score: float | None = None


class ChatResponse(BaseModel):

    answer: str
    citations: list[CitationItem]


class TitleRequest(BaseModel):
    """First user message; RAG summarizes it into a short chat title."""

    message: str = Field(min_length=1, max_length=2000)


class TitleResponse(BaseModel):

    title: str
