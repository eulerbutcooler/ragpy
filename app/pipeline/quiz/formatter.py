"""Pydantic output models and structured quiz formatting from node context."""


import logging
import re
from typing import Any, Literal

from llama_index.core.schema import BaseNode
from pydantic import BaseModel, model_validator

from app.llm.factory import get_llm
from app.llm.prompts import QUIZ_GENERATION_PROMPT, TOPIC_SYNTHESIS_PROMPT
from app.llm.structured import astructured_predict_json

logger = logging.getLogger(__name__)


class QuizChoice(BaseModel):
    """A single answer choice for an MCQ question."""

    label: str
    text: str


class QuizQuestion(BaseModel):
    """A single quiz question, either MCQ or open-ended."""

    type: Literal["mcq", "open_ended"] = "mcq"
    question: str
    choices: list[QuizChoice] | None = None
    answer: str = "A"
    explanation: str | None = None
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    topic_phrase: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_question(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Normalize question text
            if not data.get("question"):
                data["question"] = data.get("text") or data.get("prompt") or data.get("query") or "Aviation technical query"

            # Normalize choices / options
            raw_choices = data.get("choices") or data.get("options")
            if isinstance(raw_choices, list) and raw_choices:
                parsed_choices = []
                for idx, c in enumerate(raw_choices):
                    if isinstance(c, str):
                        m = re.match(r"^([A-Da-d])[\.\:\)]\s*(.*)$", c.strip())
                        if m:
                            parsed_choices.append({"label": m.group(1).upper(), "text": m.group(2).strip()})
                        else:
                            label = chr(ord('A') + idx) if idx < 4 else str(idx + 1)
                            parsed_choices.append({"label": label, "text": c.strip()})
                    elif isinstance(c, dict):
                        label = c.get("label", "")
                        text = c.get("text") or c.get("value") or c.get("choice") or ""
                        if not label and text:
                            m = re.match(r"^([A-Da-d])[\.\:\)]\s*(.*)$", str(text).strip())
                            if m:
                                label = m.group(1).upper()
                                text = m.group(2).strip()
                            else:
                                label = chr(ord('A') + idx)
                        parsed_choices.append({"label": str(label).upper() or chr(ord('A') + idx), "text": str(text)})
                data["choices"] = parsed_choices
            elif not raw_choices and data.get("type", "mcq") == "mcq":
                data["choices"] = [
                    {"label": "A", "text": "True / Standard Operation"},
                    {"label": "B", "text": "False / Abnormal Operation"},
                ]

            # Normalize answer
            raw_ans = data.get("answer") or data.get("correct_answer") or data.get("correctAnswer") or data.get("solution")
            if not raw_ans:
                data["answer"] = "A"
            else:
                ans = str(raw_ans).strip()
                m = re.match(r"^([A-Da-d])\b", ans)
                data["answer"] = m.group(1).upper() if m else ans

            # Normalize explanation
            if not data.get("explanation"):
                data["explanation"] = data.get("reasoning") or data.get("rationale") or data.get("justification") or ""

        return data


class TopicSlide(BaseModel):
    """A single micro-learning educational slide."""

    slide_number: int = 1
    slide_type: str = "concept"
    title: str = "Topic Overview"
    bullets: list[str] = []
    formula_or_rule: str | None = None
    diagram_mermaid: str | None = None
    warning: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_slide(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Title
            if not data.get("title"):
                data["title"] = data.get("name") or data.get("heading") or data.get("topic") or "Topic Briefing"

            # Bullets / Content
            raw_bullets = (
                data.get("bullets")
                or data.get("bullet_points")
                or data.get("points")
                or data.get("content")
                or data.get("key_points")
            )
            if isinstance(raw_bullets, list):
                data["bullets"] = [str(b).strip("- ").strip() for b in raw_bullets if str(b).strip()]
            elif isinstance(raw_bullets, str):
                data["bullets"] = [line.strip("- ").strip() for line in raw_bullets.splitlines() if line.strip()]

            # Formula / Equation aliases and sanitation
            raw_formula = data.get("formula_or_rule") or data.get("formula") or data.get("equation") or data.get("rule") or data.get("limits")
            if raw_formula and isinstance(raw_formula, str):
                f_str = raw_formula.strip()
                if f_str.lower() in ("null", "none", "n/a", "no formula", "none.", ""):
                    data["formula_or_rule"] = None
                else:
                    data["formula_or_rule"] = f_str
            else:
                data["formula_or_rule"] = None

            # Diagram aliases and sanitation
            raw_diag = data.get("diagram_mermaid") or data.get("diagram") or data.get("mermaid") or data.get("chart")
            if raw_diag and isinstance(raw_diag, str):
                diag_str = raw_diag.strip()
                if diag_str.lower() in ("null", "none", "n/a", "no diagram", "none.", ""):
                    data["diagram_mermaid"] = None
                else:
                    data["diagram_mermaid"] = diag_str
            else:
                data["diagram_mermaid"] = None

            # Warning / Note / Takeaway aliases
            if not data.get("warning"):
                data["warning"] = data.get("note") or data.get("takeaway") or data.get("caution") or data.get("emergency_procedure")

            # Slide type (flexible string)
            st = str(data.get("slide_type") or data.get("type") or "concept").lower()
            data["slide_type"] = st

        return data


class TopicSynthesisOutput(BaseModel):
    """Combined output for a topic: flexible slides + question bank items."""

    slides: list[TopicSlide] = []
    questions: list[QuizQuestion] = []

    @model_validator(mode="before")
    @classmethod
    def normalize_synthesis(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Capture slides from any possible key the LLM might use
            raw_slides = (
                data.get("slides")
                or data.get("topic_slides")
                or data.get("micro_learning_slides")
                or data.get("presentation")
                or data.get("lesson_slides")
                or data.get("deck")
                or []
            )
            if isinstance(raw_slides, dict):
                raw_slides = list(raw_slides.values())

            if isinstance(raw_slides, list):
                for idx, s in enumerate(raw_slides):
                    if isinstance(s, dict):
                        if not s.get("slide_number"):
                            s["slide_number"] = idx + 1
                data["slides"] = raw_slides

            # Capture questions from any possible key the LLM might use
            raw_questions = (
                data.get("questions")
                or data.get("quiz")
                or data.get("quiz_questions")
                or data.get("examination_questions")
                or data.get("mcqs")
                or data.get("mcq_questions")
                or data.get("exam_questions")
                or data.get("practice_questions")
                or []
            )
            if isinstance(raw_questions, dict):
                raw_questions = list(raw_questions.values())

            if isinstance(raw_questions, list):
                data["questions"] = raw_questions

        return data


class TopicSummary(BaseModel):
    """Summary of a topic with its slides."""

    title: str
    slides: list[TopicSlide] = []


class QuizOutput(BaseModel):
    """Structured quiz output for a course or lesson."""

    course_id: str = ""
    lesson_id: str | None = None
    questions: list[QuizQuestion]
    topics: list[TopicSummary] = []


async def synthesize_topic(
    nodes: list[BaseNode],
    topic_phrase: str,
) -> TopicSynthesisOutput:
    """Generate 4 micro-learning slides + 6 questions for a specific topic."""
    context_parts: list[str] = []
    for node in nodes:
        text = node.text or ""
        if text:
            context_parts.append(text)

    context_str = "\n\n---\n\n".join(context_parts)

    llm = get_llm()
    try:
        result: TopicSynthesisOutput = await astructured_predict_json(
            llm,
            TopicSynthesisOutput,
            TOPIC_SYNTHESIS_PROMPT,
            context_str=context_str,
            topic_phrase=topic_phrase,
        )
        # Tag each question with its topic phrase
        for q in result.questions:
            q.topic_phrase = topic_phrase
        logger.info(
            "topic_synthesized topic=%s slides=%d questions=%d",
            topic_phrase,
            len(result.slides),
            len(result.questions),
        )
        return result
    except Exception as e:
        logger.error("failed_to_synthesize_topic topic=%s err=%s", topic_phrase, e)
        return TopicSynthesisOutput(slides=[], questions=[])



async def format_quiz(nodes: list[BaseNode], course_id: str, difficulty: str = "medium", quiz_type: str = "course") -> QuizOutput:
    """Call LLM with structured prediction to produce a QuizOutput from nodes."""
    context_parts: list[str] = []
    for node in nodes:
        text = node.text or ""
        qa_meta = node.metadata.get("questions_this_excerpt_can_answer", "")
        if text:
            context_parts.append(text)
        if qa_meta:
            context_parts.append(f"Q&A hints: {qa_meta}")

    context_str = "\n\n---\n\n".join(context_parts)

    if quiz_type == "lesson":
        requirements = (
            "- Generate Exactly 5 questions total.\n"
            "- All 5 questions must be multiple-choice (type: \"mcq\"). Do not generate any open_ended questions.\n"
        )
    else:
        requirements = (
            f"Difficulty level: {difficulty}\n"
            "- easy: Basic recall and definition questions. Straightforward single-concept answers.\n"
            "- medium: Application and understanding questions. May require connecting two concepts.\n"
            "- hard: Analysis and synthesis questions. Requires deep understanding and multi-step reasoning.\n"
            "- Generate Exactly 10 questions total.\n"
            "- 8 must be multiple-choice (type: \"mcq\") and 2 open-ended (type: \"open_ended\").\n"
        )

    llm = get_llm()
    # llama.cpp does not return OpenAI tool_calls, so use prompt-enforced JSON and
    # Pydantic validation instead of function-calling structured prediction.
    result: QuizOutput = await astructured_predict_json(
        llm,
        QuizOutput,
        QUIZ_GENERATION_PROMPT,
        context_str=context_str,
        requirements=requirements,
    )

    # Stamp course_id on the result (LLM may not know it).
    result.course_id = course_id

    logger.info("quiz formatted course_id=%s questions=%d", course_id, len(result.questions))
    return result
