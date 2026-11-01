"""Generate an NIAT-format lesson plan from the supplied syllabus image.

The current Modal deployment runs the text-only Qwen3.5-9B IQ4_XS GGUF, so the
syllabus table from the image is supplied as transcribed text. The model output
is validated as JSON and rendered to a formatted A4 PDF.

Run from ragpy:
    uv run --with reportlab python experiments/lesson_plan_experiment.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.core.llms import ChatMessage  # noqa: E402
from llama_index.core.prompts.base import PromptTemplate  # noqa: E402
from pydantic import BaseModel, Field, ValidationError, field_validator  # noqa: E402

from app.llm.factory import get_llm  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
SUBJECT = "BASIC ELECTRICAL (20)"


class TrainingAid(BaseModel):
    ser: str = ""
    description: str = ""
    remarks: str = ""


class DevelopmentRow(BaseModel):
    sub: str = ""
    topics: list[str] = Field(default_factory=list)
    content: str = ""
    method: str = ""
    trg_aids: str = ""
    time: str | int = ""

    @field_validator("time", mode="before")
    @classmethod
    def _time_to_str(cls, value):
        return str(value)


class PFTRow(BaseModel):
    ser: str = "1"
    syllabus: str = ""
    references: str = ""
    related_lps: str = ""


class PERow(BaseModel):
    item: str = ""
    description: str = ""
    remarks: str = ""


_PROSE_REPLACEMENTS = (
    ("or similar the available training equipment", "or similar training aircraft"),
    ("on a the available training equipment", "on the available training equipment"),
    ("a the available training equipment", "the available training equipment"),
    ("a the ", "the "),
    ("Pilot-Static", "Pitot-Static"),
    ("pilot-static", "pitot-static"),
    ("Pilot Static", "Pitot-Static"),
    ("pilot static", "pitot-static"),
    ("Mock-ups", "Training aids"),
    ("mock-ups", "training aids"),
    ("Mock-up", "Training aid"),
    ("mock-up", "training aid"),
)


def _fix_prose(text: str) -> str:
    for old, new in _PROSE_REPLACEMENTS:
        text = text.replace(old, new)
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip()


def sanitize_plan_prose(plan: LessonPlan) -> LessonPlan:
    """Fix known equipment/phrasing artifacts without touching source syllabus text."""
    plan.practical = _fix_prose(plan.practical)
    plan.home_assignment_question = _fix_prose(plan.home_assignment_question)
    plan.intro_questions = [_fix_prose(question) for question in plan.intro_questions]
    plan.summary_questions = [_fix_prose(question) for question in plan.summary_questions]
    for row in plan.development:
        row.topics = [_fix_prose(topic) for topic in row.topics if topic.strip()]
        row.content = _fix_prose(row.content)
    for aid in plan.training_aids:
        aid.description = _fix_prose(aid.description)
        aid.remarks = _fix_prose(aid.remarks)
    return plan


class LessonPlan(BaseModel):
    subject_name: str = ""
    subject_code: str = ""
    class_name: str = ""
    lesson_plan_no: str = ""
    total_lesson_plans: str = ""
    time_allotted: str = ""

    topic: str = ""
    total_sessions: str = ""
    sequence: str = ""
    syllabus_extract: str = ""
    specific_objectives: str = ""
    training_aids: list[TrainingAid] = Field(default_factory=list)
    practical: str = ""
    existing_knowledge: str = ""

    intro_time: str | int = "10 min"
    intro_questions: list[str] = Field(default_factory=list)

    development: list[DevelopmentRow] = Field(default_factory=list)

    summary_time: str | int = "15 min"
    summary_questions: list[str] = Field(default_factory=list)

    home_assignment_time: str | int = "5 min"
    home_assignment_question: str = ""

    pft_rows: list[PFTRow] = Field(default_factory=list)
    pe_rows: list[PERow] = Field(default_factory=list)

    prepared_by: str = "Name, Rank, P No."
    approved_by: str = "Name, Rank, P No."
    prep_date: str = "Nov 26"
    approval_date: str = "Nov 26"

    queries: list[str] = Field(default_factory=list)
    timing_check: str = ""

    @field_validator("intro_time", "summary_time", "home_assignment_time", mode="before")
    @classmethod
    def _time_value_to_str(cls, value):
        return str(value)


INPUTS = """
Subject: BASIC ELECTRICAL (20)
Class / course: Air Agniveer Apprentice (Engineering)
Total no. of lesson plans in subject: Not specified in source
Lesson plan no.: 01
Time allotted for session: 80 min
Topic: Aircraft Pressure Gauges, Pilot-Static System, Altimeter, ASI and VSI
Total sessions for this topic / sequence of this session: 01 / 01
Syllabus extract: Purpose and principle operation of aircraft pressure gauges, pilot static system, altimeter, ASI, and VSI. Practical on Pitot & Static instruments. Safety precautions while working on system.
Specific objectives: 20.1.1.1 Trainee will be able to understand instrument gauges, instrument type auto pilot and maintenance.
Reference book(s)/BR: Aircraft Instruments and Integrated Systems - E.H.J. Pallett (suggested; confirm)
Available training aids: Nil specified; default to L & Ds and IB, PDF
Existing knowledge of the class: Basic electrical principles, fundamentals of flight, and general aircraft systems familiarization
Next lesson plan no. and its syllabus: LP 02 - Capacitor type fuel quantity indicator, angle of attack indicator, gyroscopic instruments attitude indicator, turn and slip indicator, direction indicator. Practical on various instruments.
Prepared by / approved by: Name, Rank, P No.
Date of preparation / approval: Nov 26
"""

PROMPT = PromptTemplate(
    template=(
        "ROLE\n"
        "You are an experienced technical instructor and curriculum designer at the Naval Institute "
        "of Aeronautical Technology. You write lesson plans for aircrew/technical trainees in the "
        "exact NIAT lesson plan format described below.\n\n"
        "INPUTS\n"
        "{inputs}\n\n"
        "TASK\n"
        "Produce one complete lesson plan covering the supplied syllabus extract. Reproduce the "
        "NIAT structure exactly: Part A items 1-8, Part B items 9-12, Part C PFT table, Part D "
        "personal experiences table, Part E review sheet, and the preparation/approval block.\n"
        "Part B item 10 must have sub-points (a), (b), (c) with Method, Trg Aids and Time. "
        "Each development row must contain 3-5 concise topic bullets in the topics list. "
        "Each bullet is a short topic or sub-topic heading (about 3-10 words), not a sentence, "
        "not an explanation, and not a paragraph. Do not describe or teach the topic in this "
        "field; only name the topics covered. The content field may repeat the same bullets "
        "joined by newlines, but topics is authoritative. The practical row must name the "
        "practical activity, safety precautions, and instructor supervision as short topics. "
        "Part B item 11 must have 2-3 questions with Method, Trg Aids and Time. "
        "Part B item 12 must have one home-assignment question and time.\n\n"
        "TIMING\n"
        "Use exactly 80 minutes total: introduction 10 min, main lesson 50 min, summary and "
        "evaluation 15 min, home assignment 5 min. Use exactly three development rows with times "
        "15, 20, and 15 minutes. Do not change these times.\n\n"
        "RULES\n"
        "1. Use only the supplied syllabus extract; do not add topics beyond it.\n"
        "2. Make introduction questions recall-level, evaluation questions understanding or "
        "application-level, and the home assignment a bridge to LP 02.\n"
        "3. Use only standard, textbook-accurate technical content. If an input is doubtful, put "
        "it in the queries list instead of inventing facts.\n"
        "4. Training aids table must have six rows: (a) BR/Text Book/Docket, (b) Wall Charts, "
        "(c) Films/Duration, (d) PPT, (e) Models, (f) CBT. Use 'Nil' where none apply.\n"
        "5. Use the abbreviations L & Ds, IB, WB, PDF, PPT.\n"
        "6. Keep the language plain, formal, and suitable for an instructor.\n\n"
        "OUTPUT\n"
        "Return a single valid JSON object matching this schema exactly. Do not wrap it in "
        "markdown. Use strings for all text and lists for repeated sections.\n"
        "{{\n"
        '  "class_name": "...",\n'
        '  "lesson_plan_no": "...",\n'
        '  "total_lesson_plans": "...",\n'
        '  "time_allotted": "80 min",\n'
        '  "topic": "...",\n'
        '  "total_sessions": "01 / 01",\n'
        '  "sequence": "01 / 01",\n'
        '  "syllabus_extract": "exact wording",\n'
        '  "specific_objectives": "exact wording",\n'
        '  "training_aids": [{{"ser":"(a)","description":"...","remarks":"..."}}],\n'
        '  "practical": "...",\n'
        '  "existing_knowledge": "...",\n'
        '  "intro_time": "10 min",\n'
        '  "intro_questions": ["...", "...", "..."],\n'
        '  "development": [{{"sub":"(a)","topics":["Topic heading 1","Topic heading 2"],"content":"Topic heading 1\\nTopic heading 2","method":"...","trg_aids":"...","time":"15 min"}}],\n'
        '  "summary_time": "15 min",\n'
        '  "summary_questions": ["...", "..."],\n'
        '  "home_assignment_time": "5 min",\n'
        '  "home_assignment_question": "...",\n'
        '  "pft_rows": [{{"ser":"1","syllabus":"...","references":"...","related_lps":"LP 02"}}],\n'
        '  "pe_rows": [{{"item":"(a)","description":"Adequacy of time","remarks":""}}],\n'
        '  "prepared_by": "Name, Rank, P No.",\n'
        '  "approved_by": "Name, Rank, P No.",\n'
        '  "prep_date": "Nov 26",\n'
        '  "approval_date": "Nov 26",\n'
        '  "queries": ["..."],\n'
        '  "timing_check": "Introduction 10 + Main 50 + Summary 15 + Home 5 = 80 min"\n'
        "}}\n"
    )
)


def _clean_plan(plan: LessonPlan) -> LessonPlan:
    # The source did not list any training aids, so do not let the model invent
    # wall charts, films, models or CBT sessions. Default to the allowed set.
    plan.topic = "Aircraft Pressure Gauges, Pitot-Static System, Altimeter, ASI and VSI"
    plan.training_aids = [
        TrainingAid(
            ser="(a)",
            description="BR/Text Book/Docket",
            remarks="Aircraft Instruments and Integrated Systems - E.H.J. Pallett (suggested; confirm)",
        ),
        TrainingAid(ser="(b)", description="Wall Charts", remarks="Nil"),
        TrainingAid(ser="(c)", description="Films/Duration", remarks="Nil"),
        TrainingAid(
            ser="(d)",
            description="PPT",
            remarks="Objective: understand pressure gauges and pitot-static instruments",
        ),
        TrainingAid(ser="(e)", description="Models", remarks="Nil"),
        TrainingAid(ser="(f)", description="CBT", remarks="Nil"),
    ]

    plan.pft_rows = [
        PFTRow(
            ser="1",
            syllabus=(
                "Capacitor type fuel quantity indicator, angle of attack indicator, "
                "gyroscopic instruments attitude indicator, turn and slip indicator, "
                "direction indicator. Practical on various instruments."
            ),
            references="Aircraft Instruments and Integrated Systems - E.H.J. Pallett",
            related_lps="LP 02",
        )
    ]

    plan.pe_rows = [
        PERow(item="(a)", description="Adequacy of time"),
        PERow(item="(b)", description="Suitability of training material/aids and additional aids required"),
        PERow(item="(c)", description="Doubts/intelligent questions raised by trainees"),
        PERow(item="(d)", description="Difficulties faced by instructor"),
        PERow(item="(e)", description="Other experience"),
    ]

    plan.practical = (
        plan.practical.replace("trainer aircraft or training equipment", "the available training equipment")
        .replace("trainer aircraft", "the available training equipment")
        .replace("mock-up or aircraft", "the available training equipment")
        .replace("mock-ups", "training equipment")
        .replace("mock-up", "training equipment")
        .replace("on a the available training equipment", "on the available training equipment")
        .replace("a the available training equipment", "the available training equipment")
        .replace("Pilot-Static", "Pitot-Static")
        .replace("pilot-static", "pitot-static")
    )

    plan.development = plan.development[:3]
    fixed_times = ["15 min", "20 min", "15 min"]
    for index, row in enumerate(plan.development):
        row.time = fixed_times[index]
        row.content = (
            row.content.replace("Pilot-Static", "Pitot-Static")
            .replace("pilot-static", "pitot-static")
            .replace("Wall Charts", "whiteboard sketches")
            .replace("wall charts", "whiteboard sketches")
            .replace("using Models", "using the available diagrams")
            .replace("using models", "using the available diagrams")
            .replace("instrument mock-ups", "the instrument panel")
            .replace("Models", "diagrams")
            .replace("models", "diagrams")
            .replace("CBT", "self-study")
        )
        if index == 0:
            row.method = "L & Ds"
            row.trg_aids = "PPT, IB"
        elif index == 1:
            row.method = "L & Ds"
            row.trg_aids = "PPT, PDF"
        else:
            row.method = "Practical"
            row.trg_aids = "IB, PDF"

    plan.intro_time = "10 min"
    plan.summary_time = "15 min"
    plan.home_assignment_time = "5 min"
    plan.timing_check = "Introduction 10 + Main 50 + Summary 15 + Home 5 = 80 min"

    known_queries = [
        "Total number of lesson plans in the subject and the exact lesson-plan number were not in the source.",
        "The image says 'pilot static system'; confirm whether this is a typo for 'pitot-static system'.",
        "Reference book/BR was not specified; the suggested Pallett reference should be confirmed.",
        "The source objective (20.1.1.1) is broader than the selected syllabus row; confirm the lesson-specific objective.",
        "Prepared-by/approved-by names, ranks, P numbers and dates must be filled before approval.",
    ]
    merged: list[str] = []
    seen: set[str] = set()
    for query in known_queries + plan.queries:
        if query and query not in seen:
            seen.add(query)
            merged.append(query)
    plan.queries = merged
    return plan


def _subject_label(plan: LessonPlan) -> str:
    name = (plan.subject_name or "").strip()
    code = (plan.subject_code or "").strip()
    if name and code:
        return f"{name.upper()} ({code})"
    if name:
        return name.upper()
    return SUBJECT


def _development_content(row: DevelopmentRow) -> str:
    topics = [topic.strip() for topic in row.topics if topic.strip()]
    if not topics:
        return row.content
    from xml.sax.saxutils import escape

    return "<br/>".join(f"&bull; {escape(topic)}" for topic in topics)


def _render_pdf(plan: LessonPlan, path: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "title", parent=styles["Title"], fontName="Times-Bold", fontSize=13, leading=16, alignment=TA_CENTER
    )
    subtitle = ParagraphStyle(
        "subtitle", parent=styles["Normal"], fontName="Times-Bold", fontSize=10, leading=13, alignment=TA_CENTER
    )
    heading = ParagraphStyle(
        "heading", parent=styles["Heading2"], fontName="Times-Bold", fontSize=10.5, leading=13, spaceBefore=8, spaceAfter=4
    )
    body = ParagraphStyle(
        "body", parent=styles["Normal"], fontName="Times-Roman", fontSize=8.5, leading=11, alignment=TA_LEFT
    )
    small = ParagraphStyle(
        "small", parent=body, fontSize=7.5, leading=9.5
    )
    cell = ParagraphStyle("cell", parent=small)
    cell_bold = ParagraphStyle("cell_bold", parent=cell, fontName="Times-Bold")

    def p(text, style=body):
        return Paragraph(str(text or "").replace("\n", "<br/>"), style)

    def header_table():
        data = [
            [p("Class", cell_bold), p(plan.class_name, cell), p("Lesson Plan No.", cell_bold), p(plan.lesson_plan_no, cell)],
            [
                p("Total No. of Lesson Plans", cell_bold),
                p(plan.total_lesson_plans, cell),
                p("Time Allotted for the Session", cell_bold),
                p(plan.time_allotted, cell),
            ],
        ]
        table = Table(data, colWidths=[38 * mm, 52 * mm, 42 * mm, 40 * mm])
        table.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.6, colors.black),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                    ("BACKGROUND", (2, 0), (2, -1), colors.whitesmoke),
                ]
            )
        )
        return table

    def two_col_table(rows, widths=(45 * mm, 127 * mm), header=None):
        data = []
        if header:
            data.append([p(header[0], cell_bold), p(header[1], cell_bold)])
        for left, right in rows:
            data.append([p(left, cell_bold), p(right, cell)])
        table = Table(data, colWidths=widths, repeatRows=1 if header else 0)
        table.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                ]
            )
        )
        return table

    subject_label = _subject_label(plan)
    story = [
        p("LESSON PLAN", title),
        p(subject_label, subtitle),
        Spacer(1, 4),
        header_table(),
        Spacer(1, 6),
        p("PART A: GUIDELINES / GENERAL INSTRUCTIONS FOR INSTRUCTORS", heading),
    ]

    story.append(
        two_col_table(
            [
                ("1. Topic", plan.topic),
                ("2. Total No. of Sessions for this Topic", plan.total_sessions),
                ("3. Sequence of this Session", plan.sequence),
                ("4. Syllabus Extract", plan.syllabus_extract),
                ("5. Specific Objectives", plan.specific_objectives),
            ]
        )
    )
    story.append(Spacer(1, 4))
    story.append(p("6. List of Training Aids", cell_bold))
    aid_data = [[p("Ser", cell_bold), p("Description", cell_bold), p("Remarks", cell_bold)]]
    for aid in plan.training_aids:
        aid_data.append([p(aid.ser, cell), p(aid.description, cell), p(aid.remarks, cell)])
    aid_table = Table(aid_data, colWidths=[15 * mm, 95 * mm, 62 * mm])
    aid_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
            ]
        )
    )
    story.extend(
        [
            aid_table,
            Spacer(1, 4),
            two_col_table(
                [
                    ("7. Practical in Lab/Equipment Rooms/Simulators", plan.practical),
                    ("8. Existing Knowledge of the Class", plan.existing_knowledge),
                ]
            ),
            Spacer(1, 8),
            p("PART B: DEVELOPMENT OF LESSON", heading),
            two_col_table(
                [
                    ("9. Checking of Previous Knowledge / Introduction", f"Time: {plan.intro_time}"),
                ]
            ),
        ]
    )
    for index, question in enumerate(plan.intro_questions, 1):
        story.append(p(f"({chr(96 + index)}) {question}", body))
    story.append(Spacer(1, 4))
    story.append(p("10. Development of Main Lesson", cell_bold))
    dev_data = [[p("Sub", cell_bold), p("Content", cell_bold), p("Method", cell_bold), p("Trg Aids", cell_bold), p("Time", cell_bold)]]
    for row in plan.development:
        dev_data.append(
            [
                p(row.sub, cell),
                p(_development_content(row), cell),
                p(row.method, cell),
                p(row.trg_aids, cell),
                p(str(row.time), cell),
            ]
        )
    dev_table = Table(dev_data, colWidths=[12 * mm, 84 * mm, 26 * mm, 30 * mm, 20 * mm], repeatRows=1)
    dev_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
            ]
        )
    )
    story.extend([dev_table, Spacer(1, 4), p("11. Summary and Evaluation", cell_bold)])
    summary_meta = Table(
        [
            [p("Method", cell_bold), p("Trg Aids", cell_bold), p("Time", cell_bold)],
            [p("Q & A", cell), p("IB", cell), p(str(plan.summary_time), cell)],
        ],
        colWidths=[45 * mm, 81 * mm, 46 * mm],
    )
    summary_meta.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(summary_meta)
    for index, question in enumerate(plan.summary_questions, 1):
        story.append(p(f"({chr(96 + index)}) {question}", body))
    story.extend(
        [
            Spacer(1, 4),
            two_col_table(
                [
                    ("12. Home Assignment and Follow Up", plan.home_assignment_question),
                    ("Time", str(plan.home_assignment_time)),
                ]
            ),
            Spacer(1, 8),
            p("PART C: PREPARATION FOR TOMORROW (PFT) DIRECTIVES FOR TRAINEES", heading),
        ]
    )
    pft_data = [[p("Ser", cell_bold), p("Syllabus for PFT", cell_bold), p("References", cell_bold), p("Related LPs", cell_bold)]]
    for row in plan.pft_rows:
        pft_data.append([p(row.ser, cell), p(row.syllabus, cell), p(row.references, cell), p(row.related_lps, cell)])
    pft_table = Table(pft_data, colWidths=[12 * mm, 95 * mm, 45 * mm, 20 * mm])
    pft_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
            ]
        )
    )
    story.extend([pft_table, Spacer(1, 8), p("PART D: PERSONAL EXPERIENCES (PE) OF INSTRUCTOR", heading)])
    pe_data = [[p("Ser", cell_bold), p("Item", cell_bold), p("Remarks", cell_bold)]]
    for row in plan.pe_rows:
        pe_data.append([p(row.item, cell), p(row.description, cell), p(row.remarks, cell)])
    pe_table = Table(pe_data, colWidths=[15 * mm, 97 * mm, 60 * mm])
    pe_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
            ]
        )
    )
    story.extend(
        [
            pe_table,
            Spacer(1, 10),
            Table(
                [
                    [p("Instructor", cell_bold), p("HoF (E/L)", cell_bold)],
                    [p("Signature: ______________________", small), p("Signature: ______________________", small)],
                ],
                colWidths=[86 * mm, 86 * mm],
                style=TableStyle(
                    [
                        ("LINEABOVE", (0, 0), (-1, 0), 0.6, colors.black),
                        ("LINEBELOW", (0, 0), (-1, -1), 0.6, colors.black),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                ),
            ),
            PageBreak(),
            p("PART E: REVIEW SHEET", heading),
        ]
    )
    review_data = [
        [p("Date", cell_bold), p("Instructor's Comment", cell_bold), p("HOF (E/L)", cell_bold), p("Oi/C TDEC Remarks", cell_bold)]
    ]
    for _ in range(6):
        review_data.append([p("", cell), p("", cell), p("", cell), p("", cell)])
    review_table = Table(review_data, colWidths=[24 * mm, 64 * mm, 40 * mm, 44 * mm], rowHeights=12 * mm)
    review_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.extend(
        [
            review_table,
            Spacer(1, 14),
            p("Prepared by / Approved by", cell_bold),
            Table(
                [
                    [p("Prepared by", cell_bold), p("Approved by", cell_bold)],
                    [p(plan.prepared_by, cell), p(plan.approved_by, cell)],
                    [p(plan.prep_date, cell), p(plan.approval_date, cell)],
                ],
                colWidths=[86 * mm, 86 * mm],
                style=TableStyle(
                    [
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                    ]
                ),
            ),
            Spacer(1, 12),
            p("Queries for Instructor", heading),
        ]
    )
    if plan.queries:
        for index, query in enumerate(plan.queries, 1):
            story.append(p(f"{index}. {query}", body))
    else:
        story.append(p("None.", body))
    story.extend(
        [
            Spacer(1, 8),
            p("Timing Check", heading),
            p(plan.timing_check or "Introduction 10 + Main 50 + Summary 15 + Home 5 = 80 min", body),
        ]
    )

    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Times-Roman", 7.5)
        canvas.drawString(
            18 * mm, 10 * mm, f"{subject_label} - Lesson Plan {plan.lesson_plan_no}"
        )
        canvas.drawRightString(192 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=14 * mm,
        bottomMargin=16 * mm,
        title=f"{subject_label} Lesson Plan",
        author="NIAT / AeroMentor",
    )
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)


async def generate_plan_from_inputs(inputs: str) -> LessonPlan:
    prompt = (
        PROMPT.template.replace("{inputs}", inputs.strip())
        .replace("{{", "{")
        .replace("}}", "}")
    )
    llm = get_llm()
    llm.max_tokens = 8192
    llm.temperature = 0.1
    messages = [ChatMessage(role="user", content=prompt)]

    last_error: ValidationError | None = None
    for attempt in range(2):
        response = await llm.achat(messages, response_format={"type": "json_object"})
        content = response.message.content or ""
        try:
            return LessonPlan.model_validate_json(content)
        except ValidationError as err:
            last_error = err
            (OUTPUT_DIR / "raw_failed_output.json").write_text(content)
            print(f"validation attempt {attempt + 1} failed: {err}")
    assert last_error is not None
    raise last_error


async def generate_plan() -> LessonPlan:
    return await generate_plan_from_inputs(INPUTS)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-json",
        default="",
        help="Render an existing lesson-plan JSON instead of calling the model.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "basic_electrical_20_lp01.json"
    pdf_path = OUTPUT_DIR / "basic_electrical_20_lp01.pdf"

    if args.from_json:
        plan = LessonPlan.model_validate_json(Path(args.from_json).read_text())
        plan = _clean_plan(plan)
        json_path.write_text(plan.model_dump_json(indent=2))
        _render_pdf(plan, pdf_path)
        print(f"rendered from {args.from_json}")
        print(f"pdf:  {pdf_path}")
        print("timing:", plan.timing_check)
        return 0

    started = time.perf_counter()
    plan = asyncio.run(generate_plan())
    elapsed = time.perf_counter() - started
    plan = _clean_plan(plan)
    json_path.write_text(plan.model_dump_json(indent=2))
    _render_pdf(plan, pdf_path)

    print(f"generated in {elapsed:.1f}s")
    print(f"json: {json_path}")
    print(f"pdf:  {pdf_path}")
    print("timing:", plan.timing_check)
    return 0


if __name__ == "__main__":
    sys.exit(main())
