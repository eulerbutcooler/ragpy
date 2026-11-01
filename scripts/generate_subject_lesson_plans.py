"""Generate one NIAT lesson-plan PDF per detailed-syllabus row.

Example:
    uv run --with reportlab python scripts/generate_subject_lesson_plans.py \
      --file "/path/to/AGNIVEER APPRENTICE BASIC AE.docx" \
      --subject-code 20 \
      --subject-name "Basic Electrical"
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docx import Document as DocxDocument  # noqa: E402

from app.pipeline.syllabus.extractor import SyllabusRow, extract_subject_rows  # noqa: E402
from experiments.lesson_plan_experiment import (  # noqa: E402
    LessonPlan,
    PERow,
    PFTRow,
    TrainingAid,
    _render_pdf,
    generate_plan_from_inputs,
    sanitize_plan_prose,
)

TRAINING_AIDS = [
    TrainingAid(
        ser="(a)",
        description="BR/Text Book/Docket",
        remarks="Refer to the master syllabus reference list",
    ),
    TrainingAid(ser="(b)", description="Wall Charts", remarks="Nil"),
    TrainingAid(ser="(c)", description="Films/Duration", remarks="Nil"),
    TrainingAid(
        ser="(d)",
        description="PPT",
        remarks="Objective: teach the syllabus extract for this lesson",
    ),
    TrainingAid(ser="(e)", description="Models", remarks="Nil"),
    TrainingAid(ser="(f)", description="CBT", remarks="Nil"),
]

PE_ROWS = [
    PERow(item="(a)", description="Adequacy of time"),
    PERow(
        item="(b)", description="Suitability of training material/aids and additional aids required"
    ),
    PERow(item="(c)", description="Doubts/intelligent questions raised by trainees"),
    PERow(item="(d)", description="Difficulties faced by instructor"),
    PERow(item="(e)", description="Other experience"),
]


def _topic_hint(row: SyllabusRow) -> str:
    text = row.syllabus or row.sub_task or row.objective
    first = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]
    return first[:140].strip()


def _inputs_for_row(
    subject_name: str,
    subject_code: str,
    row: SyllabusRow,
    next_row: SyllabusRow | None,
    index: int,
    total: int,
    class_name: str,
    session_minutes: int,
) -> str:
    next_text = (
        f"LP {index + 2:02d} - {_topic_hint(next_row)}"
        if next_row is not None
        else "End of subject"
    )
    return f"""
Subject: {subject_name} ({subject_code})
Class / course: {class_name}
Total no. of lesson plans in subject: {total}
Lesson plan no.: {index + 1:02d}
Time allotted for session: {session_minutes} min
Topic: {_topic_hint(row)}
Total sessions for this topic / sequence of this session: 01 / 01
Syllabus extract: {row.syllabus}
Specific objectives: {row.objective}
Reference book(s)/BR: Refer to the master syllabus reference list
Available training aids: Nil specified; default to L & Ds and IB, PDF
Existing knowledge of the class: As per course prerequisites
Next lesson plan no. and its syllabus: {next_text}
Prepared by / approved by: Name, Rank, P No.
Date of preparation / approval: Nov 26
""".strip()


def _normalize_topics(dev, row: SyllabusRow) -> None:
    """Keep Part B item 10 as a short topic outline, not teaching prose."""
    topics = [topic.strip(" -•") for topic in (dev.topics or []) if topic.strip()]
    if not topics and dev.content:
        parts = re.split(r"(?<=[.!?])\s+|\n+", dev.content)
        topics = [part.strip(" -•") for part in parts if part.strip()][:5]

    cleaned: list[str] = []
    for topic in topics[:5]:
        topic = " ".join(topic.split())
        if len(topic.split()) > 14:
            topic = " ".join(topic.split()[:14]).rstrip(",;:") + "…"
        if topic:
            cleaned.append(topic)
    dev.topics = cleaned
    dev.content = "\n".join(cleaned)


def _clean_plan(
    plan: LessonPlan,
    row: SyllabusRow,
    next_row: SyllabusRow | None,
    index: int,
    total: int,
    class_name: str,
    session_minutes: int,
    subject_name: str,
    subject_code: str,
) -> LessonPlan:
    plan.subject_name = subject_name
    plan.subject_code = subject_code
    plan.class_name = class_name
    plan.total_lesson_plans = str(total)
    plan.lesson_plan_no = f"{index + 1:02d}"
    plan.time_allotted = f"{session_minutes} min"
    plan.total_sessions = "01 / 01"
    plan.sequence = "01 / 01"
    plan.syllabus_extract = row.syllabus
    plan.specific_objectives = row.objective
    plan.existing_knowledge = "As per course prerequisites"
    plan.training_aids = list(TRAINING_AIDS)
    plan.pe_rows = list(PE_ROWS)
    plan.pft_rows = [
        PFTRow(
            ser="1",
            syllabus=next_row.syllabus if next_row is not None else "End of subject",
            references="Refer to the master syllabus reference list",
            related_lps=f"LP {index + 2:02d}" if next_row is not None else "-",
        )
    ]
    plan.practical = (
        plan.practical.replace(
            "trainer aircraft or training equipment", "the available training equipment"
        )
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
    for dev_index, dev in enumerate(plan.development):
        dev.time = fixed_times[dev_index]
        _normalize_topics(dev, row)
        dev.content = (
            dev.content.replace("Pilot-Static", "Pitot-Static")
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
        if dev_index == 0:
            dev.method = "L & Ds"
            dev.trg_aids = "PPT, IB"
        elif dev_index == 1:
            dev.method = "L & Ds"
            dev.trg_aids = "PPT, PDF"
        else:
            dev.method = "Practical"
            dev.trg_aids = "IB, PDF"

    plan.intro_time = "10 min"
    plan.summary_time = "15 min"
    plan.home_assignment_time = "5 min"
    plan.timing_check = "Introduction 10 + Main 50 + Summary 15 + Home 5 = 80 min"

    queries = [
        "Confirm the master syllabus reference book/BR for this subject.",
        "Confirm the class existing knowledge and prerequisites.",
        "Confirm the session duration and sequence; 80 min and 01/01 were assumed.",
        "Prepared-by/approved-by names, ranks, P numbers and dates must be filled before approval.",
        "Confirm any training aids available beyond L & Ds, IB, PDF and PPT.",
    ]
    if "pilot static" in row.syllabus.lower():
        queries.insert(
            0,
            "The master syllabus says 'pilot static system'; confirm whether this is "
            "a typo for 'pitot-static system'.",
        )
    seen = set()
    merged = []
    for query in queries + plan.queries:
        if query and query not in seen:
            seen.add(query)
            merged.append(query)
    plan.queries = merged
    return sanitize_plan_prose(plan)


async def _generate_one(
    subject_name: str,
    subject_code: str,
    rows: list[SyllabusRow],
    index: int,
    class_name: str,
    session_minutes: int,
) -> tuple[LessonPlan, Path]:
    row = rows[index]
    next_row = rows[index + 1] if index + 1 < len(rows) else None
    inputs = _inputs_for_row(
        subject_name,
        subject_code,
        row,
        next_row,
        index,
        len(rows),
        class_name,
        session_minutes,
    )
    plan = await generate_plan_from_inputs(inputs)
    plan = _clean_plan(
        plan,
        row,
        next_row,
        index,
        len(rows),
        class_name,
        session_minutes,
        subject_name,
        subject_code,
    )

    slug = re.sub(r"[^a-z0-9]+", "_", subject_name.lower()).strip("_") or "subject"
    out_dir = Path(__file__).resolve().parents[1] / "experiments" / "output" / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"lp{index + 1:02d}"
    pdf_path = out_dir / f"{stem}.pdf"
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(plan.model_dump_json(indent=2))
    _render_pdf(plan, pdf_path)
    return plan, pdf_path


async def main_async(args: argparse.Namespace) -> int:
    doc = DocxDocument(args.file)
    rows, title = extract_subject_rows(doc, args.subject_code, args.subject_name)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("No syllabus rows found.")
        return 1

    print(f"subject: {title}")
    print(f"rows: {len(rows)}")
    for index, row in enumerate(rows):
        print(f"  {index + 1:02d}. {row.objective} | {row.syllabus[:100]}")

    outputs = []
    for index in range(len(rows)):
        print(f"generating LP {index + 1:02d}/{len(rows)}...")
        plan, pdf_path = await _generate_one(
            args.subject_name,
            args.subject_code,
            rows,
            index,
            args.class_name,
            args.session_minutes,
        )
        outputs.append(pdf_path)
        print(f"  topic: {plan.topic}")
        print(f"  pdf: {pdf_path}")

    print("\nGenerated lesson plans:")
    for path in outputs:
        print(f"  {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True, help="Master syllabus DOCX")
    parser.add_argument("--subject-code", required=True, help="Subject code, e.g. 20")
    parser.add_argument("--subject-name", required=True, help="Subject name, e.g. Basic Electrical")
    parser.add_argument(
        "--class-name",
        default="Air Agniveer Apprentice (Engineering)",
    )
    parser.add_argument("--session-minutes", type=int, default=80)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
