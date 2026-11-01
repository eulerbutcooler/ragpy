"""Master-syllabus parsing and topic extraction.

The syllabus is usually a large DOCX with administration, assessment, and
reference sections. We deterministically select the teaching/curriculum
section first, then ask the LLM to extract the topic list from only that
section. This keeps unrelated administrative text out of the lesson plan.
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
from pathlib import Path

from docx import Document as DocxDocument
from docx.table import Table
from llama_index.core import Document
from llama_index.core.prompts.base import PromptTemplate
from pydantic import BaseModel

from app.config.settings import settings
from app.llm.factory import get_llm
from app.llm.structured import astructured_predict_json
from app.pipeline.ingest.loader import load_file
from app.pipeline.ingest.parser import _sanitize_text, _table_to_markdown
from app.pipeline.quiz.extractor import TopicExtraction

logger = logging.getLogger(__name__)


SYLLABUS_TOPIC_PROMPT = PromptTemplate(
    template=(
        "You are a naval aviation curriculum specialist analyzing a master syllabus.\n"
        "Extract the distinct teaching topics that should become lessons or slide decks.\n"
        "RULES:\n"
        "1. Strictly ignore administrative metadata: instructor names, ranks, dates, "
        "classroom numbers, assessment weights, and timing durations.\n"
        "2. Extract all distinct educational topic phrases.\n"
        "3. Each topic phrase must be rich, specific, and descriptive.\n"
        '4. Return a valid JSON object matching: {{"topics": ["topic 1", "topic 2", ...]}}\n\n'
        "Master Syllabus Section:\n"
        "{context_str}\n\n"
        "JSON:"
    )
)


class SyllabusRow(BaseModel):
    task: str = ""
    sub_task: str = ""
    objective: str = ""
    syllabus: str = ""
    periods_t: str = ""
    periods_p: str = ""
    periods_f: str = ""
    periods_total: str = ""
    sok: str = ""
    remarks: str = ""


class SubjectInfo(BaseModel):
    code: str
    name: str
    rows: list[SyllabusRow] = []
    topics: list[str] = []


def _clean_cell(value: str) -> str:
    return " ".join((value or "").replace("\xa0", " ").split())


def _table_header_map(table) -> dict[str, int]:
    rows = table.rows
    for row_index, row in enumerate(rows[:6]):
        cells = [_clean_cell(cell.text).lower() for cell in row.cells]
        if "syllabus" not in cells or "training objective" not in cells:
            continue

        mapping: dict[str, int] = {}
        for index, name in enumerate(cells):
            if name in {
                "task list",
                "sub-task list",
                "training objective",
                "syllabus",
                "sok",
                "remarks",
            }:
                mapping.setdefault(name, index)

        # The period columns are on the following T/P/F/T header row.
        for period_row in rows[row_index + 1 : row_index + 3]:
            values = [_clean_cell(cell.text).upper() for cell in period_row.cells]
            if "T" in values and "P" in values and "F" in values:
                t_indices = [i for i, value in enumerate(values) if value == "T"]
                mapping["periods_t"] = t_indices[0]
                mapping["periods_p"] = values.index("P")
                mapping["periods_f"] = values.index("F")
                mapping["periods_total"] = t_indices[1] if len(t_indices) > 1 else t_indices[0]
                break
        return mapping
    return {}


def _parse_table_rows(table, mapping: dict[str, int]) -> list[SyllabusRow]:
    rows: list[SyllabusRow] = []
    for row in table.rows:
        cells = row.cells
        objective_index = mapping.get("training objective")
        syllabus_index = mapping.get("syllabus")
        if objective_index is None or syllabus_index is None:
            continue
        if objective_index >= len(cells) or syllabus_index >= len(cells):
            continue

        objective = _clean_cell(cells[objective_index].text)
        syllabus = _clean_cell(cells[syllabus_index].text)
        if not syllabus or not re.match(r"^\d+\.\d+", objective):
            continue

        def cell(name: str, row_cells=cells) -> str:
            index = mapping.get(name)
            if index is None or index >= len(row_cells):
                return ""
            return _clean_cell(row_cells[index].text)

        rows.append(
            SyllabusRow(
                task=cell("task list"),
                sub_task=cell("sub-task list"),
                objective=objective,
                syllabus=syllabus,
                periods_t=cell("periods_t"),
                periods_p=cell("periods_p"),
                periods_f=cell("periods_f"),
                periods_total=cell("periods_total"),
                sok=cell("sok"),
                remarks=cell("remarks"),
            )
        )
    return rows


def extract_subject_rows(
    doc,
    subject_code: str | None = None,
    subject_name: str | None = None,
) -> tuple[list[SyllabusRow], str]:
    """Find a subject's detailed-syllabus table and return its rows.

    Multi-subject master syllabi put each subject in its own table. The task
    numbers (e.g. 20.1.1.1) carry the subject code, so ``subject_code='20'``
    selects Basic Electrical without relying on page numbers.
    """
    wanted_code = (subject_code or "").strip()
    wanted_name = (subject_name or "").strip().lower()
    candidates: list[tuple[list[SyllabusRow], str]] = []

    for table in doc.tables:
        mapping = _table_header_map(table)
        if not mapping:
            continue
        rows = _parse_table_rows(table, mapping)
        if not rows:
            continue

        row_codes = {
            match.group(1) for row in rows if (match := re.match(r"^(\d+)\.", row.objective))
        }
        if wanted_code and wanted_code not in row_codes:
            continue

        if not wanted_code and wanted_name:
            haystack = " ".join(
                f"{row.task} {row.sub_task} {row.objective} {row.syllabus}" for row in rows
            ).lower()
            if wanted_name not in haystack:
                continue

        title = f"{subject_name or 'Subject'} ({wanted_code or next(iter(row_codes), '?')})"
        candidates.append((rows, title))

    if not candidates:
        raise ValueError(
            f"no detailed syllabus table found for subject_code={subject_code!r} "
            f"subject_name={subject_name!r}"
        )

    if len(candidates) > 1 and not wanted_code:
        # Prefer the candidate with the most rows; still deterministic.
        candidates.sort(key=lambda item: len(item[0]), reverse=True)
    return candidates[0]


def _summary_subject_names(doc) -> dict[str, str]:
    """Map subject codes to names from the summary table.

    The summary table's first data row is subject code 1, the second is code 2,
    and so on. Detailed-syllabus task numbers use the same serial (e.g. 20.1.1.1
    belongs to the 20th summary row, Basic Electrical).
    """
    ignored = {
        "paper",
        "time",
        "total marks",
        "subject",
        "duration (in periods)",
        "marks",
        "total",
    }
    for table in doc.tables:
        header = {_clean_cell(cell.text).lower() for cell in table.rows[0].cells}
        if "paper" not in header or "subject" not in header:
            continue

        names: dict[str, str] = {}
        for row_index, row in enumerate(table.rows[1:], start=1):
            values: list[str] = []
            for cell in row.cells:
                value = _clean_cell(cell.text)
                if value and value not in values:
                    values.append(value)

            candidates = [
                value
                for value in values
                if value.lower() not in ignored
                and not value.lower().startswith("paper ")
                and not re.match(r"^\d", value)
                and value.strip("-–— ") not in {"", "-", "--"}
            ]
            if candidates:
                names[str(row_index)] = candidates[-1]
        if names:
            return names
    return {}


def extract_subjects_from_doc(doc) -> list[SubjectInfo]:
    """Return every detailed-syllabus subject in a multi-subject DOCX."""
    names = _summary_subject_names(doc)
    rows_by_code: dict[str, list[SyllabusRow]] = {}

    for table in doc.tables:
        mapping = _table_header_map(table)
        if not mapping:
            continue
        for row in _parse_table_rows(table, mapping):
            match = re.match(r"^(\d+)\.", row.objective)
            if not match:
                continue
            rows_by_code.setdefault(match.group(1), []).append(row)

    subjects: list[SubjectInfo] = []
    for code in sorted(rows_by_code, key=lambda value: int(value)):
        rows = rows_by_code[code]
        topics: list[str] = []
        seen: set[str] = set()
        for row in rows:
            topic = row.syllabus.strip()
            key = topic.lower()
            if topic and key not in seen:
                seen.add(key)
                topics.append(topic)
        subjects.append(
            SubjectInfo(
                code=code,
                name=names.get(code, f"Subject {code}"),
                rows=rows,
                topics=topics,
            )
        )

    logger.info("syllabus_subjects_extracted count=%d", len(subjects))
    return subjects


def _pattern_list(raw: str) -> list[str]:
    return [part.strip().lower() for part in raw.split(",") if part.strip()]


def _heading_level(style_name: str) -> int | None:
    lowered = (style_name or "").strip().lower()
    match = re.match(r"heading\s+(\d+)", lowered)
    if match:
        return int(match.group(1))
    if lowered in {"title", "subtitle"}:
        return 1
    return None


def _iter_docx_blocks(doc):
    """Yield (text, heading_level) in document order, including tables."""
    paragraph_by_element = {p._element: p for p in doc.paragraphs}

    for element in doc.element.body:
        tag = element.tag.split("}")[-1]

        if tag == "p":
            para = paragraph_by_element.get(element)
            if not para:
                continue
            text = para.text.strip()
            if text:
                yield text, _heading_level(para.style.name if para.style else "")
        elif tag == "tbl":
            table = Table(element, doc)
            raw_rows = [
                [cell.text.strip().replace("\n", " ") for cell in row.cells] for row in table.rows
            ]
            text = _table_to_markdown(raw_rows)
            if text:
                yield text, None


def extract_syllabus_section(doc) -> tuple[str, str]:
    """Select the curriculum-relevant section of a syllabus DOCX."""
    start_patterns = _pattern_list(settings.syllabus_section_patterns)
    stop_patterns = _pattern_list(settings.syllabus_stop_patterns)
    blocks = list(_iter_docx_blocks(doc))

    if not blocks:
        return "", "empty document"

    start_index: int | None = None
    start_level: int | None = None
    for index, (text, level) in enumerate(blocks):
        lowered = text.lower()
        if any(pattern in lowered for pattern in start_patterns):
            start_index = index
            start_level = level
            break

    if start_index is None:
        logger.info("syllabus_section_not_found using_full_document")
        return "\n\n".join(text for text, _ in blocks), "full document"

    section_title = blocks[start_index][0]
    selected = [section_title]

    for text, level in blocks[start_index + 1 :]:
        lowered = text.lower()
        if any(pattern in lowered for pattern in stop_patterns):
            break
        if start_level is not None and level is not None and level <= start_level:
            break
        selected.append(text)

    logger.info(
        "syllabus_section_selected title=%r blocks=%d",
        section_title[:80],
        len(selected),
    )
    return "\n\n".join(selected), section_title


def parse_syllabus_docx(file_path: Path) -> tuple[Document, str]:
    doc = DocxDocument(str(file_path))
    text, section = extract_syllabus_section(doc)
    if not text.strip():
        raise ValueError("syllabus DOCX contains no readable text")
    return Document(text=_sanitize_text(text), metadata={"syllabus_section": section}), section


def _to_title_case(value: str) -> str:
    clean = value.strip().strip('"').strip("'")
    return " ".join(word.capitalize() for word in clean.split()) if clean else ""


async def extract_syllabus_rows(
    bucket: str,
    key: str,
    file_name: str,
    subject_code: str | None = None,
    subject_name: str | None = None,
) -> tuple[list[SyllabusRow], str]:
    """Load a master-syllabus DOCX and return one subject's detailed rows."""
    file_path = await load_file(bucket=bucket, key=key, filename=file_name)
    try:
        doc = await asyncio.to_thread(DocxDocument, str(file_path))
        rows, title = await asyncio.to_thread(
            extract_subject_rows,
            doc,
            subject_code,
            subject_name,
        )
    finally:
        shutil.rmtree(file_path.parent, ignore_errors=True)

    logger.info(
        "syllabus_subject_extracted subject=%r code=%r rows=%d",
        subject_name,
        subject_code,
        len(rows),
    )
    return rows, title


async def load_syllabus_subjects(
    bucket: str,
    key: str,
    file_name: str,
) -> list[SubjectInfo]:
    """Load a master-syllabus DOCX and return every detected subject."""
    file_path = await load_file(bucket=bucket, key=key, filename=file_name)
    try:
        doc = await asyncio.to_thread(DocxDocument, str(file_path))
        subjects = await asyncio.to_thread(extract_subjects_from_doc, doc)
    finally:
        shutil.rmtree(file_path.parent, ignore_errors=True)

    logger.info("syllabus_subjects_loaded count=%d", len(subjects))
    return subjects


async def extract_syllabus_topics(bucket: str, key: str, file_name: str) -> tuple[list[str], str]:
    file_path = await load_file(bucket=bucket, key=key, filename=file_name)
    try:
        document, section = await asyncio.to_thread(parse_syllabus_docx, file_path)
    finally:
        # load_file creates a temporary directory; remove it after parsing.
        shutil.rmtree(file_path.parent, ignore_errors=True)

    prompt = SYLLABUS_TOPIC_PROMPT
    result: TopicExtraction = await astructured_predict_json(
        get_llm(),
        TopicExtraction,
        prompt,
        context_str=document.text,
    )

    cleaned: list[str] = []
    seen: set[str] = set()
    for topic in result.topics:
        title = _to_title_case(topic)
        key_lower = title.lower()
        if title and key_lower not in seen:
            seen.add(key_lower)
            cleaned.append(title)

    logger.info("syllabus_topics_extracted section=%r count=%d", section, len(cleaned))
    return cleaned, section
