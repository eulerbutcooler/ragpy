from pydantic import BaseModel

from app.pipeline.syllabus.extractor import SubjectInfo, SyllabusRow


class SyllabusSubjectsRequest(BaseModel):
    bucket: str
    key: str
    file_name: str


class SyllabusSubjectsResponse(BaseModel):
    subjects: list[SubjectInfo]


class SyllabusExtractRequest(BaseModel):
    bucket: str
    key: str
    file_name: str
    subject_code: str | None = None
    subject_name: str | None = None


class SyllabusExtractResponse(BaseModel):
    topics: list[str]
    rows: list[SyllabusRow] = []
    subject_code: str | None = None
    subject_name: str | None = None
    section: str | None = None
