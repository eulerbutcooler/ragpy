from pydantic import BaseModel


class LessonPlanGenerateRequest(BaseModel):
    bucket: str
    key: str
    file_name: str
    subject_code: str
    subject_name: str
    class_name: str = "Air Agniveer Apprentice (Engineering)"
    session_minutes: int = 80


class LessonPlanPDF(BaseModel):
    row_index: int
    lesson_plan_no: str
    title: str
    pdf_base64: str


class LessonPlanGenerateResponse(BaseModel):
    subject_code: str
    subject_name: str
    plans: list[LessonPlanPDF]
