
from llama_index.core import Document


def enrich_metadata(
    documents: list[Document],
    course_id: str,
    course_name: str,
    file_id: str,
    file_name: str,
    teacher_id: str,
    teacher_name: str,
) -> list[Document]:
    for doc in documents:
        doc.metadata["course_id"] = course_id
        doc.metadata["course_name"] = course_name
        doc.metadata["file_id"] = file_id
        doc.metadata["teacher_id"] = teacher_id
        doc.metadata["teacher_name"] = teacher_name
        doc.metadata["file_name"] = file_name

        excluded = set(getattr(doc, "excluded_embed_metadata_keys", []) or [])
        excluded.update({"course_id", "file_id", "teacher_id"})
        doc.excluded_embed_metadata_keys = list(excluded)

    return documents