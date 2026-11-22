"""Figure extraction from PDFs and PPTX decks.

Figures are stored in object storage and indexed as caption points in Qdrant
so chat, slides and quizzes can reference them. Extraction is intentionally
conservative: tiny icons, logos and full-page scans are skipped.
"""

from __future__ import annotations

import hashlib
import io
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

MIN_WIDTH = 160
MIN_HEIGHT = 160
MIN_BYTES = 1024
# A single image covering almost the whole page is usually a scan, not a figure.
MAX_PAGE_COVERAGE = 0.85


@dataclass
class ExtractedFigure:
    page_no: int | None
    width: int
    height: int
    sha256: str
    ext: str
    data: bytes
    bbox: dict[str, float] | None

    @property
    def content_type(self) -> str:
        if self.ext in {"jpg", "jpeg"}:
            return "image/jpeg"
        if self.ext == "webp":
            return "image/webp"
        if self.ext == "gif":
            return "image/gif"
        return "image/png"


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _keep(width: int, height: int, size: int) -> bool:
    return width >= MIN_WIDTH and height >= MIN_HEIGHT and size >= MIN_BYTES


def _extract_pdf(path: Path, limit: int = 0) -> list[ExtractedFigure]:
    try:
        import pymupdf as fitz  # type: ignore
    except ImportError:  # pragma: no cover - older PyMuPDF
        import fitz  # type: ignore

    figures: list[ExtractedFigure] = []
    seen: set[str] = set()

    with fitz.open(path) as doc:
        for page_index, page in enumerate(doc):
            page_area = abs(page.rect.width * page.rect.height) or 1.0
            for item in page.get_images(full=True):
                xref = item[0]
                width, height = item[2], item[3]
                if not _keep(width, height, MIN_BYTES):
                    continue

                rects = page.get_image_rects(xref)
                bbox = None
                if rects:
                    rect = rects[0]
                    coverage = abs(rect.width * rect.height) / page_area
                    if coverage >= MAX_PAGE_COVERAGE:
                        continue
                    bbox = {
                        "x0": round(rect.x0, 2),
                        "y0": round(rect.y0, 2),
                        "x1": round(rect.x1, 2),
                        "y1": round(rect.y1, 2),
                    }

                try:
                    raw = doc.extract_image(xref)
                except Exception:  # noqa: BLE001 - corrupt image stream
                    continue
                data = raw.get("image") or b""
                if len(data) < MIN_BYTES:
                    continue
                sha = _digest(data)
                if sha in seen:
                    continue
                seen.add(sha)
                figures.append(
                    ExtractedFigure(
                        page_no=page_index + 1,
                        width=int(raw.get("width") or width),
                        height=int(raw.get("height") or height),
                        sha256=sha,
                        ext=str(raw.get("ext") or "png").lower(),
                        data=data,
                        bbox=bbox,
                    )
                )
                if limit and len(figures) >= limit:
                    return figures
    return figures


def _extract_pptx(path: Path, limit: int = 0) -> list[ExtractedFigure]:
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    figures: list[ExtractedFigure] = []
    seen: set[str] = set()
    presentation = Presentation(str(path))

    for slide_index, slide in enumerate(presentation.slides):
        for shape in slide.shapes:
            if shape.shape_type != MSO_SHAPE_TYPE.PICTURE:
                continue
            try:
                image = shape.image
            except Exception:  # noqa: BLE001 - linked/OLE image
                continue
            data = image.blob
            width = int(getattr(image, "px_width", 0) or 0)
            height = int(getattr(image, "px_height", 0) or 0)
            if not _keep(width, height, len(data)):
                continue
            sha = _digest(data)
            if sha in seen:
                continue
            seen.add(sha)
            figures.append(
                ExtractedFigure(
                    page_no=slide_index + 1,
                    width=width,
                    height=height,
                    sha256=sha,
                    ext=str(image.ext or "png").lower(),
                    data=data,
                    bbox=None,
                )
            )
            if limit and len(figures) >= limit:
                return figures
    return figures


def extract_figures(path: Path, file_name: str, limit: int = 0) -> list[ExtractedFigure]:
    """Extract candidate figures, deduplicated within the file."""
    suffix = Path(file_name).suffix.lower()
    try:
        if suffix == ".pdf":
            figures = _extract_pdf(path, limit)
        elif suffix in {".pptx", ".ppsx"}:
            figures = _extract_pptx(path, limit)
        else:
            figures = []
    except Exception:  # noqa: BLE001 - figure extraction must never fail ingest
        logger.exception("figure_extract_failed file=%s", file_name)
        return []

    logger.info("figure_extract_done file=%s figures=%d", file_name, len(figures))
    return figures


def figure_object_key(course_id: str, file_id: str, figure: ExtractedFigure) -> str:
    return f"courses/{course_id}/figures/{file_id}/{figure.sha256[:24]}.{figure.ext}"


def image_bytes_to_data_uri(data: bytes, content_type: str) -> str:
    import base64

    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


def downscale_for_caption(data: bytes, max_edge: int = 1024) -> tuple[bytes, str]:
    """Bound vision token cost by re-encoding large images to JPEG."""
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow is a transitive dep
        return data, "image/png"

    try:
        with Image.open(io.BytesIO(data)) as img:
            img = img.convert("RGB")
            longest = max(img.size)
            if longest > max_edge:
                scale = max_edge / float(longest)
                img = img.resize(
                    (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
                )
            out = io.BytesIO()
            img.save(out, format="JPEG", quality=85, optimize=True)
            return out.getvalue(), "image/jpeg"
    except Exception:  # noqa: BLE001 - fall back to the original bytes
        return data, "image/png"
