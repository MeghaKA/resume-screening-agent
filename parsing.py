"""
Small helper module: reads a resume file and returns plain text,
regardless of whether it's .txt, .pdf, or .docx.
Keeping this separate from main.py keeps the agent logic easy to read.
"""

from pathlib import Path
from pypdf import PdfReader
from docx import Document


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if suffix == ".docx":
        doc = Document(str(path))
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)

    raise ValueError(f"Unsupported file type: {suffix}")
