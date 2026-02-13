import os

from pypdf import PdfReader

from .base import BaseParser, Document
from ..utils.text import normalize_text


class PdfParser(BaseParser):
    def parse(self, file_path: str):
        # Extract text per page and join to a single document.
        reader = PdfReader(file_path)
        file_name = os.path.basename(file_path)
        page_texts = []
        for page in reader.pages:
            text = page.extract_text() or ""
            text = normalize_text(text)
            if not text:
                continue
            page_texts.append(text)

        if not page_texts:
            return None

        full_text = "\n".join(page_texts)
        return Document(file_path=file_path, file_name=file_name, text=full_text)
