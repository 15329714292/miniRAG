import os

from pypdf import PdfReader

from .base import BaseParser, Document
from ..utils.text import normalize_text


class PdfParser(BaseParser):
    def parse(self, file_path: str):
        # Extract text per page to preserve page boundaries.
        reader = PdfReader(file_path)
        file_name = os.path.basename(file_path)
        documents = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = normalize_text(text)
            if not text:
                continue
            # Use page number in the document id for traceability.
            doc_id = f"{file_name}::p{i}"
            documents.append(
                Document(doc_id=doc_id, file_path=file_path, file_name=file_name, text=text)
            )
        return documents
