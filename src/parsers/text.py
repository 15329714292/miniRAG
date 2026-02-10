import os

from .base import BaseParser, Document
from ..utils.io import read_text_file
from ..utils.text import normalize_text


class TextParser(BaseParser):
    def parse(self, file_path: str):
        # Read and normalize the raw text file.
        text = normalize_text(read_text_file(file_path))
        file_name = os.path.basename(file_path)
        if not text:
            return []
        # Wrap the whole file as a single document.
        doc_id = f"{file_name}::p1"
        return [Document(doc_id=doc_id, file_path=file_path, file_name=file_name, text=text)]
