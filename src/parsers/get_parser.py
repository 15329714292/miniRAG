import os

from .docx import DocxParser
from .pdf import PdfParser
from .text import TextParser


_EXT_TO_PARSER = {
    ".txt": TextParser,
    ".md": TextParser,
    ".pdf": PdfParser,
    ".docx": DocxParser,
}


def get_parser(file_path: str):
    # Select a parser based on file extension.
    ext = os.path.splitext(file_path)[1].lower()
    parser_cls = _EXT_TO_PARSER.get(ext)
    if parser_cls is None:
        raise ValueError(f"Unsupported file extension: {ext}")
    return parser_cls()
