from dataclasses import dataclass
from typing import List


@dataclass
class Document:
    file_path: str
    file_name: str
    text: str


class BaseParser:
    def parse(self, file_path: str) -> Document:
        # Subclasses should implement format-specific parsing.
        raise NotImplementedError
