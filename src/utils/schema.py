from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: int
    file_name: str
    text: str

    def to_dict(self) -> dict:
        # Serialize chunk metadata for storage.
        return {
            "chunk_id": self.chunk_id,
            "file_name": self.file_name,
            "text": self.text,
        }


@dataclass
class Sentence:
    sent_id: int
    chunk_id: int
    file_name: str
    text: str

    def to_dict(self) -> dict:
        # Serialize sentence metadata for storage.
        return {
            "sent_id": self.sent_id,
            "chunk_id": self.chunk_id,
            "file_name": self.file_name,
            "text": self.text,
        }


@dataclass
class Candidate:
    key: str
    file_name: str
    text: str
    score: float
    source: str
