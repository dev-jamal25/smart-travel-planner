import uuid

from pydantic import BaseModel


class ChunkResult(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    destination_name: str
    source_title: str
    chunk_text: str
    chunk_index: int
    cosine_distance: float
