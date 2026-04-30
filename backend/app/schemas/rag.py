import uuid

from pydantic import BaseModel, Field, field_validator


class ChunkResult(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    destination_name: str
    source_title: str
    chunk_text: str
    chunk_index: int
    cosine_distance: float


class RagSearchRequest(BaseModel):
    """Request for ad-hoc RAG search."""

    query: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    destination_filter: str | None = Field(
        default=None, description="Optional destination name filter"
    )

    @field_validator("destination_filter", mode="before")
    @classmethod
    def normalize_destination_filter(cls, v: str | None) -> str | None:
        """Normalize destination_filter to None for common no-filter values.

        Converts the following to None (case-insensitive):
        - None
        - Empty string ""
        - Whitespace-only strings
        - "none"
        - "null"
        - "all"

        Real destination names are stripped of whitespace and returned.
        """
        if v is None:
            return None

        # Convert to string if not already
        if not isinstance(v, str):
            return v

        # Strip whitespace
        v_stripped = v.strip()

        # Check for no-filter indicators (case-insensitive)
        if v_stripped.lower() in ("", "none", "null", "all"):
            return None

        # Return stripped destination name
        return v_stripped if v_stripped else None


class RagChunkResult(BaseModel):
    """Simplified chunk result for API responses."""

    destination_name: str
    source_title: str
    source_url: str
    distance: float | None = None
    chunk_preview: str = Field(..., description="First 200 chars of chunk text")
    chunk_index: int | None = None


class RagSearchResponse(BaseModel):
    """Response from RAG search endpoints."""

    query: str
    top_k: int
    results: list[RagChunkResult]


class RagEvalCaseResponse(BaseModel):
    """Single evaluation case result."""

    name: str
    query: str
    expected_destinations: list[str]
    top_3_pass: bool
    top_5_pass: bool
    search: RagSearchResponse


class RagEvalResponse(BaseModel):
    """Response from RAG evaluation endpoint."""

    total_cases: int
    top_3_passes: int
    top_5_passes: int
    cases: list[RagEvalCaseResponse]