"""Load and validate RAG evaluation cases from JSON."""

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    """Single evaluation case for RAG retrieval testing."""

    name: str = Field(..., description="Unique name for the eval case")
    query: str = Field(..., description="User query to test retrieval against")
    expected_destinations: list[str] = Field(
        ..., description="Destinations expected in top-5 results"
    )


@lru_cache(maxsize=1)
def load_rag_eval_cases() -> list[EvalCase]:
    """Load and validate RAG evaluation cases from JSON file.

    Uses repo-relative path based on __file__, independent of current working directory.
    Results are cached for the lifetime of the process.

    Returns:
        List of EvalCase objects loaded from retrieval_eval_cases.json
    """
    # Compute path relative to this file
    eval_json_path = (
        Path(__file__).parent.parent.parent  # backend/
        / "rag_data"
        / "eval"
        / "retrieval_eval_cases.json"
    )

    with open(eval_json_path) as f:
        data = json.load(f)

    # Validate each case with Pydantic
    cases = [EvalCase(**case_dict) for case_dict in data]

    return cases
