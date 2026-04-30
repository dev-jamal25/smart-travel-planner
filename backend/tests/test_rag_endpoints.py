"""Tests for RAG schemas."""

from app.schemas.rag import (
    RagChunkResult,
    RagEvalCaseResponse,
    RagEvalResponse,
    RagSearchRequest,
    RagSearchResponse,
)


def test_rag_search_request_schema():
    """Test RagSearchRequest schema."""
    req = RagSearchRequest(query="test", top_k=5, destination_filter=None)
    assert req.query == "test"
    assert req.top_k == 5


def test_rag_search_request_default_top_k():
    """Test RagSearchRequest defaults top_k to 5."""
    req = RagSearchRequest(query="test")
    assert req.top_k == 5


def test_rag_search_request_destination_filter_none():
    """Test destination_filter normalizes None to None."""
    req = RagSearchRequest(query="test", destination_filter=None)
    assert req.destination_filter is None


def test_rag_search_request_destination_filter_empty_string():
    """Test destination_filter normalizes empty string to None."""
    req = RagSearchRequest(query="test", destination_filter="")
    assert req.destination_filter is None


def test_rag_search_request_destination_filter_whitespace():
    """Test destination_filter normalizes whitespace-only to None."""
    req = RagSearchRequest(query="test", destination_filter="   ")
    assert req.destination_filter is None


def test_rag_search_request_destination_filter_none_string():
    """Test destination_filter normalizes 'none' (case-insensitive) to None."""
    for none_val in ["none", "None", "NONE", "nOnE"]:
        req = RagSearchRequest(query="test", destination_filter=none_val)
        assert req.destination_filter is None


def test_rag_search_request_destination_filter_null_string():
    """Test destination_filter normalizes 'null' (case-insensitive) to None."""
    for null_val in ["null", "Null", "NULL", "nUlL"]:
        req = RagSearchRequest(query="test", destination_filter=null_val)
        assert req.destination_filter is None


def test_rag_search_request_destination_filter_all_string():
    """Test destination_filter normalizes 'all' (case-insensitive) to None."""
    for all_val in ["all", "All", "ALL", "aLl"]:
        req = RagSearchRequest(query="test", destination_filter=all_val)
        assert req.destination_filter is None


def test_rag_search_request_destination_filter_strips_whitespace():
    """Test destination_filter strips whitespace from real destination names."""
    req = RagSearchRequest(query="test", destination_filter="  Tokyo  ")
    assert req.destination_filter == "Tokyo"


def test_rag_search_request_destination_filter_preserves_name():
    """Test destination_filter preserves real destination names."""
    req = RagSearchRequest(query="test", destination_filter="Interlaken")
    assert req.destination_filter == "Interlaken"


def test_rag_chunk_result_schema():
    """Test RagChunkResult schema."""
    chunk = RagChunkResult(
        destination_name="Interlaken",
        source_title="Wikivoyage — Interlaken",
        source_url="https://en.wikivoyage.org/wiki/Interlaken",
        distance=0.15,
        chunk_preview="Sample text",
        chunk_index=0,
    )
    assert chunk.destination_name == "Interlaken"
    assert chunk.distance == 0.15


def test_rag_search_response_schema():
    """Test RagSearchResponse schema."""
    chunk = RagChunkResult(
        destination_name="Interlaken",
        source_title="Wikivoyage — Interlaken",
        source_url="https://en.wikivoyage.org/wiki/Interlaken",
        distance=0.15,
        chunk_preview="Sample",
        chunk_index=0,
    )
    response = RagSearchResponse(query="test", top_k=5, results=[chunk])
    assert response.query == "test"
    assert len(response.results) == 1


def test_rag_eval_case_response_schema():
    """Test RagEvalCaseResponse schema."""
    chunk = RagChunkResult(
        destination_name="Interlaken",
        source_title="Wikivoyage — Interlaken",
        source_url="https://en.wikivoyage.org/wiki/Interlaken",
        distance=0.15,
        chunk_preview="Sample",
        chunk_index=0,
    )
    search = RagSearchResponse(query="test", top_k=5, results=[chunk])
    case = RagEvalCaseResponse(
        name="adventure",
        query="test",
        expected_destinations=["Interlaken"],
        top_3_pass=True,
        top_5_pass=True,
        search=search,
    )
    assert case.name == "adventure"
    assert case.top_3_pass is True


def test_rag_eval_response_schema():
    """Test RagEvalResponse schema."""
    chunk = RagChunkResult(
        destination_name="Interlaken",
        source_title="Wikivoyage — Interlaken",
        source_url="https://en.wikivoyage.org/wiki/Interlaken",
        distance=0.15,
        chunk_preview="Sample",
        chunk_index=0,
    )
    search = RagSearchResponse(query="test", top_k=5, results=[chunk])
    case = RagEvalCaseResponse(
        name="adventure",
        query="test",
        expected_destinations=["Interlaken"],
        top_3_pass=True,
        top_5_pass=True,
        search=search,
    )
    response = RagEvalResponse(
        total_cases=7,
        top_3_passes=5,
        top_5_passes=6,
        cases=[case],
    )
    assert response.total_cases == 7
    assert response.top_3_passes == 5
    assert len(response.cases) == 1


def test_load_rag_eval_cases():
    """Test loading evaluation cases from JSON file."""
    from app.evaluation.rag_eval_cases import load_rag_eval_cases

    cases = load_rag_eval_cases()

    # Should load exactly 7 cases
    assert len(cases) == 7

    # Each case should have required fields
    for case in cases:
        assert hasattr(case, "name")
        assert hasattr(case, "query")
        assert hasattr(case, "expected_destinations")
        assert isinstance(case.name, str)
        assert isinstance(case.query, str)
        assert isinstance(case.expected_destinations, list)

    # Check specific case names exist
    case_names = [c.name for c in cases]
    assert "adventure_mountains" in case_names
    assert "culture_history" in case_names
    assert "beaches_relaxation" in case_names
    assert "luxury_modern_city" in case_names
    assert "budget_history" in case_names
    assert "family_practical_city" in case_names
    assert "logistics_safety" in case_names


def test_load_rag_eval_cases_caching():
    """Test that eval cases are cached (same object on repeated calls)."""
    from app.evaluation.rag_eval_cases import load_rag_eval_cases

    cases1 = load_rag_eval_cases()
    cases2 = load_rag_eval_cases()

    # Should be the exact same object due to lru_cache
    assert cases1 is cases2


def test_rag_eval_case_model_validation():
    """Test EvalCase Pydantic model validation."""
    from pydantic import ValidationError

    from app.evaluation.rag_eval_cases import EvalCase

    # Valid case
    case = EvalCase(
        name="test",
        query="test query",
        expected_destinations=["Dest1", "Dest2"],
    )
    assert case.name == "test"

    # Invalid case missing required field should raise ValidationError
    try:
        EvalCase(name="test", query="test query")
        raise AssertionError("Should raise ValidationError")
    except ValidationError:
        pass  # Expected
