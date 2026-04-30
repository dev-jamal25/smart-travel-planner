"""Tests for destination profiles."""
from __future__ import annotations

from app.services.destination_profiles import (
    get_destination_profile,
    list_supported_profile_destinations,
)
from app.tools.registry import get_allowed_tools, is_allowed_tool


class TestDestinationProfiles:
    """Test destination profile lookups."""

    def test_get_profile_case_insensitive(self) -> None:
        profile = get_destination_profile("bali")
        assert profile is not None
        assert profile.country == "Indonesia"

        profile = get_destination_profile("BALI")
        assert profile is not None

        profile = get_destination_profile("Bali")
        assert profile is not None

    def test_get_profile_strips_whitespace(self) -> None:
        profile = get_destination_profile("  Interlaken  ")
        assert profile is not None
        assert profile.country == "Switzerland"

    def test_get_profile_kraków_and_krakow(self) -> None:
        profile_krakow = get_destination_profile("Krakow")
        assert profile_krakow is not None
        assert profile_krakow.country == "Poland"

        profile_kraków = get_destination_profile("Kraków")
        assert profile_kraków is not None
        assert profile_kraków == profile_krakow

    def test_get_profile_unsupported_destination(self) -> None:
        profile = get_destination_profile("NonExistentPlace")
        assert profile is None

    def test_list_supported_destinations(self) -> None:
        dests = list_supported_profile_destinations()
        assert len(dests) == 10
        assert "bali" in dests
        assert "interlaken" in dests
        assert "kraków" in dests

    def test_all_profiles_have_required_fields(self) -> None:
        dests = list_supported_profile_destinations()
        for dest_name in dests:
            profile = get_destination_profile(dest_name)
            assert profile is not None
            assert profile.country
            assert profile.continent
            assert profile.destination_type
            assert profile.avg_cost_usd_per_day >= 0
            assert profile.best_season
            assert 0 <= profile.avg_rating <= 5
            assert profile.annual_visitors_m >= 0
            assert isinstance(profile.unesco_site, bool)


class TestToolRegistry:
    """Test tool registry and allowlist."""

    def test_is_allowed_tool_rag(self) -> None:
        assert is_allowed_tool("destination_knowledge_retrieval")

    def test_is_allowed_tool_classifier(self) -> None:
        assert is_allowed_tool("classify_destination_style")

    def test_is_allowed_tool_weather(self) -> None:
        assert is_allowed_tool("fetch_live_weather")

    def test_rejects_disallowed_tool(self) -> None:
        assert not is_allowed_tool("some_other_tool")
        assert not is_allowed_tool("")
        assert not is_allowed_tool("http_call")

    def test_get_allowed_tools_count(self) -> None:
        tools = get_allowed_tools()
        assert len(tools) == 3

    def test_get_allowed_tools_contains_expected(self) -> None:
        tools = get_allowed_tools()
        assert "destination_knowledge_retrieval" in tools
        assert "classify_destination_style" in tools
        assert "fetch_live_weather" in tools

    def test_get_allowed_tools_returns_copy(self) -> None:
        tools1 = get_allowed_tools()
        tools2 = get_allowed_tools()
        assert tools1 == tools2
        # Modifying one should not affect the other
        tools1.add("extra")
        assert len(get_allowed_tools()) == 3
