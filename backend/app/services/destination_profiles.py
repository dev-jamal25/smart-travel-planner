"""Destination profiles for classifier lookups."""
from __future__ import annotations

from app.schemas.classifier import ClassifierPredictionRequest  # noqa: F401

# Mapping of destination names to classifier input profiles
# These are the 10 destinations supported by RAG and classifier
_DESTINATION_PROFILES: dict[str, ClassifierPredictionRequest] = {
    "interlaken": ClassifierPredictionRequest(
        country="Switzerland",
        continent="Europe",
        destination_type="Mountain/Alpine",
        avg_cost_usd_per_day=150.0,
        best_season="Summer (Jun-Sep)",
        avg_rating=4.8,
        annual_visitors_m=3.0,
        unesco_site=False,
    ),
    "banff": ClassifierPredictionRequest(
        country="Canada",
        continent="North America",
        destination_type="Mountain/National Park",
        avg_cost_usd_per_day=120.0,
        best_season="Summer (Jun-Sep)",
        avg_rating=4.7,
        annual_visitors_m=4.0,
        unesco_site=True,
    ),
    "bali": ClassifierPredictionRequest(
        country="Indonesia",
        continent="Asia",
        destination_type="Beach/Island",
        avg_cost_usd_per_day=40.0,
        best_season="Dry (Apr-Oct)",
        avg_rating=4.5,
        annual_visitors_m=6.0,
        unesco_site=True,
    ),
    "santorini": ClassifierPredictionRequest(
        country="Greece",
        continent="Europe",
        destination_type="Beach/Island",
        avg_cost_usd_per_day=120.0,
        best_season="Spring/Fall (Apr-May, Sep-Oct)",
        avg_rating=4.6,
        annual_visitors_m=2.0,
        unesco_site=False,
    ),
    "kyoto": ClassifierPredictionRequest(
        country="Japan",
        continent="Asia",
        destination_type="Cultural/Historic",
        avg_cost_usd_per_day=100.0,
        best_season="Spring (Mar-May) / Fall (Sep-Nov)",
        avg_rating=4.7,
        annual_visitors_m=2.5,
        unesco_site=True,
    ),
    "istanbul": ClassifierPredictionRequest(
        country="Turkey",
        continent="Europe/Asia",
        destination_type="Cultural/Historic",
        avg_cost_usd_per_day=60.0,
        best_season="Spring (Apr-May) / Fall (Sep-Oct)",
        avg_rating=4.4,
        annual_visitors_m=5.0,
        unesco_site=True,
    ),
    "tbilisi": ClassifierPredictionRequest(
        country="Georgia",
        continent="Europe/Asia",
        destination_type="Cultural/Urban",
        avg_cost_usd_per_day=40.0,
        best_season="Spring/Fall (Apr-May, Sep-Oct)",
        avg_rating=4.5,
        annual_visitors_m=1.5,
        unesco_site=False,
    ),
    "kraków": ClassifierPredictionRequest(
        country="Poland",
        continent="Europe",
        destination_type="Cultural/Historic",
        avg_cost_usd_per_day=50.0,
        best_season="Spring/Fall (Apr-May, Sep-Oct)",
        avg_rating=4.6,
        annual_visitors_m=2.0,
        unesco_site=True,
    ),
    "dubai": ClassifierPredictionRequest(
        country="United Arab Emirates",
        continent="Middle East",
        destination_type="Urban/Luxury",
        avg_cost_usd_per_day=200.0,
        best_season="Winter (Nov-Mar)",
        avg_rating=4.4,
        annual_visitors_m=14.0,
        unesco_site=False,
    ),
    "singapore": ClassifierPredictionRequest(
        country="Singapore",
        continent="Asia",
        destination_type="Urban/Modern",
        avg_cost_usd_per_day=100.0,
        best_season="Year-round (driest: Feb-Apr)",
        avg_rating=4.5,
        annual_visitors_m=19.0,
        unesco_site=False,
    ),
}


def get_destination_profile(destination: str) -> ClassifierPredictionRequest | None:
    """Retrieve classifier profile for a destination.

    Matching is case-insensitive and strips whitespace.
    Also accepts "Krakow" as alias for "Kraków".
    """
    normalized = destination.strip().lower()
    # Special case: accept "krakow" as alias for "kraków"
    if normalized == "krakow":
        normalized = "kraków"
    return _DESTINATION_PROFILES.get(normalized)


def list_supported_profile_destinations() -> list[str]:
    """Return list of supported destinations with their canonical names."""
    # Return the canonical names (keys), not normalized versions
    return sorted(_DESTINATION_PROFILES.keys())
