from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DestinationCoordinates:
    name: str
    latitude: float
    longitude: float


# Fixed coordinates for the 10 supported RAG destinations.
# Geocoding is intentionally omitted: fixed coords are deterministic,
# require no external API call, and are sufficient for weather look-up.
_DESTINATIONS: dict[str, DestinationCoordinates] = {
    "interlaken": DestinationCoordinates("Interlaken", 46.6863, 7.8632),
    "banff": DestinationCoordinates("Banff", 51.1784, -115.5708),
    "bali": DestinationCoordinates("Bali", -8.3405, 115.0920),
    "santorini": DestinationCoordinates("Santorini", 36.3932, 25.4615),
    "kyoto": DestinationCoordinates("Kyoto", 35.0116, 135.7681),
    "istanbul": DestinationCoordinates("Istanbul", 41.0082, 28.9784),
    "tbilisi": DestinationCoordinates("Tbilisi", 41.6938, 44.8015),
    "krakow": DestinationCoordinates("Kraków", 50.0647, 19.9450),
    "kraków": DestinationCoordinates("Kraków", 50.0647, 19.9450),
    "dubai": DestinationCoordinates("Dubai", 25.2048, 55.2708),
    "singapore": DestinationCoordinates("Singapore", 1.3521, 103.8198),
}


def get_destination_coordinates(destination: str) -> DestinationCoordinates | None:
    """Return coordinates for a supported destination, case-insensitively."""
    return _DESTINATIONS.get(destination.strip().lower())


def list_supported_destinations() -> list[str]:
    """Return sorted unique display names for all supported destinations."""
    return sorted({v.name for v in _DESTINATIONS.values()})
