"""Fetch destination content from Wikivoyage and save as raw .txt files for RAG ingestion.

Usage (from backend/ directory):
    uv run python scripts/fetch_wikivoyage_raw.py

Output files will be created in backend/rag_data/raw/ with headers (destination, source_title,
source_url) followed by cleaned plain text body content.

Only files with real source content from Wikivoyage are written. If both API and HTML
page fetch fail, that destination is skipped with a warning.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import httpx
import structlog
from bs4 import BeautifulSoup

logger = structlog.get_logger(__name__)

# Wikivoyage destination list
DESTINATIONS = [
    "Interlaken",
    "Banff",
    "Bali",
    "Santorini",
    "Kyoto",
    "Istanbul",
    "Tbilisi",
    "Kraków",
    "Dubai",
    "Singapore",
]

WIKIVOYAGE_API_URL = "https://en.wikivoyage.org/w/api.php"
WIKIVOYAGE_BASE_URL = "https://en.wikivoyage.org/wiki"
RAW_DIR = Path(__file__).resolve().parent.parent / "rag_data" / "raw"
HTTP_TIMEOUT = 15
REQUEST_DELAY = 0.5  # seconds between requests



def _clean_text(html_text: str) -> str:
    """Convert HTML to clean plain text, removing noise and repeated whitespace.
    
    If the input is already plain text (not HTML), minimal processing is applied.
    """
    # Check if input looks like HTML
    if "<" in html_text and ">" in html_text:
        soup = BeautifulSoup(html_text, "html.parser")

        # Remove script, style, and comment tags
        for tag in soup(["script", "style"]):
            tag.decompose()

        # Remove citation/reference markers like [1], [edit], etc.
        text = soup.get_text()
    else:
        # Already plain text, use as-is
        text = html_text
    
    text = re.sub(r"\[\d+\]", "", text)  # [1], [2], etc.
    text = re.sub(r"\[edit\]", "", text)  # [edit] links
    text = re.sub(r"Listen to this article", "", text)
    text = re.sub(r"\[hide\]", "", text)

    # Remove repeated whitespace and newlines
    text = re.sub(r"\n\s*\n", "\n", text)  # collapse multiple blank lines
    text = re.sub(r"[ \t]+", " ", text)  # collapse multiple spaces/tabs
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = "\n".join(lines)

    return text.strip()


def _split_document(text: str) -> tuple[str, str]:
    """
    Split a long document into two logical chunks:
    overview (first ~50% of content) and details (remaining content).

    This is a simple heuristic split by approximate length rather than
    attempting fragile section extraction.
    """
    words = text.split()
    if len(words) < 100:
        # If document is short, don't split
        return text, ""
    
    midpoint = len(words) // 2

    # Find a good split point near the midpoint (end of a sentence)
    cumulative = 0
    split_idx = 0
    for i, word in enumerate(words):
        cumulative += len(word) + 1  # +1 for space
        if cumulative > midpoint:
            split_idx = i
            break

    if split_idx == 0 or split_idx >= len(words) - 1:
        # Fallback: split at midpoint
        split_idx = len(words) // 2

    overview = " ".join(words[:split_idx]).strip()
    details = " ".join(words[split_idx:]).strip()

    return overview, details


async def _fetch_from_api(
    client: httpx.AsyncClient, destination: str
) -> str | None:
    """Fetch parsed HTML content from Wikivoyage MediaWiki API."""
    try:
        params = {
            "action": "parse",
            "format": "json",
            "page": destination,
            "prop": "text",
            "redirects": "1",
            "disableeditsection": "1",
        }
        response = await client.get(
            WIKIVOYAGE_API_URL,
            params=params,
            timeout=HTTP_TIMEOUT,
        )
        
        # Log non-success status codes
        if response.status_code != 200:
            logger.warning(
                "wikivoyage.api.http_error",
                destination=destination,
                status=response.status_code,
            )
            return None
        
        data = response.json()

        if "error" in data:
            logger.warning(
                "wikivoyage.api.error_response",
                destination=destination,
                error=data.get("error", {}).get("info", "unknown"),
            )
            return None

        if "parse" not in data or "text" not in data["parse"]:
            logger.warning(
                "wikivoyage.api.no_content",
                destination=destination,
            )
            return None

        # The text response is a dict with key "*" containing the HTML
        text_obj = data["parse"]["text"]
        if not isinstance(text_obj, dict) or "*" not in text_obj:
            logger.warning(
                "wikivoyage.api.malformed_text",
                destination=destination,
                text_type=type(text_obj).__name__,
            )
            return None
        
        html_text = text_obj["*"]
        if not html_text:
            return None
        
        return html_text
    except json.JSONDecodeError as exc:
        logger.warning(
            "wikivoyage.api.json_error",
            destination=destination,
            error=str(exc),
        )
        return None
    except httpx.TimeoutException:
        logger.warning(
            "wikivoyage.api.timeout",
            destination=destination,
        )
        return None
    except httpx.HTTPError as exc:
        logger.warning(
            "wikivoyage.api.http_exception",
            destination=destination,
            error=str(exc),
        )
        return None
    except Exception as exc:
        logger.exception(
            "wikivoyage.api.unknown_error",
            destination=destination,
            error=str(exc),
        )
        return None


async def _fetch_from_html(
    client: httpx.AsyncClient, destination: str
) -> str | None:
    """Fetch and parse the actual Wikivoyage HTML page as fallback."""
    try:
        # Wikivoyage page titles use underscores for spaces
        page_title = destination.replace(" ", "_")
        url = f"{WIKIVOYAGE_BASE_URL}/{page_title}"
        
        response = await client.get(
            url,
            timeout=HTTP_TIMEOUT,
        )
        
        if response.status_code != 200:
            logger.warning(
                "wikivoyage.html.http_error",
                destination=destination,
                status=response.status_code,
                url=url,
            )
            return None
        
        # Parse the HTML page
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Find the main content area
        content = soup.find("div", {"id": "mw-content-text"})
        if not content:
            logger.warning(
                "wikivoyage.html.no_content",
                destination=destination,
            )
            return None
        
        # Extract text from the content div
        html_text = str(content)
        if not html_text or len(html_text) < 100:
            return None
        
        return html_text
    except httpx.TimeoutException:
        logger.warning(
            "wikivoyage.html.timeout",
            destination=destination,
        )
        return None
    except httpx.HTTPError as exc:
        logger.warning(
            "wikivoyage.html.http_error",
            destination=destination,
            error=str(exc),
        )
        return None
    except Exception as exc:
        logger.warning(
            "wikivoyage.html.parse_error",
            destination=destination,
            error=str(exc),
        )
        return None


def _save_file(
    destination: str,
    filename: str,
    section_name: str,
    text: str,
) -> bool:
    """Save a destination document to a .txt file with standard headers."""
    try:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        file_path = RAW_DIR / filename

        # Build header
        source_title = f"Wikivoyage — {destination} — {section_name}"
        source_url = f"https://en.wikivoyage.org/wiki/{destination.replace(' ', '_')}"

        header = f"""destination: {destination}
source_title: {source_title}
source_url: {source_url}
---
"""
        content = header + text + "\n"

        file_path.write_text(content, encoding="utf-8")
        logger.info(
            "wikivoyage.save.success",
            destination=destination,
            filename=filename,
            chars=len(text),
        )
        return True
    except Exception as exc:
        logger.exception(
            "wikivoyage.save.error",
            destination=destination,
            filename=filename,
            error=str(exc),
        )
        return False


async def _process_destination(
    client: httpx.AsyncClient,
    destination: str,
) -> tuple[str, int]:
    """
    Fetch real content from Wikivoyage for one destination.
    Tries API first, then HTML page fallback.
    Returns (destination, files_created).
    """
    # Try API first
    html_text = await _fetch_from_api(client, destination)
    
    # If API fails, try HTML page fallback
    if not html_text:
        logger.info(
            "wikivoyage.fetch.trying_html_fallback",
            destination=destination,
        )
        html_text = await _fetch_from_html(client, destination)
    
    # If both fail, skip this destination
    if not html_text:
        logger.warning(
            "wikivoyage.fetch.skipped",
            destination=destination,
            reason="both API and HTML page fetch failed",
        )
        return destination, 0

    # Convert HTML to plain text
    clean_text = _clean_text(html_text)
    if not clean_text or len(clean_text) < 100:
        logger.warning(
            "wikivoyage.fetch.content_too_short",
            destination=destination,
            chars=len(clean_text),
        )
        return destination, 0

    # Split into two logical documents (if content is long enough)
    overview, details = _split_document(clean_text)

    files_created = 0

    # Save overview document
    overview_filename = f"{destination.lower().replace(' ', '_')}_overview.txt"
    if overview and _save_file(destination, overview_filename, "Overview", overview):
        files_created += 1

    # Save details document (if we have meaningful split)
    if details and len(details) > 50:
        details_filename = f"{destination.lower().replace(' ', '_')}_details.txt"
        if _save_file(destination, details_filename, "Details", details):
            files_created += 1
    elif clean_text and len(clean_text) > 50:
        # If split didn't work well, save whole document as single file
        single_filename = f"{destination.lower().replace(' ', '_')}_content.txt"
        if _save_file(destination, single_filename, "Content", clean_text):
            files_created += 1

    return destination, files_created


async def main() -> None:
    """Fetch all destinations from Wikivoyage and save raw document files."""
    logger.info("wikivoyage.start", destination_count=len(DESTINATIONS))

    # Create HTTP client with Wikimedia-compliant User-Agent
    headers = {
        "User-Agent": "SmartTravelPlannerAIEBootcamp/1.0 (educational project; contact: https://github.com/dev-jamal25)"
    }

    total_files_created = 0
    skipped_count = 0

    async with httpx.AsyncClient(headers=headers, timeout=HTTP_TIMEOUT) as client:
        for destination in DESTINATIONS:
            dest_name, files_created = await _process_destination(client, destination)
            total_files_created += files_created
            if files_created == 0:
                skipped_count += 1
            
            # Small delay between requests to be respectful to Wikivoyage
            await asyncio.sleep(REQUEST_DELAY)

    # Summary
    logger.info(
        "wikivoyage.complete",
        total_files=total_files_created,
        skipped=skipped_count,
        output_directory=str(RAW_DIR),
    )

    print(f"\n{'='*60}")
    print("✓ Wikivoyage fetch complete")
    print(f"  Files created: {total_files_created}")
    print(f"  Destinations skipped: {skipped_count}")
    print(f"  Output directory: {RAW_DIR}")
    print(f"{'='*60}")
    
    if skipped_count > 0:
        print(f"\n⚠ {skipped_count} destination(s) could not be fetched.")
        print("  Check logs above for details.")


if __name__ == "__main__":
    asyncio.run(main())
