"""Prompts for Haiku model mechanical tasks in the agent pipeline."""

INTENT_EXTRACTION_PROMPT = """Extract structured travel intent from the user message.

If the user message contains travel planning context, extract:
- budget_usd: approximate budget (or null if not mentioned)
- duration_days: trip length in days (or null if not mentioned)
- travel_month: month or season (or null if not mentioned)
- preferred_style: travel style (adventure, relaxation, culture, budget, luxury, family, or null)
- preferred_activities: list of activities mentioned (hiking, beach, museums, etc.)
- climate_preference: warm, cold, moderate, or null
- candidate_destination: if a specific destination is mentioned, extract it (otherwise null)

Return as JSON matching TripIntent schema.
If the message is not travel-related, return all fields as null and note the reason.
"""

RAG_QUERY_REWRITING_PROMPT = """Rewrite user travel query for better vector search matching.

Input: user's original travel question
Output: 2-3 optimized search queries for RAG retrieval

Example:
Input: "I want a fun beach destination with good nightlife and not too expensive"
Output:
- "beach vacation nightlife tourism"
- "affordable beach resort destination"
- "coastal city entertainment activities"

Make queries specific but not too long. Include destination types and activities, not adjectives.
"""

DESTINATION_NORMALIZATION_PROMPT = """Normalize destination name to supported format.

User may write: "Bali", "BALI", "bali ", "Ballí", etc.
Your job: Match against known destinations (exact or fuzzy match).

Supported destinations:
- Interlaken
- Banff
- Bali
- Santorini
- Kyoto
- Istanbul
- Tbilisi
- Kraków (also accept "Krakow")
- Dubai
- Singapore

Input: user-provided destination string
Output: exact destination name or null if no match

Be case-insensitive. Strip whitespace. Handle common misspellings.
"""

TOOL_ARGUMENT_REPAIR_PROMPT = """Fix invalid tool arguments before execution.

Tool input may have validation errors (invalid types, out-of-range values, etc).

Errors might be:
- top_k outside [1, 10]
- forecast_days outside [1, 7]
- destination_filter containing unsupported destination
- non-empty string validation failure

If repairable, return corrected arguments.
If not repairable, return original arguments with explanation (tool will return error).

Be conservative: only fix obvious mistakes (type coercion, clamping to range).
"""
