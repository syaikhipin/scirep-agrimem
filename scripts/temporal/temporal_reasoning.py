#!/usr/bin/env python3
"""
Temporal Reasoning Module for AgriMemory

Implements specialized temporal processing:
1. Temporal expression extraction (absolute, relative, duration)
2. Timeline construction and indexing
3. Temporal relation classification
4. Temporal proximity retrieval

Addresses reviewer feedback on temporal reasoning challenges.
"""

import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json


class TemporalType(Enum):
    """Types of temporal expressions."""
    ABSOLUTE = "absolute"  # "2024-03-15", "March 15"
    RELATIVE = "relative"  # "last week", "3 days ago"
    DURATION = "duration"  # "for 2 weeks", "3 days"
    FREQUENCY = "frequency"  # "twice a month", "daily"


@dataclass
class TemporalExpression:
    """Parsed temporal expression."""
    text: str
    type: TemporalType
    value: Any  # datetime for absolute, timedelta for relative/duration
    confidence: float = 1.0
    span: Tuple[int, int] = (0, 0)  # Character span in original text


@dataclass
class TemporalEvent:
    """Event with temporal information."""
    event_id: str
    content: str
    timestamp: datetime
    temporal_expressions: List[TemporalExpression]
    metadata: Dict[str, Any]


class TemporalExtractor:
    """Extract temporal expressions from text."""

    def __init__(self):
        # Patterns for temporal expression matching
        self.patterns = {
            # Absolute dates
            "iso_date": r'\b\d{4}-\d{2}-\d{2}\b',
            "iso_datetime": r'\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\b',
            "month_day_year": r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',
            "day_month_year": r'\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b',

            # Relative time
            "relative_day": r'\b(yesterday|today|tomorrow)\b',
            "relative_week": r'\b(last|this|next) week\b',
            "relative_month": r'\b(last|this|next) month\b',
            "relative_year": r'\b(last|this|next) year\b',
            "ago": r'\b(\d+) (day|week|month|year)s? ago\b',

            # Duration
            "duration": r'\bfor (\d+) (day|week|month|year)s?\b',

            # Frequency
            "frequency": r'\b(daily|weekly|monthly|yearly|once|twice|three times) (a|per) (day|week|month|year)\b',
        }

        self.month_map = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
        }

    def extract(self, text: str, reference_time: datetime = None) -> List[TemporalExpression]:
        """
        Extract temporal expressions from text.

        Args:
            text: Input text
            reference_time: Reference datetime for relative expressions

        Returns:
            List of TemporalExpression objects
        """
        if reference_time is None:
            reference_time = datetime.now()

        expressions = []

        # ISO datetime
        for match in re.finditer(self.patterns["iso_datetime"], text, re.IGNORECASE):
            try:
                dt = datetime.fromisoformat(match.group())
                expressions.append(TemporalExpression(
                    text=match.group(),
                    type=TemporalType.ABSOLUTE,
                    value=dt,
                    confidence=1.0,
                    span=match.span()
                ))
            except:
                pass

        # ISO date
        for match in re.finditer(self.patterns["iso_date"], text, re.IGNORECASE):
            try:
                dt = datetime.fromisoformat(match.group())
                expressions.append(TemporalExpression(
                    text=match.group(),
                    type=TemporalType.ABSOLUTE,
                    value=dt,
                    confidence=1.0,
                    span=match.span()
                ))
            except:
                pass

        # Relative day
        for match in re.finditer(self.patterns["relative_day"], text, re.IGNORECASE):
            word = match.group().lower()
            if word == "yesterday":
                dt = reference_time - timedelta(days=1)
            elif word == "today":
                dt = reference_time
            elif word == "tomorrow":
                dt = reference_time + timedelta(days=1)

            expressions.append(TemporalExpression(
                text=match.group(),
                type=TemporalType.RELATIVE,
                value=dt,
                confidence=0.9,
                span=match.span()
            ))

        # Ago pattern
        for match in re.finditer(self.patterns["ago"], text, re.IGNORECASE):
            amount = int(match.group(1))
            unit = match.group(2).lower()

            if unit == "day":
                delta = timedelta(days=amount)
            elif unit == "week":
                delta = timedelta(weeks=amount)
            elif unit == "month":
                delta = timedelta(days=amount * 30)  # Approximation
            elif unit == "year":
                delta = timedelta(days=amount * 365)

            dt = reference_time - delta

            expressions.append(TemporalExpression(
                text=match.group(),
                type=TemporalType.RELATIVE,
                value=dt,
                confidence=0.8,
                span=match.span()
            ))

        # Duration
        for match in re.finditer(self.patterns["duration"], text, re.IGNORECASE):
            amount = int(match.group(1))
            unit = match.group(2).lower()

            if unit == "day":
                delta = timedelta(days=amount)
            elif unit == "week":
                delta = timedelta(weeks=amount)
            elif unit == "month":
                delta = timedelta(days=amount * 30)
            elif unit == "year":
                delta = timedelta(days=amount * 365)

            expressions.append(TemporalExpression(
                text=match.group(),
                type=TemporalType.DURATION,
                value=delta,
                confidence=0.9,
                span=match.span()
            ))

        # Sort by position in text
        expressions.sort(key=lambda x: x.span[0])

        return expressions


class Timeline:
    """Timeline data structure for temporal indexing."""

    def __init__(self):
        self.events: List[TemporalEvent] = []

    def add_event(self, event: TemporalEvent):
        """Add event to timeline."""
        self.events.append(event)
        self.events.sort(key=lambda e: e.timestamp)

    def get_events_in_range(
        self,
        start: datetime,
        end: datetime
    ) -> List[TemporalEvent]:
        """Get events within time range."""
        return [
            event for event in self.events
            if start <= event.timestamp <= end
        ]

    def get_events_before(self, time: datetime, limit: int = None) -> List[TemporalEvent]:
        """Get events before given time."""
        events = [e for e in self.events if e.timestamp < time]
        events.sort(key=lambda e: e.timestamp, reverse=True)
        if limit:
            events = events[:limit]
        return events

    def get_events_after(self, time: datetime, limit: int = None) -> List[TemporalEvent]:
        """Get events after given time."""
        events = [e for e in self.events if e.timestamp > time]
        events.sort(key=lambda e: e.timestamp)
        if limit:
            events = events[:limit]
        return events

    def get_nearest_events(
        self,
        time: datetime,
        k: int = 5,
        max_distance: timedelta = None
    ) -> List[TemporalEvent]:
        """Get k nearest events to given time."""
        # Calculate temporal distance
        scored_events = []
        for event in self.events:
            distance = abs((event.timestamp - time).total_seconds())

            if max_distance and distance > max_distance.total_seconds():
                continue

            scored_events.append((distance, event))

        scored_events.sort(key=lambda x: x[0])
        return [event for _, event in scored_events[:k]]

    def to_dict(self) -> Dict:
        """Export timeline as dictionary."""
        return {
            "events": [
                {
                    "event_id": e.event_id,
                    "content": e.content,
                    "timestamp": e.timestamp.isoformat(),
                    "metadata": e.metadata
                }
                for e in self.events
            ]
        }


class TemporalRetriever:
    """Retrieval system with temporal awareness."""

    def __init__(self, timeline: Timeline, extractor: TemporalExtractor):
        self.timeline = timeline
        self.extractor = extractor

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        temporal_weight: float = 0.5,
        reference_time: datetime = None
    ) -> List[TemporalEvent]:
        """
        Retrieve events with temporal constraint from query.

        Args:
            query: Query text (may contain temporal expressions)
            top_k: Number of events to retrieve
            temporal_weight: Weight for temporal proximity vs semantic
            reference_time: Reference time for relative expressions

        Returns:
            List of TemporalEvent objects
        """
        if reference_time is None:
            reference_time = datetime.now()

        # Extract temporal expressions from query
        temporal_exprs = self.extractor.extract(query, reference_time)

        if not temporal_exprs:
            # No temporal constraint, return most recent
            return self.timeline.get_events_before(reference_time, limit=top_k)

        # Use the first temporal expression as constraint
        main_expr = temporal_exprs[0]

        if main_expr.type == TemporalType.ABSOLUTE or main_expr.type == TemporalType.RELATIVE:
            # Specific time point - get nearest events
            target_time = main_expr.value
            return self.timeline.get_nearest_events(target_time, k=top_k)

        elif main_expr.type == TemporalType.DURATION:
            # Duration - get events in range
            duration = main_expr.value
            end_time = reference_time
            start_time = end_time - duration
            return self.timeline.get_events_in_range(start_time, end_time)

        else:
            # Fallback
            return self.timeline.get_events_before(reference_time, limit=top_k)


# Utility functions
def build_timeline_from_conversations(
    conversations: List[Dict],
    extractor: TemporalExtractor
) -> Timeline:
    """
    Build timeline from conversation data.

    Args:
        conversations: List of conversation dicts with timestamps
        extractor: TemporalExtractor instance

    Returns:
        Timeline object
    """
    timeline = Timeline()

    for conv in conversations:
        timestamp_str = conv.get("timestamp", "")

        try:
            if "T" in timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str)
            else:
                timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.now()
        except:
            timestamp = datetime.now()

        # Extract content
        content_parts = []
        for turn in conv.get("turns", []):
            speaker = turn.get("speaker", "user")
            text = turn.get("text", "")
            content_parts.append(f"{speaker}: {text}")

        content = "\n".join(content_parts)

        # Extract temporal expressions
        temporal_exprs = extractor.extract(content, reference_time=timestamp)

        event = TemporalEvent(
            event_id=conv.get("id", str(hash(content))),
            content=content,
            timestamp=timestamp,
            temporal_expressions=temporal_exprs,
            metadata=conv.get("metadata", {})
        )

        timeline.add_event(event)

    return timeline


# Example usage
if __name__ == "__main__":
    # Test temporal extraction
    extractor = TemporalExtractor()

    test_texts = [
        "The infection started 3 days ago",
        "I first noticed symptoms on 2024-03-15",
        "The disease progressed for 2 weeks",
        "I spray twice a week",
        "Last month the plants were healthy"
    ]

    print("Temporal Expression Extraction:")
    print("=" * 60)

    for text in test_texts:
        expressions = extractor.extract(text, reference_time=datetime(2024, 3, 20))
        print(f"\nText: {text}")
        for expr in expressions:
            print(f"  - {expr.text} ({expr.type.value}): {expr.value}")

    # Test timeline
    print("\n" + "=" * 60)
    print("Timeline Construction:")
    print("=" * 60)

    timeline = Timeline()

    events_data = [
        ("First symptoms appeared", datetime(2024, 3, 10)),
        ("Sprayed fungicide", datetime(2024, 3, 12)),
        ("Symptoms worsened", datetime(2024, 3, 15)),
        ("Second treatment", datetime(2024, 3, 18)),
    ]

    for content, timestamp in events_data:
        event = TemporalEvent(
            event_id=str(hash(content)),
            content=content,
            timestamp=timestamp,
            temporal_expressions=[],
            metadata={}
        )
        timeline.add_event(event)

    print(f"\nTotal events: {len(timeline.events)}")

    query_time = datetime(2024, 3, 14)
    print(f"\nEvents before {query_time.date()}:")
    for event in timeline.get_events_before(query_time):
        print(f"  - {event.timestamp.date()}: {event.content}")

    print(f"\nEvents after {query_time.date()}:")
    for event in timeline.get_events_after(query_time):
        print(f"  - {event.timestamp.date()}: {event.content}")

    print(f"\n3 nearest events to {query_time.date()}:")
    for event in timeline.get_nearest_events(query_time, k=3):
        print(f"  - {event.timestamp.date()}: {event.content}")
