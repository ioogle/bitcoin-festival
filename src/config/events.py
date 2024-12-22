"""Configuration for major market events that impacted Bitcoin price."""

import json
from enum import Enum
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, Union

class EventType(Enum):
    """Types of events and their impact on the market."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class EventCategory(Enum):
    """Categories of events."""
    FED = "fed"
    MAJOR = "major"
    HALVING = "halving"

@dataclass
class Event:
    date: datetime
    event: str
    description: str
    type: EventType
    impact: str
    category: EventCategory
    block_height: Optional[int] = None

    @classmethod
    def from_dict(cls, data: dict) -> 'Event':
        """Create an Event instance from a dictionary."""
        return cls(
            date=datetime.strptime(data['date'], '%Y-%m-%d'),
            event=data['event'],
            description=data['description'],
            type=EventType(data['type']),
            impact=data['impact'],
            category=EventCategory(data['category']),
            block_height=data.get('block_height')
        )

    def event_color(self) -> str:
        """Return the color of the event based on its type."""
        if self.type == EventType.POSITIVE:
            return 'green'
        elif self.type == EventType.NEGATIVE:
            return 'red'
        return 'gray'

    def to_dict(self) -> dict:
        """Convert Event to dictionary format."""
        return {
            'date': self.date,
            'event': self.event,
            'description': self.description,
            'type': self.type.value,
            'impact': self.impact,
            'category': self.category.value,
            'block_height': self.block_height
        }

class EventManager:
    def __init__(self):
        self.events = self._load_events()
    
    def _load_events(self) -> List[Event]:
        """Load events from JSON file."""
        events_file = Path(__file__).parent / 'events.json'
        with open(events_file, 'r') as f:
            data = json.load(f)
        
        events = []
        # Load all events using the from_dict classmethod
        for category in ['fed_events', 'major_events', 'halving_events']:
            for event_data in data[category]:
                events.append(Event.from_dict(event_data))
        
        return events
    
    def get_events_by_category(self, categories: Union[str, List[str], EventCategory, List[EventCategory]]) -> List[Event]:
        """Get events filtered by one or more categories.
        
        Args:
            categories: Single category or list of categories. Can be strings or EventCategory enums.
            
        Returns:
            List of events matching any of the specified categories.
        """
        # Convert single category to list
        if not isinstance(categories, (list, tuple)):
            categories = [categories]
        
        # Convert string categories to enums
        category_enums = []
        for category in categories:
            if isinstance(category, str):
                category_enums.append(EventCategory(category))
            elif isinstance(category, EventCategory):
                category_enums.append(category)
            else:
                raise ValueError(f"Invalid category type: {type(category)}")
        
        return [e for e in self.events if e.category in category_enums]
    
    def get_events_by_type(self, type_: str) -> List[Event]:
        """Get events filtered by type."""
        type_enum = EventType(type_)
        return [e for e in self.events if e.type == type_enum]
    
    def get_events_in_range(self, start_date: datetime, end_date: datetime) -> List[Event]:
        """Get events within a date range."""
        return [e for e in self.events if start_date <= e.date <= end_date]
    
    def to_dict_list(self, events: List[Event]) -> List[dict]:
        """Convert Event objects to dictionary format."""
        return [
            {
                'date': e.date,
                'event': e.event,
                'description': e.description,
                'type': e.type.value,
                'impact': e.impact,
                'category': e.category.value,
                'block_height': e.block_height
            }
            for e in events
        ]

# Initialize event manager
event_manager = EventManager()

# Export commonly used event lists
FED_EVENTS = event_manager.get_events_by_category(EventCategory.FED)
MAJOR_EVENTS = event_manager.get_events_by_category(EventCategory.MAJOR)
HALVING_EVENTS = event_manager.get_events_by_category(EventCategory.HALVING)

# Export combined event lists
ALL_MARKET_EVENTS = event_manager.get_events_by_category([EventCategory.MAJOR, EventCategory.FED])
ALL_EVENTS = event_manager.get_events_by_category([cat for cat in EventCategory]) 