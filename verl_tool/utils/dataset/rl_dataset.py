"""
Utility functions and classes for RL dataset processing.
"""
from copy import deepcopy
from typing import Any, List


def nested_copy(obj: Any) -> Any:
    """Perform a deep copy of nested structures.
    
    Args:
        obj: The object to copy (can be dict, list, or any nested structure)
        
    Returns:
        A deep copy of the input object
    """
    return deepcopy(obj)


class RolloutMessagesMixin:
    """A wrapper class for rollout messages.
    
    This class wraps a list of messages to provide a consistent interface
    for rollout message handling in the dataset pipeline.
    
    Attributes:
        messages: List of message dictionaries
    """
    
    def __init__(self, messages: List[dict]):
        """Initialize with a list of messages.
        
        Args:
            messages: List of message dictionaries
        """
        self.messages = messages
    
    def __iter__(self):
        """Make the class iterable."""
        return iter(self.messages)
    
    def __getitem__(self, index):
        """Support indexing."""
        return self.messages[index]
    
    def __len__(self):
        """Return the length of messages."""
        return len(self.messages)
    
    def __repr__(self):
        """String representation."""
        return f"RolloutMessagesMixin({self.messages})"
    
    def __str__(self):
        """String representation."""
        return str(self.messages)

