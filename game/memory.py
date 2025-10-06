from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional
import json
import time
import uuid
import traceback
import inspect
from typing import get_type_hints
from abc import ABC, abstractmethod


@dataclass
class Prompt:
    """Enhanced prompt structure for BDD LLM interactions."""
    messages: List[Dict] = field(default_factory=list)
    tools: List[Dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

@dataclass(frozen=True)
class Goal:
    priority: int
    name: str
    description: str

class Memory:
    def __init__(self):
        self.items = []

    def add_memory(self, memory: dict):
        memory_item = memory.copy()
        memory_item["timestamp"] = time.time()
        self.items.append(memory_item)

    def get_memories(self, limit: int = None) -> List[Dict]:
        if limit:
            return self.items[-limit:]
        return self.items
    
    def clear_memory(self):
        self.items = []
    
    # def get_bdd_context(self, context_type: str = None) -> List[Dict]:
    #     """Get BDD-specific context like scenarios, features, test results."""
    #     if not context_type:
    #         return self.items[-5:]
        
    #     return [item for item in self.items[-10:] 
    #             if item.get('bdd_type') == context_type]

    # def get_feature_history(self) -> List[Dict]:
    #     """Get all feature-related memories."""
    #     return [item for item in self.items 
    #             if item.get('type') in ['feature', 'scenario', 'step_definition']]