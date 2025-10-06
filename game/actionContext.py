from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional
import json
import time
import uuid
import traceback
import inspect
from typing import get_type_hints
from abc import ABC, abstractmethod

from game.memory import Memory

class ActionContext:
    """ Class for shared resources that tools need access to. 
    """
    def __init__(self, properties: dict = None):
        self.properties = properties or {}

    def get(self, key: str, default=None):
        return self.properties.get(key, default)
    
    def set(self, key: str, value: Any):
        self.properties[key] = value

    def update(self, properties:dict):
        self.properties.update(properties)

def create_action_context_with_registry(registry, llm_function, memory: Memory, target_language: str = "python"):
    """Create action context with all necessary components"""
    from game.agent import AgentRegistry
    return ActionContext({
        "agent_registry": registry,
        "llm": llm_function,
        "memory": memory,
        "target_language": target_language
    })