from dataclasses import dataclass, field
import os
from typing import List, Dict, Any, Callable, Optional
import json
import time
import uuid
import traceback
import inspect
from typing import get_type_hints
from abc import ABC, abstractmethod

tools = {}
tools_by_tag = {}

def get_tool_metadata(func, tool_name=None, description=None,
                     parameters_override=None, terminal=False,
                     tags=None):
    """Extract metadata while ignoring special parameters like action_context."""
    signature = inspect.signature(func)
    type_hints = get_type_hints(func)

    args_schema = {
        "type": "object",
        "properties": {},
        "required": []
    }

    for param_name, param in signature.parameters.items():
        # Skip special parameters - agent doesn't need to know about these
        if param_name in ["action_context", "action_agent"] or \
           param_name.startswith("_"):
            continue

        # Add regular parameters to the schema
        param_type = type_hints.get(param_name, str)
        args_schema["properties"][param_name] = {
            "type": "string"  # Simplified for example, could be enhanced
        }

        if param.default == param.empty:
            args_schema["required"].append(param_name)

    return {
        "name": tool_name or func.__name__,
        "description": description or func.__doc__,
        "parameters": args_schema,
        "tags": tags or [],
        "terminal": terminal,
        "function": func
    }

def register_tool(tool_name=None, description=None, parameters_override=None, 
                 terminal=False, tags=None):
    """
    Decorator to automatically register BDD tools with metadata extraction.
    Perfect for BDD operations like parse_feature, generate_tests, implement_code.
    """
    def decorator(func):
        # Extract metadata using the helper function
        metadata = get_tool_metadata(
            func=func,
            tool_name=tool_name,
            description=description,
            parameters_override=parameters_override,
            terminal=terminal,
            tags=tags
        )

        # Register in global tools dictionary
        tools[metadata["name"]] = {
            "description": metadata["description"],
            "parameters": metadata["parameters"],
            "function": metadata["function"],
            "terminal": metadata["terminal"],
            "tags": metadata["tags"]
        }

        # Register by tags for easy filtering
        for tag in metadata["tags"]:
            if tag not in tools_by_tag:
                tools_by_tag[tag] = []
            tools_by_tag[tag].append(metadata["name"])

        return func
    return decorator

def has_named_parameter(func: Callable, param_name: str) -> bool:
    """Check if function has a specific parameter name."""
    signature = inspect.signature(func)
    return param_name in signature.parameters
