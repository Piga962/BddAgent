from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional
import json
import time
import uuid
import traceback
import inspect
from typing import get_type_hints
from abc import ABC, abstractmethod

from game.tools import tools


class Action:
    def __init__(self,
                 name: str,
                 function: Callable,
                 description: str,
                 parameters: Dict,
                 terminal: bool = False):
        self.name = name
        self.function = function
        self.description = description
        self.parameters = parameters
        self.terminal = terminal

    def execute(self, **args) -> Any:
        return self.function(**args)
    
class ActionRegistry:
    def __init__(self):
        self.actions = {}

    def register(self, action: Action):
        self.actions[action.name] = action
    
    def get_action(self, name: str) -> Action:
        if name not in self.actions:
            raise ValueError(f"Action '{name}' not found in registry")
        return self.actions[name]
    
    def get_actions(self) -> List[Action]:
        return list(self.actions.values())
    
class DecoratorActionRegistry(ActionRegistry):
    def __init__(self, tags: List[str] = None, tool_names: List[str] = None):
        super().__init__()
        self.terminate_tool = None

        for tool_name, tool_desc in tools.items():
            if tool_name == "terminate":
                self.terminate_tool = tool_desc

            if tool_names and tool_name not in tool_names:
                continue

            tool_tags = tool_desc.get("tags", [])
            if tags and not any(tag in tool_tags for tag in tags):
                continue

            self.register(Action(
                name = tool_name,
                function=tool_desc["function"],
                description=tool_desc["description"],
                parameters=tool_desc.get("parameters", {}),
                terminal=tool_desc.get("terminal", False)
            ))

    def register_terminate_tool(self):
        if self.terminate_tool:
            self.register(Action(
                name="terminate",
                function=self.terminate_tool["function"],
                description=self.terminate_tool["description"],
                parameters=self.terminate_tool.get("parameters",{}),
                terminal=self.terminate_tool.get("terminal", False)
            ))
        else:
            raise Exception("Terminate tool not found in global registry")
        
class ReversibleAction:
    def __init__(self, execute_func: Callable, reverse_func: Callable):
        self.execute_func = execute_func
        self.reverse_func = reverse_func
        self.execution_record = None

    def run(self, **args):
        result = self.execute_func(**args)
        self.execution_record = {"args": args, "result": result}
        return result
    
    async def undo(self):
        if not self.execution_record:
            raise ValueError("No action to reverse")
        return self.reverse_func(**self.execution_record)

class ActionTransaction:
    def __init__(self):
        self.actions = []
        self.executed = []
        self.commited = False
        self.transaction_id = str(uuid.uuid4())

    def add(self, action: ReversibleAction, **args):
        if self.commited:
            raise ValueError("Transaction already commited")
        self.actions.append((action, args))

    async def execute(self):
        try:
            for action, args in self.actions:
                result = action.run(**args)
                self.executed.append(action)
        except Exception as e:
            await self.rollback()
            raise e
        
    async def rollback(self):
        for action in reversed(self.executed):
            await action.undo()
        self.executed = []

    def commit(self):
        self.committed = True