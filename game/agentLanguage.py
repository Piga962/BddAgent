import json
from typing import List, Dict, Any
from abc import ABC, abstractmethod

from game.memory import Memory, Prompt, Goal
from game.actions import Action
from game.environment import Environment

class AgentLanguage(ABC):

    def __init__(self):
        pass

    def construct_prompt(self,
                         actions: List[Action],
                         environment: Environment,
                         goals: List[Goal],
                         memory: Memory) -> Prompt:
        raise NotImplementedError("Must implement construct_prompt method")

    def parse_response(self, ressponse: str) -> dict:
        raise NotImplementedError("Subclasses must implement parse_response method")
    
class AgentFunctionCallingActionLanguage(AgentLanguage):
    def __init__(self):
        super().__init__()

    def format_goals(self, goals: List[Goal]):
        if not goals:
            return []
        
        sorted_goals = sorted(goals, key=lambda g: g.priority)

        sep = "\n" + "="*50 + "\n"
        goal_instructions = "\n\n".join([
            f"GOAL {goal.priority}: {goal.name}{sep}{goal.description}{sep}"
            for goal in sorted_goals
        ])

        system_message = f"""

{goal_instructions}

INSTRUCTIONS:
- Focus on achieving your goals in priority order
- Usae the available tools to accomplist tasks efficiently
- Be thorough but concise with you reasoning
- Handle error gracefully and adapt your approach as needed
- Always strive to privde high-quality results

When you need to use a tool, the system will handle the function calling automatically.

Think step by step and choose the mostg appropriate tool for each task.
"""
        return [{
            "role": "system",
            "content": system_message
        }]

    def format_memory(self, memory: Memory) -> List[Dict]:
        items = memory.get_memories()
        mapped_items = []

        for item in items:
            content = item.get("content", None)
            if not content:
                content = json.dumps(item, indent=2)

            if item.get("role") == "assistant":
                mapped_items.append({"role": "assistant", "content": content})
            elif item.get("role") == "environment":
                mapped_items.append({"role": "assistant", "content": content})
            else:
                mapped_items.append({"role": "user", "content": content})

        return mapped_items
    
    def format_actions(self, actions: List[Action]) -> List[Dict]:
        tools = []

        for action in actions:
            tool_def ={
                "type": "function",
                "function": {
                    "name": action.name,
                    "description": action.description[:1024] if action.description else f"Execute {action.name}",
                    "parameters": action.parameters or {
                        "type": "object",
                        "properties": {},
                        "required": []
                    },
                },
            }
            tools.append(tool_def)
        return tools
    
    def construct_prompt(self,
                         actions: List[Action],
                         environment: Environment,
                         goals: List[Goal],
                         memory: Memory) -> Prompt:
        
        prompt_messages = []
        prompt_messages.extend(self.format_goals(goals))
        prompt_messages.extend(self.format_memory(memory))

        tools = self.format_actions(actions)

        return Prompt(
            messages=prompt_messages,
            tools=tools,
            metadata={
                "agent_language": "function_calling",
                "num_goals": len(goals),
                "num_actions": len(actions),
                "memory_items": len(memory.get_memories()) if memory else 0 
            }
        )
    
    def parse_response(self, response: str) -> dict:
        try:
            parsed = json.loads(response)

            if "tool_name" in parsed and "args" in parsed:
                return parsed
            elif "function" in parsed and "arguments" in parsed:
                return {
                    "tool_name": parsed["function"]["name"],
                    "args": parsed["function"]["arguments"]
                }
            else:
                return {
                    "tool_name": "terminate",
                    "args": {"message": response}
                }
            
        except json.JSONDecodeError:
            return {
                "tool_name": "terminate",
                "args": {"message": response}
            }
        
    def adapt_prompt_after_parsing_error(self,
                                         prompt: Prompt,
                                         response: str,
                                         traceback: str,
                                         error: Any,
                                         retries_left: int) -> Prompt:
        """
        Adapt prompt after parsing errors (extensibility hook).
        
        Can be overridden to implement retry logic with modified prompts.
        """
        return prompt