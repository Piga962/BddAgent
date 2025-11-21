from typing import List, Dict, Any, Callable, Optional
import json
import time
from datetime import datetime

from game.memory import Memory, Prompt, Goal
from game.actions import Action, ActionRegistry
from game.actionContext import ActionContext
from game.environment import Environment, AIReviewBDDEnvironment
from game.capabilities import Capability
from game.agentLanguage import AgentLanguage

class Agent:
    def __init__(self,
                 goals: List[Goal],
                 agent_language: AgentLanguage,
                 action_registry: ActionRegistry,
                 generate_response: Callable[[Prompt], str],
                 environment: Environment,
                 agent_name: str = "Agent",
                 capabilities: List[Capability] = None,
                 max_iterations: int = 30):
        
        self.goals = goals
        self.generate_response = generate_response
        self.agent_language = agent_language
        self.action_registry = action_registry
        self.environment = environment
        self.agent_name = agent_name
        self.capabilities = capabilities or []
        self.max_iterations = max_iterations

    def construct_prompt(self, context: ActionContext, goals: List[Goal], memory: Memory) -> Prompt:
        return self.agent_language.construct_prompt(
            actions=self.action_registry.get_actions(),
            environment=self.environment,
            goals=goals,
            memory=memory
        )
    
    def get_action(self, response: str) -> tuple:
        invocation = self.agent_language.parse_response(response)
        action = self.action_registry.get_action(invocation["tool_name"])
        return action, invocation
    
    def should_terminate(self, action_context: ActionContext, response: str) -> bool:
        try:
            action_def, _ = self.get_action(response)
            return action_def.terminal
        except Exception as e:
            print(f"Error in termination check: {e}")
            return False
        
    def set_current_task(self, memory: Memory, task: str):
        memory.add_memory({
            "role": "user",
            "content": task,
            "timestamp": datetime.now().isoformat()
        })

    def update_memory(self, memory: Memory, response: str, result: dict):

        memory.add_memory({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })

        memory.add_memory({
            "role": "user",
            "content": json.dumps(result),
            "timestamp": datetime.now().isoformat()
        })

    def prompt_llm_for_action(self, context: ActionContext, full_prompt: Prompt) -> str:
        try:
            response = self.generate_response(full_prompt)
            return response
        except Exception as e:
            error_response = f"Error generating response: {str(e)}"
            print(f"LLM Error: {error_response}")
            return error_response
        
    def handle_agent_response(self, action_context: ActionContext, response: str) -> dict:
        try:
            action_def, action_invocation = self.get_action(response)
            print(f"Action chosen by {self.agent_name}: {action_def.name}")
            if hasattr(self.environment, 'execute_with_ai_review'):
                result = self.environment.execute_with_ai_review(
                    self, action_context, action_def, action_invocation["args"]
                )
            elif hasattr(self.environment, 'execute_action') and len(self.environment.execute_action.__code__.co_varnames) > 3:
                result = self.environment.execute_action(
                    action_context, action_def, action_invocation["args"]
                )
            else:
                result = self.environment.execute_action(action_def, action_invocation["args"])

            return result
        except Exception as e:
            return {
                "tool_executed": False,
                "error": f"Error handling agent response: {str(e)}",
                "response": response
            }
        
    def run(self, user_input: str, memory: Memory = None, action_context_props: dict= None) -> Memory:
        memory = memory or Memory()
        self.set_current_task(memory, user_input)

        action_context = ActionContext({
            'memory': memory,
            'llm': self.generate_response,
            'environment_type': type(self.environment).__name__,
            **(action_context_props or {})
        })

        for capability in self.capabilities:
            capability.init(self, action_context)

        iteration = 0
        try:
            for iteration in range(self.max_iterations):
                print(f"\n--- Iteration {iteration + 1}/{self.max_iterations} ---")

                for capability in self.capabilities:
                    capability.process_prompt(self,action_context,memory)

                prompt = self.construct_prompt(action_context, self.goals, memory)

                response = self.prompt_llm_for_action(action_context, prompt)

                for capability in self.capabilities:
                    capability.process_response(self,action_context,memory,response)

                result = self.handle_agent_response(action_context,response)

                for capability in self.capabilities:
                    capability.process_action(self,action_context,result)

                self.update_memory(memory, response, result)

                if self.should_terminate(action_context, response):
                    print("🏁 Termination condition met. Stopping agent.")
                    break

                if hasattr(self.environment, 'review_and_execute_staged'):
                    if hasattr(self.environment, 'current_task_id') and self.environment.current_task_id:
                        review_result = self.environment.review_and_execute_staged()
                        if review_result.get('success'):
                            print("Staged actions executed successfully.")
                        else:
                            print(f"Error executing staged actions: {review_result.get('message')}")
            
        except KeyboardInterrupt:
            print("Agent run interrupted by user.")
        except Exception as e:
            print(f"Unexpected error during agent run: {str(e)}")
        finally:
            for capability in self.capabilities:
                capability.terminate(self, action_context)

        print(f"Iterations: {iteration + 1}")
        print(f"Memory items: {len(memory.items)}")

        return memory
    
    
class AgentRegistry:
    def __init__(self):
        self.agents = {}

    def register_agent(self, name: str, run_function: callable):
        self.agents[name] = run_function

    def get_agent(self, name: str) -> Optional[callable]:
        return self.agents.get(name)
    
    def list_agents(self) -> List[str]:
        return list(self.agents.keys())
    
