import datetime
from game.actionContext import ActionContext
from game.memory import Memory
from typing import List

class Capability:
    def init(self, action_context: ActionContext):
        """Initialize capability with the action context."""
        pass

    def process_prompt(self, action_context: ActionContext, memory: Memory):
        """Modify the prompt or action context before LLM call."""
        pass

    def process_response(self, action_context: ActionContext, memory: Memory, response: str):
        """Process the LLM response before action handling."""
        pass

    def process_action(self, action_context: ActionContext, result: dict):
        """Process the action before execution."""
        pass

    def terminate(self, action_context: ActionContext):
        """Clean up resources or finalize state."""
        pass

class ProgressTrackingCapability(Capability):
    def init(self):
        self.actions_completed = 0
        self.errors_encountered = 0

    def process_action(self, action_context, result):
        """Track action results."""
        if result.get('tool_executed'):
            self.actions_completed += 1
            print(f"Progress: {self.actions_completed} actions completed.")
        else:
            self.errors_encountered += 1
            print(f"Errors: {self.errors_encountered} encountered.")

class TimeAwareCapability(Capability):
    def process_prompt(self, action_context: ActionContext, memory: Memory):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        action_context.set('current_time', current_time)