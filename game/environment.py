from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional
import json
import time
import uuid
import traceback
import inspect
from typing import get_type_hints
from abc import ABC, abstractmethod

from game.actions import Action, ActionTransaction
from game.actionContext import ActionContext
from game.tools import has_named_parameter

class Environment:
    def execute_action(self, action: Action, args: dict) -> dict:
        try:
            result = action.execute(**args)
            return self.format_result(result)
        except Exception as e:
            return {
                "tool_executed": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def format_result(self, result: Any, success: bool = True) -> dict:
        return {
            "tool_executed": success,
            "result": result,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")
        }
    
class ActionContextEnvironment(Environment):
    """
    Environment with automatic dependency injection.
    Automatically provides action_context and other resources to tools.
    """
    def execute_action(self, action_context: ActionContext, 
                      action: Action, args: dict) -> dict:
        """Execute action with automatic dependency injection."""
        try:
            args_copy = args.copy()

            # Inject action_context if tool expects it
            if has_named_parameter(action.function, "action_context"):
                args_copy["action_context"] = action_context

            # Inject other context properties with underscore prefix
            for key, value in action_context.properties.items():
                param_name = "_" + key
                if has_named_parameter(action.function, param_name):
                    args_copy[param_name] = value
                
            result = action.execute(**args_copy)
            return self.format_result(result)
        except Exception as e:
            return {
                "tool_executed": False,
                "error": str(e)
            }

class StagedActionEnvironment(Environment):
    """
    Environment with review-before-commit patterns.
    Perfect for BDD where code generation should be reviewed before execution.
    """
    def __init__(self):
        self.staged_transactions = {}
        self.llm = None  # Will be set from action_context

    def stage_actions(self, task_id: str) -> ActionTransaction:
        """Create a new transaction for staging actions."""
        transaction = ActionTransaction()
        self.staged_transactions[task_id] = transaction
        return transaction

    def review_transaction(self, task_id: str) -> bool:
        """Have LLM review staged actions for safety."""
        transaction = self.staged_transactions.get(task_id)
        if not transaction:
            raise ValueError(f"No transaction found for task {task_id}")

        # Create description of staged actions
        staged_actions = [
            f"Action: {action.__class__.__name__}\nArgs: {args}"
            for action, args in transaction.actions
        ]
        
        review_prompt = f"""Review these staged BDD actions for safety:
        
        Task ID: {task_id}  @ 
        
        Staged Actions:
        {chr(10).join(staged_actions)}
        
        Consider:
        1. Are all actions necessary for the BDD task?
        2. Could any action have unintended consequences?
        3. Are the actions in a safe order?
        4. Will this properly implement the BDD scenario?
        5. Is there a safer way to achieve the same goal?
        
        Should these actions be approved?
        """
        
        if self.llm:
            response = self.llm.generate(review_prompt)
            return "approved" in response.lower()
        
        # Fallback: require manual approval
        print(f"Review required for transaction {task_id}")
        print(review_prompt)
        return input("Approve? (y/n): ").lower().startswith('y')
    

class AIReviewBDDEnvironment(Environment):
    """
    Environment where AI agents review each other's work automatically.
    No human interaction required - agents iterate until they're satisfied.
    """
    
    def __init__(self):
        self.context_env = ActionContextEnvironment()
        self.review_history = []
        self.max_review_iterations = 3
        
        # Define what needs AI review vs immediate execution
        self.immediate_operations = {
            'parse_gherkin',           # Parse .feature files → Always works
            'extract_scenarios',       # Extract data → Deterministic  
            'validate_gherkin_syntax', # Syntax check → Clear rules
            'read_existing_code',      # File reading → No side effects
        }
        
        self.ai_review_operations = {
            'generate_step_definitions',  # Code quality matters
            'generate_test_implementation', # Test logic correctness
            'generate_production_code',   # Business logic correctness
            'refactor_existing_code',     # Code quality and maintainability
        }

    def execute_with_ai_review(self, action_context: ActionContext, 
                              action: Action, args: dict) -> dict:
        """
        Execute operation with AI review loop if needed.
        """
        action_name = action.name
        
        if action_name in self.immediate_operations:
            # No review needed - execute immediately
            return self._execute_immediate(action_context, action, args)
            
        elif action_name in self.ai_review_operations:
            # AI review needed - iterate until approved
            return self._execute_with_review_loop(action_context, action, args)
            
        else:
            # Default to immediate execution
            return self._execute_immediate(action_context, action, args)

    def _execute_immediate(self, action_context: ActionContext, 
                          action: Action, args: dict) -> dict:
        """Execute immediately - no review needed."""
        print(f"✅ Immediate execution: {action.name}")
        return self.context_env.execute_action(action_context, action, args)

    def _execute_with_review_loop(self, agent, action_context: ActionContext, 
                                 action: Action, args: dict) -> dict:
        """
        Execute with AI review loop until quality standards are met.
        """
        print(f"🔄 Starting AI review loop for: {action.name}")
        
        for iteration in range(self.max_review_iterations):
            print(f"  📝 Attempt {iteration + 1}")
            
            # 1. Generate the code/content
            result = self.context_env.execute_action(agent, action_context, action, args)
            
            if not result.get('tool_executed'):
                return result  # Return error immediately
                
            generated_content = result['result']
            
            # 2. AI Review the generated content
            review_result = self._ai_review_content(
                action_context, action.name, generated_content, args
            )
            
            # 3. Check if approved
            if review_result['approved']:
                print(f"  ✅ Approved after {iteration + 1} attempts")
                return {
                    'tool_executed': True,
                    'result': generated_content,
                    'review_iterations': iteration + 1,
                    'review_feedback': review_result['feedback']
                }
            else:
                print(f"  ❌ Rejected: {review_result['feedback']}")
                # Update args with feedback for next iteration
                args['previous_attempt'] = generated_content
                args['review_feedback'] = review_result['feedback']
        
        # Failed to meet standards after max iterations
        return {
            'tool_executed': False,
            'error': f"Failed AI review after {self.max_review_iterations} attempts",
            'last_attempt': generated_content,
            'final_feedback': review_result['feedback']
        }

    def _ai_review_content(self, action_context: ActionContext, 
                          operation_type: str, content: str, original_args: dict) -> dict:
        """
        Have AI agent review generated content for quality and correctness.
        """
        llm = action_context.get('llm')
        
        # Create specific review prompts based on operation type
        review_prompts = {
            'generate_step_definitions': f"""
            Review this BDD step definitions code for:
            
            CONTENT TO REVIEW:
            {content}
            
            ORIGINAL REQUIREMENTS:
            {original_args.get('scenarios', 'N/A')}
            
            CHECK FOR:
            1. Correct Gherkin step matching (Given/When/Then)
            2. Proper parameter extraction and handling
            3. Clear, maintainable step implementations
            4. Appropriate assertions and validations
            5. No duplicate or conflicting step definitions
            6. Follows best practices for the framework
            
            APPROVE if the code is production-ready.
            REJECT if there are issues that need fixing.
            
            Response format:
            DECISION: [APPROVE/REJECT]
            FEEDBACK: [Specific feedback for improvement]
            """,
            
            'generate_test_implementation': f"""
            Review this test implementation code for:
            
            CONTENT TO REVIEW:
            {content}
            
            ORIGINAL REQUIREMENTS:
            {original_args.get('step_definitions', 'N/A')}
            
            CHECK FOR:
            1. Tests actually test the business requirements
            2. Proper test setup and teardown
            3. Clear test data and mocking
            4. Edge cases and error conditions covered
            5. Test independence (no test depends on another)
            6. Appropriate assertions and error messages
            
            APPROVE if tests are comprehensive and correct.
            REJECT if tests are incomplete or incorrect.
            
            Response format:
            DECISION: [APPROVE/REJECT]  
            FEEDBACK: [Specific feedback for improvement]
            """,
            
            'generate_production_code': f"""
            Review this production code for:
            
            CONTENT TO REVIEW:
            {content}
            
            ORIGINAL REQUIREMENTS:
            {original_args.get('requirements', 'N/A')}
            
            CHECK FOR:
            1. Code correctly implements business requirements
            2. Proper error handling and edge cases
            3. Clean, readable, maintainable code structure
            4. Follows SOLID principles and best practices
            5. Appropriate logging and monitoring
            6. Security considerations addressed
            7. Performance implications considered
            
            APPROVE if code is production-ready.
            REJECT if there are quality or correctness issues.
            
            Response format:
            DECISION: [APPROVE/REJECT]
            FEEDBACK: [Specific feedback for improvement]
            """
        }
        
        prompt = review_prompts.get(operation_type, f"""
        Review this generated content:
        {content}
        
        Response format:
        DECISION: [APPROVE/REJECT]
        FEEDBACK: [Feedback]
        """)
        
        # Get AI review
        review_response = llm(prompt)
        
        # Parse AI response
        approved = 'APPROVE' in review_response.upper()
        
        # Extract feedback (everything after "FEEDBACK:")
        feedback_start = review_response.find('FEEDBACK:')
        feedback = review_response[feedback_start + 9:].strip() if feedback_start != -1 else review_response
        
        return {
            'approved': approved,
            'feedback': feedback,
            'full_response': review_response
        }