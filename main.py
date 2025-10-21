import json
from typing import List, Dict
import os
import re
import traceback

from game.actionContext import create_action_context_with_registry
from game.actions import DecoratorActionRegistry
from game.agent import Agent, AgentRegistry
from game.environment import ActionContextEnvironment
from game.llms import create_simple_llm_function
from game.memory import Goal, Memory
from game.agentLanguage import AgentFunctionCallingActionLanguage

import tools.agentTools, tools.fileTools, tools.promptTools, tools.otherTools


class DevEvalProcessor:
    def __init__(self,
                 lm_prompt_jsonl_path: str,
                 mode: str,
                 output_path: str = "unified_test_data.jsonl"):
        self.lm_prompt_jsonl_path = lm_prompt_jsonl_path
        self.mode = mode
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        self.tests = self.load_tests()
        self.results = []

    def load_tests(self) -> List[Dict]:
        tests = []
        with open(self.lm_prompt_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    test_case = json.loads(line)
                    input_code = test_case['input_code']
                    namespace = test_case['namespace']
                    newJSON = {
                        "namespace": namespace,
                        "input_code": input_code,
                    }   
                    if self.mode == 'local_file_completion':
                        newJSON['context_above'] = test_case['contexts_above']
                    elif self.mode == 'local_file_infiling':
                        newJSON['context_above'] = test_case['contexts_above']
                        newJSON['context_below'] = test_case['contexts_below']
                    
                    tests.append(newJSON)

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON line: {line}\nError: {str(e)}")
        return tests[0:100]  # For testing, limit to first 2 tests
    
    def generate_jsonl(self):
        out_file = os.path.join(self.output_path, self.mode + '_results.jsonl')
        with open(out_file, 'w', encoding='utf-8') as f:
            for result in self.results:
                json_line = json.dumps(result)
                f.write(json_line + '\n')
        print(f"Generated JSONL file at {out_file}")

    def process(self):
        llm_function = create_simple_llm_function("gemini/gemini-2.0-flash")
        for test in self.tests:
            try:
                namespace = test['namespace']
                requirement = test['input_code']
                if 'context_above' in test:
                    context_above = test['context_above']
                if 'context_below' in test:
                    context_below = test['context_below']
                sharedMemory = Memory()
                registry = AgentRegistry()

                codeReviewer = create_code_reviewer_agent(llm_function)
                mainAgent = Agent(
                    goals=[
                        Goal(1, "Requirements Analysis",
                              "Analyze the requirements and context if given."),
                        Goal(2, "Function Implementation",
                              "Implement the function according to the requirements. With proper indentation (4 spaces)."),
                        Goal(3, "Extract and finalize",
                              "Extract the final code from the output and terminate.")
                    ],
                    agent_language=AgentFunctionCallingActionLanguage(),
                    action_registry=DecoratorActionRegistry(tags=["expert", "agent", "coordination", "analysis", "file_operations", "general"]),
                    generate_response=llm_function,
                    environment=ActionContextEnvironment(),
                    agent_name="DevEvalAgent",
                    max_iterations=8
                )
                codingAgent = Agent(
                    goals = [
                        Goal(1, "Code Implementation",
                             "Implement the function as per the requirements and context."),
                        Goal(2, "Code Quality",
                             "Write clean, maintainable code with proper error handling. But only focus on the function body. Do not write the function signature. Generate the code with proper indentation (4 spaces)."),
                    ],
                    agent_language=AgentFunctionCallingActionLanguage(),
                    action_registry=DecoratorActionRegistry(tags=["expert", "file_operations", "coding"]),
                    generate_response=llm_function,
                    environment=ActionContextEnvironment(),
                    agent_name="CodingAgent",
                    max_iterations=8
                )
                
                registry.register_agent("code_reviewer", codeReviewer)
                registry.register_agent("coding_agent", codingAgent)
                registry.register_agent("main_agent", mainAgent)

                action_context = {
                    "agent_registry": registry,
                    "target_language": "python",
                    "project_type": "function_implementation",
                }
                if self.mode == 'without_context':
                    task = f"""
Given this requirements, implement ONLY the function body (without signature):
Requirements:
{requirement}

Return only the function body without any additional explanations, and with proper indentation (4 spaces).
"""
                elif self.mode == 'local_file_completion':
                    task = f"""
Given this context and requirements, implement ONLY the function body (without signature):
Context:
{context_above}

Requirements:
{requirement}

Return ONLY the function body without any additional explanations, and with proper indentation (4 spaces).
"""
                elif self.mode == 'local_file_infiling':
                    task = f"""
Given this context and requirements, implement ONLY the function body (without signature):
Context above:
{context_above}

Context below:
{context_below}

Requirements:
{requirement}

Return ONLY the function body without any additional explanations, with proper indentation (4 spaces).
"""

                result_memory = mainAgent.run(
                    user_input=task,
                    memory=sharedMemory,
                    action_context_props=action_context
                )
                if result_memory.items:
                    final_result = result_memory.items[-1].get("content", {})
                    final_code = ""
                    try:
                        final_result_data = json.loads(final_result)
                        result_content = final_result_data.get("result", "")
                        code_match = re.search(r"```python\n(.*?)\n```", result_content, re.DOTALL)

                        if code_match:
                            final_code = code_match.group(1)
                        else:
                            if "🎉 Agent session completed." in result_content:
                                final_code = result_content.split("🎉 Agent session completed.")[0].strip()
                            else:
                                final_code = result_content.strip()
                        print(f"Final extracted code for test {namespace}:\n{final_code}\n")


                    # print(final_result)
                    # print(f"\nFinal Result for test {namespace}: ", final_result.get("content", "No content"))
                        self.results.append({
                            "namespace": namespace,
                            "completion": final_code
                        })
                    except json.JSONDecodeError as e:
                        print(f"Error decoding final result JSON for test {namespace}: {str(e)}")
                        self.results.append({
                            "namespace": namespace,
                            "completion": final_result
                        })

            except Exception as e:
                print(f"Error processing test {test}: {str(e)}")
                traceback.print_exc()
        self.generate_jsonl()

    


def create_project_manager_agent(llm_function) -> Agent:
    goals = [
        Goal(1, "Requirments Analysis",
                "Analyze the project requirements, break them into tasks, and coordinate team workflow"),
        Goal(2, "Project Planning",
                "Create implementation plans and ensure all requirements are met"),
        Goal(3, "Quality Coordination",
                "Ensure code quality through proper review and testing workflows")    
    ]

    action_registry = DecoratorActionRegistry(tags=["expert", "agent", "coordination", "analysis", "file_operations", "general"])
    action_registry.register_terminate_tool()

    agent_language = AgentFunctionCallingActionLanguage()
    environment = ActionContextEnvironment()
    
    return Agent(
        goals = goals,
        agent_language = agent_language,
        action_registry = action_registry,
        generate_response = llm_function,
        environment = environment,
        agent_name = "Project Manager Agent",
        max_iterations=20
    )

def create_developer_agent(llm_function) -> Agent:

    goals = [
        Goal(1, "Backend Development", 
             "Develop clean, functional backend APIs using best practices"),
        Goal(2, "Code Quality", 
             "Write maintainable, well-documented code with proper error handling"),
        Goal(3, "API Design", 
             "Create RESTful APIs with appropriate endpoints and response formats")
    ]
    
    # Tools for development and coding
    action_registry = DecoratorActionRegistry(tags=["expert", "file_operations", "coding"])
    action_registry.register_terminate_tool()
    
    agent_language = AgentFunctionCallingActionLanguage()
    environment = ActionContextEnvironment()
    
    return Agent(
        goals=goals,
        agent_language=agent_language,
        action_registry=action_registry,
        generate_response=llm_function,
        environment=environment,
        agent_name="Developer",
        max_iterations=10
    )

def create_code_reviewer_agent(llm_function) -> Agent:

    goals = [
        Goal(1, "Code Quality Review", 
             "Review code for quality, best practices, and potential issues"),
        Goal(2, "Security Analysis", 
             "Identify security vulnerabilities and suggest improvements"),
        Goal(3, "Performance Review", 
             "Analyze code for performance optimization opportunities")
    ]
    
    # Tools for review and quality assurance
    action_registry = DecoratorActionRegistry(tags=["expert", "code_review", "file_operations"])
    action_registry.register_terminate_tool()
    
    agent_language = AgentFunctionCallingActionLanguage()
    environment = ActionContextEnvironment()
    
    return Agent(
        goals=goals,
        agent_language=agent_language,
        action_registry=action_registry,
        generate_response=llm_function,
        environment=environment,
        agent_name="CodeReviewer",
        max_iterations=10
    )

def main():

    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY environment variable not set")
    models = [
        "gemini/gemini-1.5-flash",
        "gemini/gemini-2.0-flash",
        "gemini/gemini-2.0-flash-lite",
        "gemini/gemini-2.5-flash",
        "gemini/gemini-2.5-pro"
    ]
    llm_function = create_simple_llm_function(models[2])

    project_manager = create_project_manager_agent(llm_function)
    developer = create_developer_agent(llm_function)
    code_reviewer = create_code_reviewer_agent(llm_function)

    registry = AgentRegistry()


    registry.register_agent("project_manager", project_manager)
    registry.register_agent("developer", developer)
    registry.register_agent("code_reviewer", code_reviewer)

    shared_memory = Memory()

    # action_context = create_action_context_with_registry(
    #     registry=registry,
    #     llm_function=llm_function,
    #     memory=shared_memory,
    #     target_language="python"
    # )

    task = """
    Create a backend API in a folder called 'project' with the following requirements:
    1. A basic CRUD API for managing users
    2. Use Java with a web framework (Spring Boot)
    3. Implement endpoints for Create, Read, Update, Delete operations
    4. Use the MVC architecture, with separate models, views, and controllers
    5. Include proper error handling
    6. Add basic logging
    7. Make it production-ready with proper structure
    8. Include requirements.txt file
    9. Add a README.md with setup instructions

    The backend should be simple but follow best practices for a real application.
    """

    processor = DevEvalProcessor(lm_prompt_jsonl_path="C:/Users/cesar/7mo Semestre/DevEval/DevEval/Experiments/prompt/LM_prompt_elements.jsonl", mode='local_file_infiling', output_path='results')
    processor.process()

    # try:
    #     result_memory = project_manager.run(
    #         user_input=task,
    #         memory=shared_memory,
    #         action_context_props={
    #             "agent_registry": registry,
    #             "target_language": "Java",
    #             "project_type": "backend_api"
    #         })

    #     if result_memory.items:
    #         final_result = result_memory.items[-1]
    #         print("\nFinal Result: ", final_result.get("content", "No content"))
    #         print("\nFull Memory:")
    #         for item in result_memory.items:
    #             timestamp = item.get("timestamp", "N/A")
    #             role = item.get("role", "N/A")
    #             content = item.get("content", "N/A")
    #             print(f"[{timestamp}] ({role}): {content}\n")

    # except Exception as e:
    #     print(f"Error during agent execution: {str(e)}")
    #     traceback.print_exc()

if __name__ == "__main__":
    main()