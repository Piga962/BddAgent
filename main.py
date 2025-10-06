from typing import List, Dict
import os
import traceback

from game.actionContext import create_action_context_with_registry
from game.actions import DecoratorActionRegistry
from game.agent import Agent, AgentRegistry
from game.environment import ActionContextEnvironment
from game.llms import create_simple_llm_function
from game.memory import Goal, Memory
from game.agentLanguage import AgentFunctionCallingActionLanguage

import tools.agentTools, tools.fileTools, tools.promptTools, tools.otherTools

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

    action_context = create_action_context_with_registry(
        registry=registry,
        llm_function=llm_function,
        memory=shared_memory,
        target_language="python"
    )

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
    
    try:
        result_memory = project_manager.run(
            user_input=task,
            memory=shared_memory,
            action_context_props={
                "agent_registry": registry,
                "target_language": "Java",
                "project_type": "backend_api"
            })

        if result_memory.items:
            final_result = result_memory.items[-1]
            print("\nFinal Result: ", final_result.get("content", "No content"))
            print("\nFull Memory:")
            for item in result_memory.items:
                timestamp = item.get("timestamp", "N/A")
                role = item.get("role", "N/A")
                content = item.get("content", "N/A")
                print(f"[{timestamp}] ({role}): {content}\n")

    except Exception as e:
        print(f"Error during agent execution: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()