import json
from typing import List, Dict
import os
import sys
import re
import traceback

from game.actionContext import create_action_context_with_registry
from game.actions import DecoratorActionRegistry
from game.agent import Agent, AgentRegistry
from game.environment import ActionContextEnvironment
from game.llms import create_simple_llm_function
from game.memory import Goal, Memory
from game.agentLanguage import AgentFunctionCallingActionLanguage

import tools.agentTools, tools.fileTools, tools.promptTools, tools.otherTools, tools.devEvalTools


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
        return tests

    def generate_jsonl(self):
        out_file = os.path.join(self.output_path, self.mode + '_results24.jsonl')
        with open(out_file, 'w', encoding='utf-8') as f:
            for result in self.results:
                json_line = json.dumps(result)
                f.write(json_line + '\n')
        print(f"Generated JSONL file at {out_file}")

    def create_deveval_task(self, test, mode):
        namespace = test['namespace']
        requirements = test['input_code']
        
        context_str = ""
        if mode == 'local_file_completion':
            context_str = f"Context: {test['context_above']}"
        elif mode == 'local_file_infiling':
            context_str = f"Context Above: {test['context_above']}\nContext Below: {test['context_below']}"
        
        base_task = f"""
    DEVEVAL TASK: {namespace}
    Requirements: {requirements}
    {context_str}

    MANDATORY WORKFLOW:
    1. Analyze requirements and context to produce an analysis_result (you can use tools if needed)
    2. Use analysis_result to generate the complete function.
    3. Extract ONLY the function body (no 'def' line, no signature) from the generated function.
    4. Ensure the function body uses exactly 4 spaces for indentation.

    CRITICAL: Generate a complete working Python function, the extract only the boy.
    """
        return base_task

    def extract_final_code_from_memory(self, memory, namespace=None):
        """Extraer código con debugging."""

        def extract_function_body_from_complete(text, is_from_json=False):
            """Extracción que maneja correctamente funciones helper internas."""
            print("\n🔍 DEBUGGING EXTRACTION - RAW TEXT:")
            print(text[:300] + "...")

            # Limpiar markdown
            text = re.sub(r'```python\s*\n(.*?)\n```', r'\1', text, flags=re.DOTALL)
            text = re.sub(r'```\s*\n(.*?)\n```', r'\1', text, flags=re.DOTALL)
            
            lines = text.split('\n')
            body_lines = []
            found_main_def = False
            main_function_indent = 0
            
            for line in lines:
                stripped = line.strip()
                current_indent = len(line) - len(line.lstrip())
                
                # Si es JSON, empezar desde la primera línea (sin buscar def)
                if is_from_json:
                    if not found_main_def and stripped.startswith('def ') and ':' in stripped:
                        found_main_def = True
                        main_function_indent = current_indent  # Asumir que ya está limpio
                        continue
                    
                    if stripped:
                        if current_indent <= main_function_indent:
                            body_lines.append('    ' + stripped)
                        else:
                            relative_indent = current_indent - main_function_indent
                            new_indent = 4 + relative_indent
                            body_lines.append(' ' * new_indent + stripped)
                    else:
                        body_lines.append('')
                else:
                    # Si es texto, buscar y saltar la línea def
                    if not found_main_def and stripped.startswith('def ') and ':' in stripped:
                        found_main_def = True
                        main_function_indent = current_indent
                        continue  # SALTAR la línea def
                    
                    if found_main_def:
                        if stripped:
                            if current_indent <= main_function_indent:
                                body_lines.append('    ' + stripped)
                            else:
                                relative_indent = current_indent - main_function_indent
                                new_indent = 4 + relative_indent
                                body_lines.append(' ' * new_indent + stripped)
                        else:
                            body_lines.append('')
            
            while body_lines and not body_lines[-1].strip():
                body_lines.pop()
            
            result = '\n'.join(body_lines) if body_lines else None
    
            if result:
                # Remover texto de sesión completada
                result = re.sub(r'\s*🎉.*?Agent session completed.*$', '', result, flags=re.DOTALL)
                result = result.rstrip()
            
            return result
        
        print(f"\n🔍 DEBUGGING EXTRACTION for {namespace}:")
        
        for item in reversed(memory.items):
            content = str(item.get("content", ""))
            
            # Buscar en JSON result
            try:
                data = json.loads(content)
                if "result" in data:
                    result_text = str(data["result"])
                    if "def " in result_text:
                        print(f"📦 Found function in JSON result")
                        function_body = extract_function_body_from_complete(result_text, is_from_json=True)
                        if function_body and len(function_body.strip()) > 10:
                            print(f"✅ Extracted from JSON result")
                            return function_body
            except json.JSONDecodeError:
                pass
            
            # Buscar en texto plano
            if "def " in content and len(content) > 50:
                print(f"📝 Trying text content")
                function_body = extract_function_body_from_complete(content, is_from_json=False)
                if function_body and len(function_body.strip()) > 10:
                    print(f"✅ Extracted from text")
                    return function_body
        
        return "    pass  # No implementation found"

    def clean_extracted_code(self, raw_code):
        """Limpiar código extraído para DevEval."""
        
        lines = raw_code.split('\n')
        cleaned_lines = []
        skip_next = False
        
        for line in lines:
            # Saltar líneas de documentación y comentarios largos
            if skip_next:
                if '"""' in line or "'''" in line:
                    skip_next = False
                continue
                
            stripped = line.strip()
            
            # Saltar docstrings
            if '"""' in line or "'''" in line:
                if line.count('"""') == 1 or line.count("'''") == 1:
                    skip_next = True
                continue
            
            # Saltar signature de función
            if stripped.startswith('def ') and '(' in stripped and ':' in stripped:
                continue
                
            # Saltar líneas vacías o comentarios explicativos
            if not stripped or \
            stripped.startswith('# Note:') or \
            stripped.startswith('# This') or \
            any(skip_phrase in stripped.lower() for skip_phrase in ['here is', 'this completes']):
                continue
            
            # Limpiar indentación - asegurar 4 espacios mínimo
            if stripped:
                current_indent = len(line) - len(line.lstrip())
                if current_indent == 0:
                    # Agregar indentación base de 4 espacios
                    cleaned_lines.append("    " + stripped)
                else:
                    # Normalizar indentación a múltiplos de 4, mínimo 4
                    new_indent = max(4, ((current_indent + 3) // 4) * 4)
                    cleaned_lines.append(" " * new_indent + stripped)
        
        # Remover líneas vacías al final
        while cleaned_lines and not cleaned_lines[-1].strip():
            cleaned_lines.pop()
        
        result = '\n'.join(cleaned_lines)
        
        # Validar que tenemos código útil
        if not result.strip() or len(result.split('\n')) < 2:
            return "    pass"
        
        return result

    def process(self):
        llm_function = create_simple_llm_function("azure/gpt-4.1-mini")
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

                mainAgent = Agent(
                    goals=[
                        Goal(1, "DevEval Analysis","Use tools to analyze DevEval requirements thoroughly"),
                        Goal(2, "Agent Coordination", "Delegate to coding agent and coordinate review process"),
                        Goal(3, "Quality Assurance", "Ensure code meets DevEval stgandards through review tools")
                    ],
                    agent_language=AgentFunctionCallingActionLanguage(),
                    action_registry=DecoratorActionRegistry(tags=[ "agent", "deveval"]),
                    generate_response=llm_function,
                    environment=ActionContextEnvironment(),
                    agent_name="DevEvalMainAgent",
                    max_iterations=12
                )
                codingAgent = Agent(
                    goals = [
                        Goal(1, "Complete Function Generation", "Generate complete Python functions with proper signatures"),
                        Goal(2, "Working Implementation", "Write actual working Python code that solves the given requirements"),
                    ],
                    agent_language=AgentFunctionCallingActionLanguage(),
                    action_registry=DecoratorActionRegistry(tags=[ "deveval"]),
                    generate_response=llm_function,
                    environment=ActionContextEnvironment(),
                    agent_name="DevEvalCodingAgent",
                    max_iterations=8
                )
                codeReviewer = create_code_reviewer_agent(llm_function)

                registry.register_agent("coding_agent", codingAgent.run)
                registry.register_agent("main_agent", mainAgent.run)
                registry.register_agent("code_reviewer", codeReviewer.run)
                mainAgent.action_registry.register_terminate_tool()
                codingAgent.action_registry.register_terminate_tool()
                codeReviewer.action_registry.register_terminate_tool()

                task = self.create_deveval_task(test, self.mode)

                action_context = {
                    "agent_registry": registry,
                    "target_language": "python",
                    "project_type": "deveval_function",
                    "namespace": namespace
                }
#                 if self.mode == 'without_context':
#                     task = f"""
# Given this requirements, implement ONLY the function body (without signature):
# Requirements:
# {requirement}

# Return only the function body without any additional explanations, and with proper indentation (4 spaces).
# """
#                 elif self.mode == 'local_file_completion':
#                     task = f"""
# Given this context and requirements, implement ONLY the function body (without signature):
# Context:
# {context_above}

# Requirements:
# {requirement}

# Return ONLY the function body without any additional explanations, and with proper indentation (4 spaces).
# """
#                 elif self.mode == 'local_file_infiling':
#                     task = f"""
# Given this context and requirements, implement ONLY the function body (without signature):
# Context above:
# {context_above}

# Context below:
# {context_below}

# Requirements:
# {requirement}

# Return ONLY the function body without any additional explanations, with proper indentation (4 spaces).
# """

                result_memory = mainAgent.run(
                    user_input=task,
                    memory=sharedMemory,
                    action_context_props=action_context
                )
                final_code = self.extract_final_code_from_memory(result_memory, namespace=namespace)

                print(f"\n🛠️ Cleaned Final Code for {namespace}:\n{final_code}\n")


                    # print(final_result)
                    # print(f"\nFinal Result for test {namespace}: ", final_result.get("content", "No content"))
                self.results.append({
                    "namespace": namespace,
                    "completion": final_code
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
    action_registry = DecoratorActionRegistry(tags=[ "deveval"])
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

def dump_memory_jsonl(memory, out_dir: str = "results", filename: str = "final_memory.jsonl"):
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, filename)
    with open(out_file, "w", encoding="utf-8") as fh:
        for item in getattr(memory, "items", []):
            # Use default=str to serialize non-JSON types (timestamps, etc.)
            fh.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
    print(f"Saved memory JSONL to {out_file}")
    return out_file

def main():

    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY environment variable not set")
    models = [
        "gemini/gemini-1.5-flash",
        "gemini/gemini-2.0-flash",
        "gemini/gemini-2.0-flash-lite",
        "gemini/gemini-2.5-flash",
        "gemini/gemini-2.5-pro",
        "azure/gpt-4.1-mini"
    ]
    llm_function = create_simple_llm_function(models[5])

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
    Create a backend API in a folder called 'backend' with the following requirements:
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
    mode = 'without_context'
    if len(sys.argv) >1:
        mode = sys.argv[1]

    processor = DevEvalProcessor(lm_prompt_jsonl_path="C:/Users/cesar/7mo Semestre/DevEval/DevEval/Experiments/prompt/LM_prompt_elements.jsonl", mode='local_file_infiling', output_path='results')
    # processor = DevEvalProcessor(lm_prompt_jsonl_path="/home/piga/BddAgent/data/LM_prompt_elements.jsonl", mode=mode, output_path='results')
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
    #         # print("\nFull Memory:")
    #         # for item in result_memory.items:
    #         #     timestamp = item.get("timestamp", "N/A")
    #         #     role = item.get("role", "N/A")
    #         #     content = item.get("content", "N/A")
    #         #     print(f"[{timestamp}] ({role}): {content}\n")
    #     dump_memory_jsonl(result_memory, out_dir="results", filename="project_manager_memory.jsonl")

    # except Exception as e:
    #     print(f"Error during agent execution: {str(e)}")
    #     traceback.print_exc()

if __name__ == "__main__":
    main()