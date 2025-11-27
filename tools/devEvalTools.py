import re
from game.agent import Agent, ActionContext
from game.tools import register_tool    
from game.llms import Prompt
from tools.promptTools import prompt_llm_for_json

@register_tool(tags=["deveval", "analysis"])
def analyze_deveval_requirements(action_context: ActionContext, namespace: str, requirements: str, context: str = None) -> dict:

    generate_response = action_context.get('llm')
    context_info = f"\nRepository Context:\n{context}" if context else ""

    analysis_prompt = f"""
Analyze this DevEval function implementation task:

Namespace: {namespace}
Requirements: {requirements}{context_info}

Provide detailed analysis:
1. Function name (extract from namespace - it's the last part after the dot)
2. What does this function need to do?
3. Input parameters and their types (infer from requirements and context)
4. Expected return value and type
5. Required imports or dependencies from context
6. Edge cases and error handling needed
7. Implementation strategy (step by step)

If context is provided, identify:
- Helper functions that can be reused
- Existing classes or data structures
- Import patterns used in the codebase

Focus on the SIMPLEST approach using existing code patterns.
"""
    
    return prompt_llm_for_json(
        action_context=action_context,
        schema={
            "type": "object",
            "properties": {
                "function_name": {"type": "string"},
                "function_purpose": {"type": "string"},
                "input_parameters": {"type": "array", "items": {"type": "string"}},
                "return_value": {"type": "string"},
                "dependencies": {"type": "array", "items": {"type": "string"}},
                "error_handling": {"type": "string"},
                "implementation_strategy": {"type": "string"},
                "reusable_code": {"type": "string"}
            },
            "required": ["function_name", "function_purpose", "implementation_strategy"]
        },
        prompt=analysis_prompt
    )

@register_tool(tags=["deveval", "coding"])
def generate_complete_function(action_context: ActionContext, analysis: str, requirements: str, context: str = None) -> str:
    generate_response = action_context.get('llm')
    context_info = f"\nRepository Context:\n{context}" if context else ""

    prompt = f"""
Generate a complete, working Python function based on this analysis.

Analysis: {analysis}
Requirements: {requirements}{context_info}

CRITICAL RULES:
1. Include the complete function definition starting with 'def function_name(...):'
2. Use exactly 4 spaces for indentation throughout
3. If imports are needed, include them INSIDE the function at the top
4. Handle all edge cases mentioned in the analysis
5. Ensure the function actually works and returns correct values
6. Use patterns and helper functions from context if provided

IMPORTANT:
- Generate ONLY the function code
- No markdown code blocks
- No explanations or comments
- No docstrings
- Just working Python code

Example format:
def function_name(param1, param2):
    import necessary_module
    
    if edge_case:
        return edge_value
    
    result = main_computation()
    return result

Generate the complete function now:
"""
    
    response = generate_response(Prompt(messages=[
        {"role": "system", "content": "You are a Python developer. Generate complete, working functions with proper signatures and implementations."},
        {"role": "user", "content": prompt}
    ]))
    print(response)
    return response

@register_tool(tags=["deveval", "bdd"])
def generate_bdd_tests(action_context: ActionContext, namespace: str, analysis: str) -> str:
    generate_response = action_context.get('llm')
    prompt = f"""
Generate BDD-style scenarios for testing the DevEval function.

Namespace: {namespace}
Analysis: {analysis}

Provide all the necessary BDD scenarios to fully test the function's behavior.
CRITICAL:
- Use Gherkin syntax with Given, When, Then
- Cover normal cases, edge cases, and error handling
- No additional explanations, just the scenarios
Example format:
Feature: Functionality of {namespace}

  Scenario: Description of scenario
    Given some initial context
    When an action is performed
    Then expect a specific outcome

"""
    response = generate_response(Prompt(messages=[
        {"role": "system", "content": "You are a BDD test case generator. Produce clear Gherkin scenarios."},
        {"role": "user", "content": prompt}
    ]))
    print(response)
    return response


@register_tool(tags=["deveval", "review"])
def review_deveval_code(actioncontext: ActionContext, code: str, requirements: str, namespace: str) -> dict:
    return prompt_llm_for_json(
        action_context=actioncontext,
        schema={
            "type": "object",
            "properties": {
                "is_correct": {"type": "boolean"},
                "identation_ok": {"type": "boolean"},
                "meet_requirements": {"type": "boolean"},
                "issues": {"type": "array", "items": {"type": "string"}},
                "improved_code": {"type": "string"},
                "confidence": {"type": "number"}
            }
        },
        prompt=f"""
Review this DevEval function implementation:

Namespace: {namespace}
Requirements: {requirements}
Generated Code:
{code}

Check:
1. Does it meet all requirements?
2. Are the parameters and return types and names the same as the requirements?
2. Is identation exactly 4 spaces?
3. Is it only the function body (no signature)?
4. Are there any syntax errors?
5. Does it handle edge cases?

CRITICAL: Return ONLY the JSON response. No additional text or explanations.

If issues found, provide improved_code with fixes.

"""
)

@register_tool(tags=["deveval", "extraction"])
def extract_clean_code(action_context: ActionContext, raw_output: str) -> str:
    """Extract function body ONLY (no def line, no docstrings)."""
    
    # Remove markdown
    code = re.sub(r"```python\s*\n(.*?)\n```", r"\1", raw_output, flags=re.DOTALL)
    code = re.sub(r'```\s*\n(.*?)\n```', r'\1', code, flags=re.DOTALL)

    lines = code.split("\n")
    cleaned_lines = []
    inside_function = False
    function_indent = 0
    in_docstring = False
    docstring_char = None

    for line in lines:
        stripped = line.strip()
        current_indent = len(line) - len(line.lstrip())
        
        # Find function definition
        if not inside_function and stripped.startswith('def ') and '(' in stripped and ':' in stripped:
            inside_function = True
            function_indent = current_indent
            continue  # Skip the def line
            
        # Process function body
        if inside_function:
            # Handle docstrings
            if not in_docstring:
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    docstring_char = '"""' if stripped.startswith('"""') else "'''"
                    # Check if single-line docstring
                    if stripped.count(docstring_char) >= 2:
                        continue  # Skip single-line docstring
                    else:
                        in_docstring = True
                        continue
            else:
                # We're inside a docstring, look for the end
                if docstring_char and docstring_char in line:
                    in_docstring = False
                    docstring_char = None
                continue
            
            # Check if we've left the function (new function or unindented code)
            if stripped and current_indent <= function_indent:
                if stripped.startswith('def '):
                    break  # New function found
                # If it's other code at same/lower indent, might be end of function
                if not stripped.startswith('#'):
                    break
            
            # Add the line with proper indentation
            if stripped:  # Non-empty line
                # Calculate relative indent from function body
                relative_indent = current_indent - function_indent
                # Normalize to 4-space base
                new_indent = max(4, 4 + (relative_indent // 4) * 4)
                cleaned_lines.append(' ' * new_indent + stripped)
            else:
                # Keep empty lines within function
                cleaned_lines.append('')
    
    # Remove trailing empty lines
    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()
    
    result = '\n'.join(cleaned_lines)
    
    # Validation
    if not result.strip() or len(result.strip()) < 5:
        return "    pass  # No implementation extracted"
    
    # Ensure minimum indentation is 4 spaces
    result_lines = result.split('\n')
    final_lines = []
    for line in result_lines:
        if line.strip():  # Non-empty
            current_indent = len(line) - len(line.lstrip())
            if current_indent < 4:
                final_lines.append('    ' + line.strip())
            else:
                final_lines.append(line)
        else:
            final_lines.append('')
    
    return '\n'.join(final_lines)