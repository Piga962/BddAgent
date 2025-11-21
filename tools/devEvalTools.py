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
1. What does this function need to do?
2. What are the input parameters and their types?
3. What should be returned?
4. Are there any dependencies or imports needed?
5. What error cases should be handled?
6. What patterns do you see in the context code?
7. What is the function name?

Responde with structured analysis.
"""
    
    return prompt_llm_for_json(
        action_context=action_context,
        schema={
            "type": "object",
            "properties": {
                "function_purpose": {"type": "string"},
                "input_parameters": {"type": "array", "items": {"type": "string"}},
                "return_value": {"type": "string"},
                "dependencies": {"type": "array", "items": {"type": "string"}},
                "error_handling": {"type": "string"},
                "implementation_strategy": {"type": "string"}
            }
        },
        prompt=analysis_prompt
    )

@register_tool(tags=["deveval", "coding"])
def generate_complete_function(action_context: ActionContext, analysis: str, requirements: str, context: str = None) -> str:
    generate_response = action_context.get('llm')
    context_info = f"\nRepository Context:\n{context}" if context else ""

    prompt = f"""
Analysis: {analysis}

Based on this analysis, implement the complete function:
Requirements: {requirements}

{context_info}

Generate a complete, working Python function with:
1. Proper function signature
2. Complete implementation
3. Proper error handling if needed
4. Return appropriate values
5. No explanations, no comments

IMPORTANT:
If you need to create helper functions, include them within the function, also if you need to add imports, include them at the top within the function. DO NOT generate code outside the function.

Example format:
```python
def function_name(params):
    import necessary_module
    def helper_function():
        pass
    # implementation here
    if condition:
        return value
    return default_value
```
Generate the complete function now:
"""
    
    response = generate_response(Prompt(messages=[
        {"role": "system", "content": "You are a Python developer. Generate complete, working functions with proper signatures and implementations."},
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
2. Is identation exactly 4 spaces?
3. Is it only the function body (no signature)?
4. Are there any syntax errors?
5. Does it handle edge cases?

If issues found, provide improved_code with fixes.

"""
)

@register_tool(tags=["deveval", "extraction"])
def extract_clean_code(action_context: ActionContext, raw_output: str) -> str:

    code = re.sub(r"```python\s*\n(.*?)\n```", r"\1", raw_output, flags=re.DOTALL)
    code = re.sub(r'```\s*\n(.*?)\n```', r'\1', code, flags=re.DOTALL)

    lines = code.split("\n")
    cleaned_lines = []
    inside_function = False
    function_indent = 0

    for line in lines:
        stripped = line.strip()
        
        # Encontrar la línea de definición de función
        if stripped.startswith('def ') and '(' in stripped and ':' in stripped:
            inside_function = True
            function_indent = len(line) - len(line.lstrip())
            continue
            
        # Si estamos dentro de la función
        if inside_function:
            current_indent = len(line) - len(line.lstrip())
            
            # Si encontramos una línea con indentación igual o menor que la función, terminamos
            if stripped and current_indent <= function_indent:
                break
                
            # Agregar línea del cuerpo (ajustando indentación a 4 espacios)
            if stripped:  # Línea con contenido
                body_indent = current_indent - function_indent
                new_indent = max(4, body_indent)  # Mínimo 4 espacios
                cleaned_lines.append(' ' * new_indent + stripped)
            else:  # Línea vacía
                cleaned_lines.append('')
    
    # Limpiar líneas vacías al final
    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()
    
    result = '\n'.join(cleaned_lines)
    
    if not result.strip():
        return "    pass  # No implementation extracted"
        
    return result

@register_tool(tags=["deveval", "coding"])
def generate_deveval_function_body(action_context: ActionContext,
                                  namespace: str,
                                  requirements: str,
                                  analysis: str = None,
                                  context: str = None) -> str:
    """Generar SOLO el cuerpo de la función Python."""
    
    generate_response = action_context.get('llm')
    
    context_info = f"\nRepository Context:\n{context[:1000]}" if context else ""
    analysis_info = f"\nAnalysis:\n{analysis}" if analysis else ""
    
    prompt = f"""
You must generate ONLY Python function body code for DevEval.

Namespace: {namespace}
Requirements: {requirements}{context_info}{analysis_info}

CRITICAL INSTRUCTIONS:
1. Write ONLY the function body (no 'def' line)
2. Use exactly 4 spaces for indentation
3. Generate actual working Python code
4. No explanations, no comments, no markdown
5. No docstrings, no descriptions
6. Return actual implementation code

Example for "check if number is even":
    if x % 2 == 0:
        return True
    return False

Now generate ONLY the function body for the requirements above:
"""
    
    response = generate_response(Prompt(messages=[
        {"role": "system", "content": "You are a Python code generator. Generate ONLY function bodies with 4-space indentation. No explanations."},
        {"role": "user", "content": prompt}
    ]))
    
    return response
