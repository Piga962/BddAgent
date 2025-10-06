from game.actionContext import ActionContext
from game.tools import register_tool
from game.memory import Prompt
import json


@register_tool(tags=["expert", "consultation", "general"])
def prompt_expert(action_context: ActionContext, description_of_expert: str, prompt: str) -> str:
    generate_response = action_context.get("llm")
    if not generate_response:
        return "Error: LLM not available in action context."
    
    expert_response = generate_response(Prompt(
        messages=[
            {"role": "system", "content": f"Act as the following expert and respond accordingly: {description_of_expert}"},
            {"role": "user", "content": prompt}
        ]
    ))
    
    return expert_response

@register_tool(tags=["expert", "coding", "general"])
def consult_senior_developer(action_context: ActionContext, technical_requirements: str, language: str = "python") -> str:
    return prompt_expert(
        action_context=action_context,
        description_of_expert=f"""
You are a senior software developer with 10+ years of experience in {language} development.
You are expert in:
- Writing clean, mantainable, and efficient code.
- Software architecture and design patterns.
- Best practices for texting, debugging and deployment.
- Performance optimization and scalability considerations.
- Code reviews and mentoring junior developers.
""",
        prompt=f"Provide technical implementation guidance for: {technical_requirements}"
    )

@register_tool(tags=["expert", "review", "general"])
def consult_code_reviewer(action_context: ActionContext, code: str) -> str:
    return prompt_expert(
        action_context=action_context,
        description_of_expert="""
You are a senior code reviewer and technical lead with expertise in:
- Code quality asssessment and best practices enforcment
- Security vulnerability identification
- Performance optimization recommendations
- Maintainability and readability improvements
- Testing strategies and coverage analysis
""",
        prompt=f"Please review this code and provide detailed feedback:\n\n{code}"
    )

@register_tool(tags=["json", "llm"])
def prompt_llm_for_json(action_context: ActionContext, schema: dict, prompt: str) -> dict:
    """
    Have the LLM generate JSON according to a schema.
    Enhanced version with retry logic and better error handling.
    """
    generate_response = action_context.get("llm")
    if not generate_response:
        raise ValueError("No LLM function available in action context")

    
    for attempt in range(3):  # Try up to 3 times
        try:
            response = generate_response(Prompt(messages=[
                {
                    "role": "system",
                    "content": f"You MUST produce output that adheres to the following JSON schema:\n\n{json.dumps(schema, indent=2)}\n\nOutput your JSON in a ```json markdown block."
                },
                {"role": "user", "content": prompt}
            ]))

            # Extract JSON from markdown block
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.rfind("```")
                if end > start:
                    json_text = response[start:end].strip()
                else:
                    json_text = response[start:].strip()
            else:
                json_text = response.strip()

            return json.loads(json_text)
        
        except (json.JSONDecodeError, ValueError) as e:
            if attempt == 2:  # Last attempt
                raise e
            continue  # Try again