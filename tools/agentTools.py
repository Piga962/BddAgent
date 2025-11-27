from game.memory import Memory
from game.actionContext import ActionContext
from game.tools import register_tool
from tools.promptTools import prompt_llm_for_json


@register_tool(tags=["agent", "coordination"])
def call_agent(action_context: ActionContext, agent_name: str, task: str) -> str:
    
    agent_registry = action_context.get("agent_registry")
    if not agent_registry:
        raise ValueError("No agent registry found in action context.")
    
    agent_run = agent_registry.get_agent(agent_name)
    if not agent_run:
        raise ValueError(f"Agent '{agent_name}' not found in registry. Available: {agent_registry.list_agents()}")
    
    invoked_memory = Memory()

    try:
        result_memory = agent_run(
            user_input=task,
            memory=invoked_memory,
        )

        if result_memory.items:
            last_memory = result_memory.items[-1]
            return {
                "success": True,
                "agent": agent_name,
                "result": last_memory.get("content", "No content in last memory item."),
                "memory_items": len(result_memory.items)
            }
        else:
            return {
                "success": False,
                "result": "Agent completed but produced no output"
            }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
    
@register_tool(tags=["agent", "coordination", "reflection"])
def call_agent_with_reflection(action_context: ActionContext, agent_name: str, task: str) -> dict:
    agent_registry = action_context.get("agent_registry")
    if not agent_registry:
        raise ValueError("No agent registry found in action context.")
    
    agent_run = agent_registry.get_agent(agent_name)
    if not agent_run:
        raise ValueError(f"Agent '{agent_name}' not found in registry.")
    
    invoked_memory = Memory()

    result_memory = agent_run(
        user_input=task,
        memory=invoked_memory
    )

    caller_memory = action_context.get("memory")

    memories_added = 0
    if caller_memory:
        for memory_item in result_memory.items:
            caller_memory.add_memory({
                "type": f"{agent_name}_thought",
                "content": memory_item.get("content", ""),
                "timestamp": memory_item.get("timestamp"),
                "agent_source": agent_name
            })
            memories_added += 1
    
    return {
        "success": True,
        "agent": agent_name,
        "result": result_memory.items[-1].get("content", "No content in last memory item.") if result_memory.items else "No output",
        "memories_added": memories_added,
        "reasoning_shared": True
    }

@register_tool(tags=["agent", "coordination", "handoff"])
def hand_off_to_agent(action_context: ActionContext, agent_name: str, task: str) -> dict:
    agent_registry = action_context.get("agent_registry")
    if not agent_registry:
        raise ValueError("No agent registry found in action context.")
    
    agent_run = agent_registry.get_agent(agent_name)
    if not agent_run:
        raise ValueError(f"Agent '{agent_name}' not found in registry.")

    current_memory = action_context.get("memory")

    result_memory = agent_run(
        user_input=task,
        memory=current_memory
    )

    return {
        "success": True,
        "agent": agent_name,
        "result": result_memory.items[-1].get("content", "No result") if result_memory.items else "No output",
        "memory_id": id(result_memory),
        "shared_memory": True,
        "memory_items": len(result_memory.items) if current_memory else 0
    }

@register_tool(tags=["agent", "coordination", "selective"])
def call_agent_with_selected_context(action_context: ActionContext, 
                                    agent_name: str, 
                                    task: str) -> dict:
    agent_registry = action_context.get("agent_registry")
    if not agent_registry:
        raise ValueError("No agent registry found in context")
        
    agent_run = agent_registry.get_agent(agent_name)
    if not agent_run:
        raise ValueError(f"Agent '{agent_name}' not found in registry")

    current_memory = action_context.get("memory")
    if not current_memory or not current_memory.items:
        # No memory to select from, use regular call
        return call_agent(action_context, agent_name, task)
    
    # Add IDs to memories for selection
    memory_with_ids = []
    for idx, item in enumerate(current_memory.items):
        memory_with_ids.append({
            **item,
            "memory_id": f"mem_{idx}"
        })

    # Schema for memory selection
    selection_schema = {
        "type": "object",
        "properties": {
            "selected_memories": {
                "type": "array",
                "items": {
                    "type": "string",
                    "description": "ID of a memory to include"
                }
            },
            "reasoning": {
                "type": "string",
                "description": "Explanation of why these memories were selected"
            }
        },
        "required": ["selected_memories", "reasoning"]
    }

    # Create memory summary for selection
    memory_text = "\n".join([
        f"Memory {m['memory_id']}: {m['content'][:100]}..." if len(m.get('content', '')) > 100 else f"Memory {m['memory_id']}: {m.get('content', '')}"
        for m in memory_with_ids
    ])

    selection_prompt = f"""Review these memories and select the ones relevant for this task:

Task: {task}
Agent to call: {agent_name}

Available Memories:
{memory_text}

Select memories that provide important context, requirements, constraints, or background information for this specific task. Focus on relevance and avoid redundant information.
"""
    
    # Use the LLM to select relevant memories
    selection = prompt_llm_for_json(
        action_context=action_context,
        schema=selection_schema,
        prompt=selection_prompt
    )

    # Create filtered memory with selected items
    filtered_memory = Memory()
    selected_ids = set(selection["selected_memories"])
    for item in memory_with_ids:
        if item["memory_id"] in selected_ids:
            item_copy = item.copy()
            del item_copy["memory_id"]
            filtered_memory.add_memory(item_copy)

    # Run agent with filtered memory
    result_memory = agent_run(
        user_input=task,
        memory=filtered_memory
    )

    # Add selection reasoning to current memory
    current_memory.add_memory({
        "type": "system",
        "content": f"Memory selection for {agent_name}: {selection['reasoning']}",
        "agent_source": "memory_selector"
    })

    # Add results back to current memory
    for memory_item in result_memory.items:
        current_memory.add_memory({
            **memory_item,
            "agent_source": agent_name
        })

    return {
        "success": True,
        "agent": agent_name,
        "result": result_memory.items[-1].get("content", "No result") if result_memory.items else "No output",
        "shared_memories": len(filtered_memory.items),
        "selection_reasoning": selection["reasoning"],
        "total_memories_available": len(memory_with_ids),
        "optimization": "memory_selective"
    }