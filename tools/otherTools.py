from game.tools import register_tool

@register_tool(tags=["general"], terminal=True)
def terminate(message: str = "Task completed successfully") -> str:
    """
    Terminate the agent session with a summary message.
    
    Args:
        message: Final message to display
    
    Returns:
        Termination message
    """
    return f"{message}\n\n🎉 Agent session completed."