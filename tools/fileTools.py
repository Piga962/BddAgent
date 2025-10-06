from game.tools import register_tool
from game.actionContext import ActionContext
import os

#Tools = tools
@register_tool(tags=["file_operations", "general"])
def list_files(action_context: ActionContext, directory: str=".") -> str:
    try:
        files = []
        for item in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, item)):
                files.append(item)

        memory = action_context.get("memory")
        if memory:
            memory.add_memory({
                "role": "system",
                "content": f"Directory '{directory}' contains {len(files)} files: {', '.join(files[:10])}{'...' if len(files) > 10 else ''}",
                "operation": "list_files"
            })
        
        return files
    except Exception as e:
        return [f"Error listing files: {str(e)}"]
    
@register_tool(tags=["file_operations", "general"])
def read_file(action_context: ActionContext, file_name: str) -> str:
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            content = file.read()
        
        memory = action_context.get("memory")
        if memory:
            memory.add_memory({
                "role": "system",
                "content": f"Read file '{file_name}' ({len(content)} characters)",
                "operation": "read_file",
                "file_name": file_name
            })
        return content
    
    except FileNotFoundError:
        return f"Error: File '{file_name}' not found."
    except Exception as e:
        return f"Error reading file '{file_name}': {str(e)}"

@register_tool(tags=["file_operations", "general"])
def write_file(action_context: ActionContext, file_name: str, content: str) -> str:
    try:
        with open(file_name, 'w', encoding='utf-8') as file:
            file.write(content)

        memory = action_context.get("memory")
        if memory:
            memory.add_memory({
                "role": "system",
                "content": f"Created/updated file '{file_name}' with {len(content)} characters",
                "operation": "write_file",
                "file_name": file_name
            })

        return f"File '{file_name}' written successfully."
    except Exception as e:
        return f"Error writing to file '{file_name}': {str(e)}"
    
@register_tool(tags=["file_operations", "general"])
def append_to_file(action_context: ActionContext, file_name: str, content: str) -> str:
    try:
        with open(file_name, 'a', encoding='utf-8') as file:
            file.write(content)
        return f"Content appended to file '{file_name}' successfully."
    except Exception as e:
        return f"Error appending to file '{file_name}': {str(e)}"

@register_tool(tags=["file_operations", "general"])
def delete_file(action_context: ActionContext, file_name: str) -> str:
    try:
        os.remove(file_name)
        return f"File '{file_name}' deleted successfully."
    except FileNotFoundError:
        return f"Error: File '{file_name}' not found."
    except Exception as e:
        return f"Error deleting file '{file_name}': {str(e)}"
    
@register_tool(tags=["file_operations", "general"])
def create_directory(action_context: ActionContext, directory_name: str) -> str:
    try:
        os.makedirs(directory_name, exist_ok=True)
        return f"Directory '{directory_name}' created successfully."
    except Exception as e:
        return f"Error creating directory '{directory_name}': {str(e)}"