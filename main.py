import json
from typing import List, Dict
import os
import sys
import re
import traceback
import torch

from game.actionContext import create_action_context_with_registry, ActionContext
from game.actions import DecoratorActionRegistry
from game.agent import Agent, AgentRegistry
from game.environment import ActionContextEnvironment
from game.llms import create_simple_llm_function, create_qwen_llm_function
from game.memory import Goal, Memory
from game.agentLanguage import AgentFunctionCallingActionLanguage, AgentJSONActionLanguage

import tools.agentTools, tools.fileTools, tools.promptTools, tools.otherTools, tools.tritonTools
from tools.tritonTools import execute_pytorch_code, execute_triton_code, validate_outputs

class PyTorchToTritonProcessor:
    def __init__(self, llm_function, input_path: str, output_path: str = "triton_kernels.jsonl"):
        self.llm_function = llm_function
        self.input_path = input_path
        self.output_path = output_path
        self.pytorch_code_samples = self.load_pytorch_samples()

    def load_pytorch_samples(self) -> List[Dict]:
        samples = []
        with open(self.input_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    sample = json.loads(line)
                    pytorch_code = sample['code']
                    newJSON = {
                        "pytorch_code": pytorch_code
                    }
                    samples.append(newJSON)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON line: {line}\nError: {str(e)}")
        print(samples)
        return samples
      
    def extract_triton_from_memory(self, memory):
        for item in reversed(memory.items):
            content = str(item.get("content", ""))
            
            # Buscar en contenido que tenga código Triton
            if "@triton.jit" in content or "triton.language" in content:
                # Limpiar JSON escapado si es necesario
                try:
                    data = json.loads(content)
                    if "result" in data:
                        content = str(data["result"])
                except json.JSONDecodeError:
                    pass
                
                # Extraer el código entre comillas si está escapado
                content = content.replace("\\n", "\n").replace('\\"', '"')
                
                # Encontrar desde el import hasta el final del código
                start = content.find("import torch")
                if start == -1:
                    start = content.find("import triton")
                if start != -1:
                    code = content[start:]
                    cut = code.find('{"tool_name"')
                    if cut != -1:
                        code = code[:cut]
                    for marker in ['# Example usage', '# Run the example', 'if __name__', 
                                  '\nA = torch.tensor', '\nA = torch.randn', 
                                  '# Kernel and launcher', '\n🎉', '\\ud83c',
                                  '```\\"', '```json', '\n```\n', '"}\n}']:
                        cut2 = code.find(marker)
                        if cut2 != -1:
                            code = code[:cut2]
                    
                    # Limpiar caracteres de escape JSON
                    code = code.replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')
                    
                    return code.strip()
        
        return None
    
    def create_triton_task(self, pytorch_code: str) -> str:
        task = f"""
    Complete these 4 steps in order. Do not skip any step.

    1. get_gpu_specs
    2. call_agent_with_selected_context → "TritonDeveloper" → generate a Triton kernel for this PyTorch code:
    {pytorch_code}
    3. call_agent_with_selected_context → "TritonReviewer" → review the generated kernel for correctness
    4. terminate with the final reviewed Triton kernel code as the message

    Do not execute any code. Do not call validate_outputs or save_to_dataset. Just generate, review, and terminate.
    """
        return task

    def process(self):
        for idx, sample in enumerate(self.pytorch_code_samples):
            try:
                pytorch_code = sample['pytorch_code']
                
                sharedMemory = Memory()
                registry = AgentRegistry()

                mainAgent = create_project_manager_agent(self.llm_function)
                codingAgent = create_developer_agent(self.llm_function)
                codeReviewer = create_code_reviewer_agent(self.llm_function)

                registry.register_agent("TritonDeveloper", codingAgent.run)
                registry.register_agent("TritonReviewer", codeReviewer.run)

                task = self.create_triton_task(pytorch_code)

                action_context = {
                    "agent_registry": registry,
                    "target_language": "python",
                    "project_type": "triton_kernel",
                    "shared_memory": sharedMemory,
                    "llm": self.llm_function,
                    "dataset_path": self.output_path,
                    "seed": idx
                }

                result_memory = mainAgent.run(
                    user_input=task,
                    memory=sharedMemory,
                    action_context_props=action_context)

                triton_code = self.extract_triton_from_memory(sharedMemory)

                gpu_specs = None
                pytorch_execution = None
                triton_execution = None
                validation = None


                for item in sharedMemory.items:
                    content = str(item.get("content", ""))
                    try:
                        data = json.loads(content)
                        result = data.get("result", {})
                        if isinstance(result, str):
                            result = json.loads(result)
                        if "device_name" in result:
                            gpu_specs = result
                        if "execution_time_ms" in result and pytorch_execution is None:
                            pytorch_execution = result
                        elif "execution_time_ms" in result:
                            triton_execution = result
                        if "max_difference" in result or "shape_match" in result:
                            validation = result
                    except:
                        pass

                print(pytorch_code)
                print()
                print(triton_code)

                if triton_code:
                    action_context_exec = ActionContext({"seed": idx})
                    pytorch_result = execute_pytorch_code(action_context=action_context_exec, pytorch_code=pytorch_code)
                    input_tensors = pytorch_result.pop("input_tensors", {})
                    triton_result = execute_triton_code(action_context=action_context_exec, triton_code=triton_code, input_tensors=input_tensors)

                    validation_result = None
                    if pytorch_result.get("success") and triton_result.get("success"):
                        validation_result = validate_outputs(
                            action_context=action_context_exec,
                            pytorch_result=json.dumps(pytorch_result["result"]),
                            triton_result=json.dumps(triton_result["result"])
                        )

                    record = {
                        "pytorch_code": pytorch_code,
                        "triton_code": triton_code,
                        "gpu_specs": gpu_specs,
                        "pytorch_execution": pytorch_result,
                        "triton_execution": triton_result,
                        "validation": validation_result,
                        "success": triton_result.get("success", False)
                    }
                    with open(self.output_path, "a") as f:
                        f.write(json.dumps(record) + "\n")
                    print("✅ Guardado correctamente")
                else:
                    print("❌ No se encontró código Triton en memoria")

            except Exception as e:
                print(f"Error processing sample {sample}: {str(e)}")
                traceback.print_exc()

        return result_memory


def create_project_manager_agent(llm_function) -> Agent:
    goals = [
        Goal(1, "Generate Triton Kernel", "Get GPU specs and coordinate with TritonDeveloper to generate a Triton kernel"),
        Goal(2, "Review Kernel", "Coordinate with TritonReviewer to validate the kernel statically"),
        Goal(3, "Terminate", "Terminate with the final kernel code as the message"),
    ]

    action_registry = DecoratorActionRegistry(tags=["triton", "hardware", "selective"])
    action_registry.register_terminate_tool()

    agent_language = AgentJSONActionLanguage()
    environment = ActionContextEnvironment()

    return Agent(
        goals = goals,
        agent_language = agent_language,
        action_registry = action_registry,
        generate_response = llm_function,
        environment = environment,
        agent_name = "Triton Project Manager Agent",
    )

def create_developer_agent(llm_function) -> Agent:

    goals = [
        Goal(1, "Generate Triton Code", "Write efficient Triton code based on the PyTorch implementation and requirements provided by the project manager"),
        Goal(2, "Implement Functionality", "Ensure the generated code correctly implements the required functionality and meets performance standards"),
        Goal(3, "Iterate Based on Feedback", "Refine and improve code based on feedback from the project manager and code reviewer")
    ]


    # Tools for development and coding
    action_registry = DecoratorActionRegistry(tags=["generation", "triton"])
    action_registry.register_terminate_tool()

    agent_language = AgentJSONActionLanguage()
    environment = ActionContextEnvironment()

    return Agent(
        goals=goals,
        agent_language=agent_language,
        action_registry=action_registry,
        generate_response=llm_function,
        environment=environment,
        agent_name="Triton Developer Agent",
        max_iterations=10
    )

def create_code_reviewer_agent(llm_function) -> Agent:

    goals = [
        Goal(1, "Code Quality Review",
             "Review code for quality, best practices, and potential issues"),
        Goal(2, "Performance Review",
             "Analyze code for performance optimization opportunities"),
        Goal(3, "Security Analysis",
             "Identify security vulnerabilities and suggest improvements"),
    ]

    # Tools for review and quality assurance
    action_registry = DecoratorActionRegistry(tags=["validation", "triton", "review"])
    action_registry.register_terminate_tool()

    agent_language = AgentJSONActionLanguage()
    environment = ActionContextEnvironment()

    return Agent(
        goals=goals,
        agent_language=agent_language,
        action_registry=action_registry,
        generate_response=llm_function,
        environment=environment,
        agent_name="Triton Code Reviewer",
        max_iterations=10
    )

def load_qwen_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    model_path = "/content/drive/MyDrive/Colabs/TritonProject/qwen_model"

    if os.path.exists(model_path):
        model_id = model_path
    else:
        model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    return model, tokenizer

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

    model, tokenizer = load_qwen_model()
    
    model_path = "/content/drive/MyDrive/Colabs/TritonProject/qwen_model"
    if not os.path.exists(model_path):
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

    qwen_llm_function = create_qwen_llm_function(model, tokenizer)

    base_path = "/content/drive/MyDrive/Colabs/TritonProject/BddAgent-Triton-"

    tritonProcessor = PyTorchToTritonProcessor(
        qwen_llm_function,
        input_path=f"{base_path}/tritonCodeBlocks.jsonl",
        output_path=f"{base_path}/triton_results.jsonl"
    )
    memory = tritonProcessor.process()
    print(memory.items[-1])

if __name__ == "__main__":
    main()