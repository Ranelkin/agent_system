import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional, Any, Union
import os
import json
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.tools import BaseTool

class AutoLLM(BaseChatModel):
    """LangChain-compatible LLM that automatically uses all available GPUs/CPUs"""
    
    model_id: str = "openai/gpt-oss-20b"
    dtype: Optional[str] = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    pipe: Any = None
    device_map: str = "auto"
    num_gpus: int = 0
    supports_chat: bool = False
    _bound_tools: List[Any] = []
    
    def __init__(
        self,
        model_id: str = "openai/gpt-oss-20b",
        dtype: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        _skip_load: bool = False,
        **kwargs
    ):
        """
        Initialize pipeline with automatic hardware detection
        
        Args:
            model_id: Hugging Face model ID or local path
            dtype: "auto", "float16", "bfloat16", "float32", or None for auto
            load_in_8bit: Use 8-bit quantization for memory efficiency
            load_in_4bit: Use 4-bit quantization for even more memory efficiency
            _skip_load: Internal flag to skip model loading (for bind_tools)
        """
        super().__init__(**kwargs)
        self.model_id = model_id
        self.dtype = dtype
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self._bound_tools = []
        
        # Skip loading if this is a bound copy
        if _skip_load:
            return
        
        # Detect available hardware
        self.device_map = "auto"
        self.num_gpus = torch.cuda.device_count()
        
        print(f"Detected {self.num_gpus} GPU(s)")
        if self.num_gpus > 0:
            for i in range(self.num_gpus):
                gpu_name = torch.cuda.get_device_properties(i).name
                gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            print("Running on CPU")
        
        # Determine dtype based on hardware
        if dtype is None or dtype == "auto":
            if self.num_gpus > 0:
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
        elif isinstance(dtype, str):
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            torch_dtype = dtype_map.get(dtype, torch.float32)
        else:
            torch_dtype = dtype
        
        # Model loading configuration
        model_kwargs = {
            "device_map": self.device_map,
            "torch_dtype": torch_dtype
        }
        
        if self.num_gpus > 0:  
            if load_in_4bit:
                model_kwargs["load_in_4bit"] = True
                model_kwargs["bnb_4bit_compute_dtype"] = torch.float16
                print("Using 4-bit quantization")
            elif load_in_8bit:
                model_kwargs["load_in_8bit"] = True
                print("Using 8-bit quantization")
        else:
            if load_in_4bit or load_in_8bit:
                print("Warning: Quantization requested but no GPU available. Running in full precision.")
        
        # Initialize the pipeline
        print(f"Loading model: {model_id}")
        try:
            self.pipe = pipeline(
                "text-generation",
                model=model_id,
                model_kwargs=model_kwargs
            )
        except Exception as e:
            print(f"Error loading model with auto settings: {e}")
            print("Attempting fallback loading...")
            
            fallback_kwargs = {
                "device_map": self.device_map,
                "torch_dtype": torch_dtype,
                "trust_remote_code": True,
            }
            self.pipe = pipeline(
                "text-generation",
                model=model_id,
                model_kwargs=fallback_kwargs
            )
        
        # Configure tokenizer
        if self.pipe.tokenizer.pad_token is None:
            self.pipe.tokenizer.pad_token = self.pipe.tokenizer.eos_token
            self.pipe.tokenizer.pad_token_id = self.pipe.tokenizer.eos_token_id
        
        # Check if model supports chat templates
        self.supports_chat = hasattr(self.pipe.tokenizer, 'chat_template') and self.pipe.tokenizer.chat_template is not None
        if not self.supports_chat:
            print("Note: This model doesn't have a chat template. Using plain text mode.")
        
        print("Model loaded successfully!")
        self._print_memory_usage()
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "local_huggingface"
    
    def _format_tools_for_prompt(self) -> str:
        """Format bound tools as a prompt section"""
        if not self._bound_tools:
            return ""
        
        tool_descriptions = ["\n=== AVAILABLE TOOLS ===\n"]
        
        for tool in self._bound_tools:
            if hasattr(tool, 'name') and hasattr(tool, 'description'):
                name = tool.name
                description = tool.description
                
                if hasattr(tool, 'args_schema') and tool.args_schema:
                    try:
                        schema = tool.args_schema.schema()
                        properties = schema.get('properties', {})
                        required = schema.get('required', [])
                        args_list = [f"{k} ({v.get('type', 'any')}){' [required]' if k in required else ''}" 
                                    for k, v in properties.items()]
                        args_desc = ", ".join(args_list)
                    except:
                        args_desc = "See documentation"
                else:
                    args_desc = "No arguments"
                
                tool_descriptions.append(f"\nTool: {name}\n")
                tool_descriptions.append(f"Description: {description}\n")
                tool_descriptions.append(f"Arguments: {args_desc}\n")
        
        tool_descriptions.append(
            "\n=== INSTRUCTIONS ===\n"
            "To use a tool, respond with ONLY this JSON format (no other text):\n"
            "```json\n"
            "{\n"
            '  "tool": "tool_name",\n'
            '  "arguments": {"arg1": "value1"}\n'
            "}\n"
            "```\n"
            "\nFor the comment_codebase tool, extract the exact path from the user's message.\n"
            "Example: If user says 'comment /home/user/code', use:\n"
            '```json\n{"tool": "comment_codebase", "arguments": {"path": "/home/user/code"}}```\n'
        )
        
        return "".join(tool_descriptions)
    
    def bind_tools(
        self,
        tools: List[Any],
        **kwargs: Any,
    ) -> "AutoLLM":
        """Bind tools to the model - required for create_react_agent.
        
        For local models without native tool support, we store the tools
        and include them in the prompt so the model knows what's available.
        """
        # Create a new instance without loading the model again
        bound = self.__class__(
            model_id=self.model_id,
            dtype=self.dtype,
            load_in_8bit=self.load_in_8bit,
            load_in_4bit=self.load_in_4bit,
            _skip_load=True
        )
        # Copy all the loaded attributes
        bound.pipe = self.pipe
        bound.device_map = self.device_map
        bound.num_gpus = self.num_gpus
        bound.supports_chat = self.supports_chat
        
        # Store tools - THIS IS KEY!
        bound._bound_tools = tools
        
        print(f"Bound {len(tools)} tools to local LLM:")
        for tool in tools:
            tool_name = getattr(tool, 'name', getattr(tool, '__name__', 'Unknown'))
            print(f"  - {tool_name}")
        
        return bound
    
    def with_structured_output(self, schema: Any, **kwargs: Any) -> "AutoLLM":
        """Return a model that outputs structured data.
        
        For local models, this is a pass-through since we don't have
        native structured output support. The agent framework will
        handle parsing.
        """
        return self
    
    def _generate(
    self,
    messages: List[BaseMessage],
    stop: Optional[List[str]] = None,
    run_manager: Optional[Any] = None,
    **kwargs: Any,
    ) -> ChatResult:
        """Generate chat completion with tool calling support"""
        
        formatted_messages = self._convert_messages(messages)
        
        # Inject tool information
        if self._bound_tools and formatted_messages:
            tools_prompt = self._format_tools_for_prompt()
            system_msg_idx = next(
                (i for i, msg in enumerate(formatted_messages) if msg.get('role') == 'system'),
                None
            )
            
            if system_msg_idx is not None:
                formatted_messages[system_msg_idx]['content'] += "\n" + tools_prompt
            else:
                formatted_messages.insert(0, {'role': 'system', 'content': tools_prompt})
        
        # Generate response
        max_new_tokens = kwargs.pop("max_new_tokens", 512)
        temperature = kwargs.pop("temperature", 0.7)
        top_p = kwargs.pop("top_p", 0.9)
        do_sample = kwargs.pop("do_sample", True)
        
        response_text = self._generate_text(
            formatted_messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            **kwargs
        )
        
        # Try to parse tool calls from response
        tool_calls = self._parse_tool_calls(response_text)
        print(f"[DEBUG] Parsed tool calls: {tool_calls}")  # Add this

        if tool_calls:
            message = AIMessage(
                content="",  # Empty content when tool calling
                additional_kwargs={"tool_calls": tool_calls}
            )
        else:
            message = AIMessage(content=response_text)    
                    
                    
        if tool_calls:
            # Create AIMessage with tool calls
            message = AIMessage(
                content=response_text,
                additional_kwargs={"tool_calls": tool_calls}
            )
        else:
            message = AIMessage(content=response_text)
        
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _parse_tool_calls(self, text: str) -> List[Dict]:
        """Parse tool calls from model output"""
        import re
        
        tool_calls = []
        
        # Pattern 1: JSON in code blocks
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        # Pattern 2: Plain JSON with "tool" and "arguments" keys
        if not matches:
            # Look for {...} containing both "tool" and "arguments"
            json_pattern = r'\{[^{}]*"tool"[^{}]*"arguments"[^{}]*\}'
            matches = re.findall(json_pattern, text)
            
            # If nested braces, try extracting the full object
            if not matches:
                # More aggressive: find any {...} that might be a tool call
                potential_jsons = re.findall(r'\{(?:[^{}]|\{[^{}]*\})*\}', text)
                for pj in potential_jsons:
                    try:
                        data = json.loads(pj)
                        if "tool" in data and "arguments" in data:
                            matches.append(pj)
                    except:
                        continue
        
        for match in matches:
            try:
                tool_data = json.loads(match)
                if "tool" in tool_data and "arguments" in tool_data:
                    tool_calls.append({
                        "id": f"call_{len(tool_calls)}",
                        "type": "function",
                        "function": {
                            "name": tool_data["tool"],
                            "arguments": json.dumps(tool_data["arguments"])
                        }
                    })
            except json.JSONDecodeError as e:
                print(f"[DEBUG] Failed to parse JSON: {match[:100]}... Error: {e}")
                continue
        
        return tool_calls

    def _convert_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Convert LangChain messages to dict format"""
        converted = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                converted.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                converted.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                converted.append({"role": "system", "content": msg.content})
            else:
                # Fallback for other message types
                converted.append({"role": "user", "content": str(msg.content)})
        return converted
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to appropriate format for the model"""
        
        # If model supports chat templates, return as-is for pipeline
        if self.supports_chat:
            return messages
        
        # Otherwise, convert to plain text format
        formatted_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                formatted_parts.append(f"System: {content}")
            elif role == "user":
                formatted_parts.append(f"User: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}")
            else:
                formatted_parts.append(content)
        
        return "\n\n".join(formatted_parts) + "\n\nAssistant:"
    
    def _generate_text(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """Internal method to generate text"""
        
        formatted_input = self._format_messages(messages)
        
        # Prepare gen params
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
            "pad_token_id": self.pipe.tokenizer.pad_token_id,
            "eos_token_id": self.pipe.tokenizer.eos_token_id,
            "return_full_text": False,
        }
        
        excluded_params = {
            'device_map', 'torch_dtype', 'dtype', 'low_cpu_mem_usage', 
            'attn_implementation', 'load_in_8bit', 
            'load_in_4bit', 'bnb_4bit_compute_dtype', 'model_kwargs'
        }
        for k, v in kwargs.items():
            if k not in excluded_params:
                gen_kwargs[k] = v
        
        # Generate with automatic mixed precision if GPU available
        if self.num_gpus > 0 and torch.cuda.is_available():
            try:
                with torch.amp.autocast('cuda', enabled=True):
                    outputs = self.pipe(formatted_input, **gen_kwargs)
            except Exception as e:
                print(f"AMP failed, using standard precision: {e}")
                outputs = self.pipe(formatted_input, **gen_kwargs)
        else:
            outputs = self.pipe(formatted_input, **gen_kwargs)
        
        # Extract text from outputs
        return self._extract_text(outputs)
    
    def _extract_text(self, outputs) -> str:
        """Extract text from pipeline outputs"""
        if isinstance(outputs, list) and len(outputs) > 0:
            if isinstance(outputs[0], dict):
                if "generated_text" in outputs[0]:
                    gen_text = outputs[0]["generated_text"]
                    
                    if isinstance(gen_text, list) and len(gen_text) > 0:
                        if isinstance(gen_text[-1], dict) and "content" in gen_text[-1]:
                            return gen_text[-1]["content"]
                        return str(gen_text[-1])
                    
                    if isinstance(gen_text, str):
                        return gen_text.strip()
                    
                    return str(gen_text)
        
        return str(outputs)
    
    def _print_memory_usage(self):
        """Print current memory usage"""
        if self.num_gpus > 0 and torch.cuda.is_available():
            for i in range(self.num_gpus):
                try:
                    allocated = torch.cuda.memory_allocated(i) / 1e9
                    reserved = torch.cuda.memory_reserved(i) / 1e9
                    print(f"  GPU {i} memory: {allocated:.1f}/{reserved:.1f} GB allocated/reserved")
                except:
                    pass


# Create default instance for easy import
MODEL_ID = os.environ.get("LLM_MODEL_ID", "openai/gpt-oss-20b")
USE_8BIT = os.environ.get("USE_8BIT", "false").lower() == "true"
USE_4BIT = os.environ.get("USE_4BIT", "false").lower() == "true"

print("Initializing default local LLM instance...")




# Example usage
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage, SystemMessage
    
    try:
        local_llm = AutoLLM(
            model_id=MODEL_ID,
            dtype="auto",
            load_in_8bit=USE_8BIT,
            load_in_4bit=USE_4BIT
        )
    except Exception as e:
        print(f"Failed to load {MODEL_ID}: {e}")
        print("Falling back to GPT-2...")
        local_llm = AutoLLM(
            model_id="gpt2",
            dtype="float32",
            load_in_8bit=False,
            load_in_4bit=False
        )

    # Test the LangChain interface
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Explain quantum mechanics in one sentence.")
    ]
    
    try:
        response = local_llm.invoke(messages)
        print("\nGenerated text:")
        print(response.content)
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()