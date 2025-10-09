import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional, Any, Union
import os

class AutoLLM:
    """Automatically uses all available GPUs/CPUs for LLM inference"""
    
    def __init__(
        self,
        model_id: str = "openai/gpt-oss-20b",
        dtype: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        """
        Initialize pipeline with automatic hardware detection
        
        Args:
            model_id: Hugging Face model ID or local path
            dtype: "auto", "float16", "bfloat16", "float32", or None for auto
            load_in_8bit: Use 8-bit quantization for memory efficiency
            load_in_4bit: Use 4-bit quantization for even more memory efficiency
        """
        self.model_id = model_id
        
        # Detect available hardware
        self.device_map = "auto"  # Automatically distribute across all GPUs/CPU
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
                # Use float16 on GPU for efficiency
                torch_dtype = torch.float16
            else:
                # Use float32 on CPU (more stable)
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
            
            # Fallback: Try simpler loading
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
    
    def _format_messages(self, messages: Union[List[Dict[str, str]], str]) -> str:
        """Convert messages to appropriate format for the model"""
        # If it's already a string, return it
        if isinstance(messages, str):
            return messages
        
        # If model supports chat templates, let the pipeline handle it
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
    
    def generate(
        self,
        messages: Union[List[Dict[str, str]], str],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate text from messages"""
        
        formatted_input = self._format_messages(messages)
        
        # Prepare gen params
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
            "pad_token_id": self.pipe.tokenizer.pad_token_id,
            "eos_token_id": self.pipe.tokenizer.eos_token_id,
            "return_full_text": False,  # Only return generated text, not input
        }
        
        excluded_params = {
            'device_map', 'torch_dtype', 'dtype', 'low_cpu_mem_usage', 
            'attn_implementation', 'load_in_8bit', 
            'load_in_4bit', 'bnb_4bit_compute_dtype', 'model_kwargs'
        }
        for k, v in kwargs.items():
            if k not in excluded_params:
                gen_kwargs[k] = v
        
        # Use automatic mixed precision if GPU is available
        if self.num_gpus > 0 and torch.cuda.is_available():
            try:
                with torch.amp.autocast('cuda', enabled=True):
                    outputs = self.pipe(formatted_input, **gen_kwargs)
            except Exception as e:
                # If AMP fails, try without it
                print(f"AMP failed, using standard precision: {e}")
                outputs = self.pipe(formatted_input, **gen_kwargs)
        else:
            outputs = self.pipe(formatted_input, **gen_kwargs)
        
        return outputs
    
    def __call__(self, messages: Union[List[Dict[str, str]], str], **kwargs):
        """Make the class callable"""
        return self.generate(messages, **kwargs)
    
    def query(self, messages: Union[List[Dict[str, str]], str], **kwargs) -> str:
        """Simple query that returns just the text"""
        outputs = self.generate(messages, **kwargs)
        
        # Handle different output formats
        if isinstance(outputs, list) and len(outputs) > 0:
            if isinstance(outputs[0], dict):
                if "generated_text" in outputs[0]:
                    gen_text = outputs[0]["generated_text"]
                    
                    # If it's a chat-formatted response
                    if isinstance(gen_text, list) and len(gen_text) > 0:
                        if isinstance(gen_text[-1], dict) and "content" in gen_text[-1]:
                            return gen_text[-1]["content"]
                        return str(gen_text[-1])
                    
                    # If it's plain text response
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
# Can be configured via environment variables
MODEL_ID = os.environ.get("LLM_MODEL_ID", "openai/gpt-oss-20b")
USE_8BIT = os.environ.get("USE_8BIT", "false").lower() == "true"
USE_4BIT = os.environ.get("USE_4BIT", "false").lower() == "true"

print("Initializing default LLM instance...")

# Try to load with specified settings
try:
    llm = AutoLLM(
        model_id=MODEL_ID,
        dtype="auto",
        load_in_8bit=USE_8BIT,
        load_in_4bit=USE_4BIT
    )
except Exception as e:
    print(f"Failed to load {MODEL_ID}: {e}")
    print("Falling back to GPT-2...")
    # Fallback to a smaller model that works on CPU
    llm = AutoLLM(
        model_id="gpt2",
        dtype="float32",
        load_in_8bit=False,
        load_in_4bit=False
    )

# Convenience function for direct usage
def query(messages: Union[List[Dict[str, str]], str], **kwargs) -> str:
    """Quick query function that returns generated text"""
    return llm.query(messages, **kwargs)
# For export / use in other modules 
llm = AutoLLM(
        model_id=MODEL_ID,
        dtype="auto",
        load_in_8bit=USE_8BIT,
        load_in_4bit=USE_4BIT
    )

# Example usage
if __name__ == "__main__":
    # Test the pipeline
    messages = [
        {"role": "user", "content": "Explain quantum mechanics in one sentence."},
    ]
    
    try:
        # Method 1: Using the singleton
        response = llm.query(messages, max_new_tokens=100)
        print("\nGenerated text:")
        print(response)
        
        # Method 2: Get full output
        full_output = llm(messages, max_new_tokens=100)
        print("\nFull output structure available if needed")
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
        print("The model may require specific hardware or configuration.")