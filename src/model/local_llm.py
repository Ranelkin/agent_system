import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional, Any
import os

class AutoLLM:
    """Automatically uses all available GPUs/CPUs for LLM inference"""
    
    def __init__(
        self,
        model_id: str = "openai/gpt-oss-20b",
        dtype: Optional[str] = None,  # Changed from torch_dtype
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
                dtype = torch.float16
            else:
                # Use float32 on CPU (more stable)
                dtype = torch.float32
        elif isinstance(dtype, str):
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            dtype = dtype_map.get(dtype, torch.float32)
        
        # Model loading configuration
        model_kwargs = {
            "device_map": self.device_map,
            "torch_dtype": dtype,  # Still use torch_dtype for transformers
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        
        # Add quantization if requested (mutually exclusive)
        if self.num_gpus > 0:  # Quantization typically requires GPU
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
        
        # Try to use Flash Attention 2 if available (only on GPU)
        if self.num_gpus > 0:
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                print("Flash Attention 2 enabled")
            except:
                pass
        
        # Initialize the pipeline
        print(f"Loading model: {model_id}")
        try:
            self.pipe = pipeline(
                "text-generation",
                model=model_id,
                **model_kwargs
            )
        except Exception as e:
            print(f"Error loading model with auto settings: {e}")
            print("Attempting fallback loading...")
            
            # Fallback: Try simpler loading
            fallback_kwargs = {
                "device_map": self.device_map,
                "torch_dtype": dtype,
                "trust_remote_code": True,
            }
            self.pipe = pipeline(
                "text-generation",
                model=model_id,
                **fallback_kwargs
            )
        
        # Configure tokenizer
        if self.pipe.tokenizer.pad_token is None:
            self.pipe.tokenizer.pad_token = self.pipe.tokenizer.eos_token
            self.pipe.tokenizer.pad_token_id = self.pipe.tokenizer.eos_token_id
        
        print("Model loaded successfully!")
        self._print_memory_usage()
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate text from messages"""
        
        # Prepare generation parameters (excluding model loading params)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
            "pad_token_id": self.pipe.tokenizer.pad_token_id,
            "eos_token_id": self.pipe.tokenizer.eos_token_id,
        }
        
        # Add any additional generation parameters
        # Filter out any model loading parameters that might have been passed
        excluded_params = {'device_map', 'torch_dtype', 'dtype', 'low_cpu_mem_usage', 
                          'trust_remote_code', 'attn_implementation', 'load_in_8bit', 
                          'load_in_4bit', 'bnb_4bit_compute_dtype'}
        for k, v in kwargs.items():
            if k not in excluded_params:
                gen_kwargs[k] = v
        
        # Use automatic mixed precision if GPU is available
        if self.num_gpus > 0 and torch.cuda.is_available():
            try:
                with torch.cuda.amp.autocast(enabled=True):
                    outputs = self.pipe(messages, **gen_kwargs)
            except Exception as e:
                print(f"AMP failed, falling back to normal generation: {e}")
                outputs = self.pipe(messages, **gen_kwargs)
        else:
            outputs = self.pipe(messages, **gen_kwargs)
        
        return outputs
    
    def __call__(self, messages: List[Dict[str, str]], **kwargs):
        """Make the class callable"""
        return self.generate(messages, **kwargs)
    
    def query(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Simple query that returns just the text"""
        outputs = self.generate(messages, **kwargs)
        # Handle different output formats
        if isinstance(outputs, list) and len(outputs) > 0:
            if isinstance(outputs[0], dict):
                if "generated_text" in outputs[0]:
                    gen_text = outputs[0]["generated_text"]
                    if isinstance(gen_text, list) and len(gen_text) > 0:
                        if isinstance(gen_text[-1], dict) and "content" in gen_text[-1]:
                            return gen_text[-1]["content"]
                        return str(gen_text[-1])
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
MODEL_ID = os.environ.get("LLM_MODEL_ID", "microsoft/phi-2")
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
def query(messages: List[Dict[str, str]], **kwargs) -> str:
    """Quick query function that returns generated text"""
    return llm.query(messages, **kwargs)


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
        print("The model may require specific hardware or configuration.")