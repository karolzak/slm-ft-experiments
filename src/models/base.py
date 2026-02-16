from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Information about a model"""
    name: str
    type: str  # 'pretrained', 'finetuned'
    base_model: Optional[str]
    parameters: Optional[int]
    context_length: int
    deployment_endpoint: Optional[str]


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None


class BaseModelWrapper(ABC):
    """
    Abstract base class providing unified interface for all models.
    
    This enables seamless comparison between different model types
    and sizes (GPT-5, Phi, finetuned variants, etc.).
    """
    
    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize model wrapper.
        
        Args:
            model_name: Identifier for the model
            config: Optional configuration dictionary
        """
        self.model_name = model_name
        self.config = config or {}
        self.model = None
        self.tokenizer = None
        
    @abstractmethod
    def load(self) -> None:
        """
        Load the model and tokenizer into memory.
        
        Should handle:
        - Model downloading/caching
        - Device placement (CPU/GPU)
        - Authentication if needed
        """
        pass
    
    @abstractmethod
    def generate(self, 
                prompt: str,
                generation_config: Optional[GenerationConfig] = None,
                **kwargs) -> str:
        """
        Generate text from a single prompt.
        
        Args:
            prompt: Input text prompt
            generation_config: Generation parameters
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated text string
        """
        pass
    
    def batch_generate(self,
                      prompts: List[str],
                      generation_config: Optional[GenerationConfig] = None,
                      batch_size: int = 8,
                      **kwargs) -> List[str]:
        """
        Generate text for multiple prompts efficiently.
        
        Args:
            prompts: List of input prompts
            generation_config: Generation parameters
            batch_size: Number of prompts to process at once
            **kwargs: Additional parameters
            
        Returns:
            List of generated text strings
        """
        pass
    
    def get_embeddings(self, text: str) -> List[float]:
        """
        Get embeddings for input text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as list of floats
        """
        pass
    
    def get_model_info(self) -> ModelInfo:
        """
        Get information about the model.
        
        Returns:
            ModelInfo dataclass instance
        """
        pass
    
    def estimate_cost(self, 
                     num_input_tokens: int,
                     num_output_tokens: int) -> float:
        """
        Estimate cost for generation.
        
        Args:
            num_input_tokens: Number of input tokens
            num_output_tokens: Number of output tokens
            
        Returns:
            Estimated cost in USD
        """
        pass
    
    def unload(self) -> None:
        """
        Unload model from memory.
        
        Useful for freeing up resources when switching models.
        """
        pass
