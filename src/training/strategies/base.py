from abc import ABC, abstractmethod
from typing import Any


class TrainingStrategy(ABC):
    """
    Abstract base class for training strategies.
    
    Different strategies (Full, LoRA, QLoRA) implement this interface.
    """
    
    @abstractmethod
    def prepare_model(self, model: Any, config: dict[str, Any]) -> Any:
        """
        Prepare model for training with this strategy.
        
        Args:
            model: Base model instance
            config: Strategy-specific configuration
            
        Returns:
            Modified model ready for training
        """
        pass
    
    @abstractmethod
    def get_trainable_parameters(self, model: Any) -> int:
        """
        Get number of trainable parameters.
        
        Args:
            model: Model instance
            
        Returns:
            Number of trainable parameters
        """
        pass
