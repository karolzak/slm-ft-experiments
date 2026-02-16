from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass
import pandas as pd


@dataclass
class DatasetConfig:
    """Configuration for dataset generation"""
    num_samples: int
    task_type: str
    difficulty_level: str  # 'easy', 'medium', 'hard'
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    seed: int = 42
    additional_params: dict[str, Any] | None = None
    
    def __post_init__(self):
        """Validate that splits sum to 1.0"""
        total = self.train_split + self.val_split + self.test_split
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(
                f"train_split ({self.train_split}) + val_split ({self.val_split}) + "
                f"test_split ({self.test_split}) must sum to 1.0, got {total}"
            )


class BaseDatasetGenerator(ABC):
    """
    Abstract base class for synthetic dataset generation.
    
    All task-specific generators should inherit from this class.
    """
    
    def __init__(self, config: DatasetConfig):
        """
        Initialize the dataset generator.
        
        Args:
            config: DatasetConfig object with generation parameters
        """
        self.config = config
        self.seed = config.seed
        
    @abstractmethod
    def generate(self) -> dict[str, pd.DataFrame]:
        """
        Generate synthetic dataset.
        
        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames
        """
        pass
    
    @abstractmethod
    def validate(self, dataset: dict[str, pd.DataFrame]) -> bool:
        """
        Validate generated dataset quality.
        
        Args:
            dataset: Dictionary of DataFrames to validate
            
        Returns:
            True if valid, False otherwise
        """
        pass
    
    def save(self, dataset: dict[str, pd.DataFrame], output_dir: str) -> None:
        """
        Save dataset to disk.
        
        Args:
            dataset: Dictionary of DataFrames
            output_dir: Directory to save files
        """
        pass
    
    def load(self, input_dir: str) -> dict[str, pd.DataFrame]:
        """
        Load dataset from disk.
        
        Args:
            input_dir: Directory containing dataset files
            
        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames
        """
        pass
    
    def get_statistics(self, dataset: dict[str, pd.DataFrame]) -> dict[str, Any]:
        """
        Compute dataset statistics.
        
        Args:
            dataset: Dictionary of DataFrames
            
        Returns:
            Dictionary with statistics
        """
        pass
