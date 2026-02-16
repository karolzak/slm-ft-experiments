from abc import ABC, abstractmethod
from typing import List, Any, Dict


class Metric(ABC):
    """
    Abstract base class for evaluation metrics.
    """
    
    def __init__(self, name: str):
        """
        Initialize metric.
        
        Args:
            name: Metric name
        """
        self.name = name
    
    @abstractmethod
    def compute(self,
               predictions: List[str],
               references: List[str],
               **kwargs) -> float:
        """
        Compute metric score.
        
        Args:
            predictions: Model predictions
            references: Ground truth references
            **kwargs: Additional parameters
            
        Returns:
            Metric score
        """
        pass
    
    def batch_compute(self,
                     predictions_list: List[List[str]],
                     references_list: List[List[str]]) -> List[float]:
        """
        Compute metric for multiple batches.
        
        Args:
            predictions_list: List of prediction batches
            references_list: List of reference batches
            
        Returns:
            List of scores
        """
        pass
