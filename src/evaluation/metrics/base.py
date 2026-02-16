from abc import ABC, abstractmethod


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
               predictions: list[str],
               references: list[str],
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
                     predictions_list: list[list[str]],
                     references_list: list[list[str]]) -> list[float]:
        """
        Compute metric for multiple batches.
        
        Args:
            predictions_list: List of prediction batches
            references_list: List of reference batches
            
        Returns:
            List of scores
        """
        pass
