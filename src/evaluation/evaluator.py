from typing import List, Dict, Any
from dataclasses import dataclass
import pandas as pd

from ..models.base import BaseModelWrapper
from .metrics.base import Metric


@dataclass
class EvaluationResults:
    """Results from model evaluation"""
    model_name: str
    task_type: str
    metrics: Dict[str, float]
    predictions: List[str]
    references: List[str]
    latency_stats: Dict[str, float]
    cost_estimate: float
    metadata: Dict[str, Any]


class ModelEvaluator:
    """
    Evaluate models on various metrics and tasks.
    
    Supports:
    - Multiple evaluation metrics
    - Batch evaluation
    - Performance tracking (latency, cost)
    - Result aggregation and comparison
    """
    
    def __init__(self, metrics: List[Metric]):
        """
        Initialize evaluator.
        
        Args:
            metrics: List of Metric instances to compute
        """
        self.metrics = metrics
        
    def evaluate(self,
                model: BaseModelWrapper,
                dataset: pd.DataFrame,
                task_type: str,
                batch_size: int = 8) -> EvaluationResults:
        """
        Evaluate model on dataset.
        
        Args:
            model: Model wrapper instance
            dataset: Evaluation dataset
            task_type: Type of task
            batch_size: Batch size for generation
            
        Returns:
            EvaluationResults object
        """
        pass
    
    def compare(self,
               results_list: List[EvaluationResults]) -> pd.DataFrame:
        """
        Compare results from multiple models.
        
        Args:
            results_list: List of EvaluationResults
            
        Returns:
            DataFrame with comparison metrics
        """
        pass
    
    def _generate_predictions(self,
                            model: BaseModelWrapper,
                            dataset: pd.DataFrame,
                            task_type: str,
                            batch_size: int) -> List[str]:
        """
        Generate predictions for entire dataset.
        
        Args:
            model: Model wrapper
            dataset: Input dataset
            task_type: Task type
            batch_size: Batch size
            
        Returns:
            List of predictions
        """
        pass
    
    def _compute_metrics(self,
                        predictions: List[str],
                        references: List[str]) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Args:
            predictions: Model predictions
            references: Ground truth references
            
        Returns:
            Dictionary of metric scores
        """
        pass
