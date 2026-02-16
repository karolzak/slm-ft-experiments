from typing import Dict, Any, List
import pandas as pd


class DataPreprocessor:
    """
    Preprocess datasets for different model types and training frameworks.
    
    Handles tokenization, formatting, and conversion to training-ready format.
    """
    
    def __init__(self, tokenizer=None):
        """
        Initialize preprocessor.
        
        Args:
            tokenizer: Optional tokenizer instance
        """
        self.tokenizer = tokenizer
    
    def preprocess_for_training(self, 
                               dataset: pd.DataFrame,
                               model_type: str,
                               task_type: str) -> Any:
        """
        Preprocess dataset for training.
        
        Args:
            dataset: Raw dataset DataFrame
            model_type: Type of model ('gpt', 'phi', etc.)
            task_type: Type of task ('classification', 'summarization', etc.)
            
        Returns:
            Preprocessed dataset ready for training
        """
        pass
    
    def preprocess_for_evaluation(self,
                                  dataset: pd.DataFrame,
                                  model_type: str,
                                  task_type: str) -> Any:
        """
        Preprocess dataset for evaluation.
        
        Args:
            dataset: Raw dataset DataFrame
            model_type: Type of model
            task_type: Type of task
            
        Returns:
            Preprocessed dataset ready for evaluation
        """
        pass
    
    def format_prompt(self, 
                     sample: Dict[str, Any],
                     task_type: str,
                     include_system_message: bool = True) -> str:
        """
        Format a single sample into a prompt.
        
        Args:
            sample: Dictionary with sample data
            task_type: Type of task
            include_system_message: Whether to include system message
            
        Returns:
            Formatted prompt string
        """
        pass
    
    def tokenize(self, texts: List[str], **kwargs) -> Any:
        """
        Tokenize texts.
        
        Args:
            texts: List of text strings
            **kwargs: Additional tokenization parameters
            
        Returns:
            Tokenized output
        """
        pass
