# LLM/SLM Finetuning Experimentation Framework - Project Specification

## Project Overview

**Purpose**: Create a comprehensive framework to test and compare off-the-shelf pretrained models (GPT-5, GPT-5 Mini, GPT-5 Nano, Phi) against finetuned versions for various tasks.

**Key Features**:
- Synthetic dataset generation for multiple task types
- Code-first finetuning pipeline
- Azure AI Foundry integration
- Comprehensive evaluation and comparison framework
- Support for multiple finetuning strategies (full, LoRA, QLoRA)

**Tech Stack**:
- Python 3.10+
- Azure ML SDK / Azure AI Foundry SDK
- PyTorch / Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- MLflow or Azure ML tracking
- Pandas, NumPy, Scikit-learn

---

## Project Structure

```
llm-finetuning-experiments/
├── config/
│   ├── models.yaml                 # Model specifications and endpoints
│   ├── experiments.yaml            # Experiment definitions
│   ├── tasks.yaml                  # Task configurations
│   └── azure_foundry.yaml          # Azure AI Foundry settings
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── generators/
│   │   │   ├── __init__.py
│   │   │   ├── base.py             # BaseDatasetGenerator
│   │   │   └── task_generators.py  # Task-specific generators
│   │   ├── preprocessors/
│   │   │   ├── __init__.py
│   │   │   └── preprocessor.py     # DataPreprocessor
│   │   └── dataset.py              # Dataset utilities
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                 # BaseModelWrapper
│   │   ├── pretrained/
│   │   │   ├── __init__.py
│   │   │   ├── gpt5.py             # GPT-5 wrappers
│   │   │   └── phi.py              # Phi model wrapper
│   │   └── finetuned/
│   │       ├── __init__.py
│   │       └── wrapper.py          # FinetunedModelWrapper
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py              # FineTuner class
│   │   ├── config.py               # TrainingConfig, LoRAConfig
│   │   └── strategies/
│   │       ├── __init__.py
│   │       ├── base.py             # TrainingStrategy
│   │       ├── full.py             # FullFineTuning
│   │       ├── lora.py             # LoRAFineTuning
│   │       └── qlora.py            # QLoRAFineTuning
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py            # ModelEvaluator
│   │   ├── metrics/
│   │   │   ├── __init__.py
│   │   │   ├── base.py             # Metric base class
│   │   │   ├── accuracy.py         # AccuracyMetric
│   │   │   ├── nlg_metrics.py      # BLEU, ROUGE, etc.
│   │   │   ├── performance.py      # Latency, Cost metrics
│   │   │   └── custom.py           # CustomTaskMetric
│   │   └── comparators/
│   │       ├── __init__.py
│   │       └── comparator.py       # ModelComparator
│   │
│   ├── azure_foundry/
│   │   ├── __init__.py
│   │   ├── client.py               # AzureFoundryClient
│   │   └── job_manager.py          # FineTuningJobManager
│   │
│   ├── orchestration/
│   │   ├── __init__.py
│   │   ├── experiment.py           # Experiment class
│   │   └── pipeline.py             # ExperimentPipeline
│   │
│   └── utils/
│       ├── __init__.py
│       ├── tracking/
│       │   ├── __init__.py
│       │   └── tracker.py          # ExperimentTracker
│       ├── visualization/
│       │   ├── __init__.py
│       │   └── visualizer.py       # ResultsVisualizer
│       └── config/
│           ├── __init__.py
│           └── manager.py          # ConfigManager
│
├── experiments/
│   └── runs/
│       └── {experiment_id}/
│           ├── models/
│           ├── results/
│           ├── logs/
│           └── artifacts/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_evaluation.ipynb
│   └── 03_results_analysis.ipynb
│
├── tests/
│   ├── test_data/
│   ├── test_models/
│   ├── test_training/
│   └── test_evaluation/
│
├── scripts/
│   ├── generate_data.py
│   ├── run_experiment.py
│   ├── evaluate_models.py
│   └── compare_results.py
│
├── requirements.txt
├── setup.py
├── README.md
└── .env.example
```

---

## Component Specifications

### 1. Data Generation Module

#### `src/data/generators/base.py`

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
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
    additional_params: Optional[Dict[str, Any]] = None

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
    def generate(self) -> Dict[str, pd.DataFrame]:
        """
        Generate synthetic dataset.
        
        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames
        """
        pass
    
    @abstractmethod
    def validate(self, dataset: Dict[str, pd.DataFrame]) -> bool:
        """
        Validate generated dataset quality.
        
        Args:
            dataset: Dictionary of DataFrames to validate
            
        Returns:
            True if valid, False otherwise
        """
        pass
    
    def save(self, dataset: Dict[str, pd.DataFrame], output_dir: str) -> None:
        """
        Save dataset to disk.
        
        Args:
            dataset: Dictionary of DataFrames
            output_dir: Directory to save files
        """
        pass
    
    def load(self, input_dir: str) -> Dict[str, pd.DataFrame]:
        """
        Load dataset from disk.
        
        Args:
            input_dir: Directory containing dataset files
            
        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames
        """
        pass
    
    def get_statistics(self, dataset: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Compute dataset statistics.
        
        Args:
            dataset: Dictionary of DataFrames
            
        Returns:
            Dictionary with statistics
        """
        pass
```

#### `src/data/generators/task_generators.py`

```python
from .base import BaseDatasetGenerator, DatasetConfig
from typing import Dict, List
import pandas as pd

class ClassificationDataGenerator(BaseDatasetGenerator):
    """
    Generate synthetic classification datasets.
    
    Supports binary and multi-class classification.
    Generates text samples with corresponding labels.
    """
    
    def __init__(self, config: DatasetConfig, num_classes: int = 2):
        """
        Initialize classification data generator.
        
        Args:
            config: DatasetConfig object
            num_classes: Number of classes (default: 2 for binary)
        """
        super().__init__(config)
        self.num_classes = num_classes
    
    def generate(self) -> Dict[str, pd.DataFrame]:
        """
        Generate classification dataset with prompts and labels.
        
        Returns:
            Dict with 'train', 'val', 'test' DataFrames
            Each DataFrame has columns: ['prompt', 'label', 'difficulty']
        """
        pass
    
    def validate(self, dataset: Dict[str, pd.DataFrame]) -> bool:
        """
        Validate classification dataset.
        
        Checks:
        - Label distribution balance
        - No missing values
        - Appropriate text length distribution
        """
        pass

class SummarizationDataGenerator(BaseDatasetGenerator):
    """
    Generate synthetic summarization datasets.
    
    Creates document-summary pairs with varying complexity.
    """
    
    def __init__(self, config: DatasetConfig, 
                 min_doc_length: int = 500,
                 max_doc_length: int = 2000):
        """
        Initialize summarization data generator.
        
        Args:
            config: DatasetConfig object
            min_doc_length: Minimum document length in words
            max_doc_length: Maximum document length in words
        """
        super().__init__(config)
        self.min_doc_length = min_doc_length
        self.max_doc_length = max_doc_length
    
    def generate(self) -> Dict[str, pd.DataFrame]:
        """
        Generate summarization dataset.
        
        Returns:
            Dict with DataFrames containing columns:
            ['document', 'summary', 'compression_ratio', 'difficulty']
        """
        pass

class QADataGenerator(BaseDatasetGenerator):
    """
    Generate synthetic Question-Answering datasets.
    
    Creates context-question-answer triplets.
    """
    
    def generate(self) -> Dict[str, pd.DataFrame]:
        """
        Generate QA dataset.
        
        Returns:
            Dict with DataFrames containing columns:
            ['context', 'question', 'answer', 'answer_type', 'difficulty']
        """
        pass

class InstructionFollowingGenerator(BaseDatasetGenerator):
    """
    Generate synthetic instruction-following datasets.
    
    Creates instruction-response pairs for general task completion.
    """
    
    def generate(self) -> Dict[str, pd.DataFrame]:
        """
        Generate instruction-following dataset.
        
        Returns:
            Dict with DataFrames containing columns:
            ['instruction', 'input', 'output', 'task_category', 'difficulty']
        """
        pass
```

#### `src/data/preprocessors/preprocessor.py`

```python
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
```

---

### 2. Model Interface Layer

#### `src/models/base.py`

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time

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
```

#### `src/models/pretrained/gpt5.py`

```python
from ..base import BaseModelWrapper, ModelInfo, GenerationConfig
from typing import List, Optional
import openai

class GPT5Wrapper(BaseModelWrapper):
    """Wrapper for GPT-5 (full model) via Azure OpenAI"""
    
    def __init__(self, deployment_name: str, endpoint: str, api_key: str):
        """
        Initialize GPT-5 wrapper.
        
        Args:
            deployment_name: Azure OpenAI deployment name
            endpoint: Azure OpenAI endpoint URL
            api_key: API key for authentication
        """
        super().__init__(model_name="gpt-5", config={
            "deployment_name": deployment_name,
            "endpoint": endpoint,
            "api_key": api_key
        })
        
    def load(self) -> None:
        """Initialize Azure OpenAI client"""
        pass
    
    def generate(self, prompt: str, generation_config: Optional[GenerationConfig] = None, **kwargs) -> str:
        """Generate using Azure OpenAI API"""
        pass

class GPT5MiniWrapper(BaseModelWrapper):
    """Wrapper for GPT-5 Mini"""
    
    def load(self) -> None:
        """Initialize for GPT-5 Mini"""
        pass
    
    def generate(self, prompt: str, generation_config: Optional[GenerationConfig] = None, **kwargs) -> str:
        """Generate using GPT-5 Mini"""
        pass

class GPT5NanoWrapper(BaseModelWrapper):
    """Wrapper for GPT-5 Nano"""
    
    def load(self) -> None:
        """Initialize for GPT-5 Nano"""
        pass
    
    def generate(self, prompt: str, generation_config: Optional[GenerationConfig] = None, **kwargs) -> str:
        """Generate using GPT-5 Nano"""
        pass
```

#### `src/models/pretrained/phi.py`

```python
from ..base import BaseModelWrapper, ModelInfo, GenerationConfig
from typing import Optional

class PhiWrapper(BaseModelWrapper):
    """
    Wrapper for Microsoft Phi models.
    
    Supports loading from HuggingFace or Azure ML model registry.
    """
    
    def __init__(self, model_variant: str = "phi-4", source: str = "huggingface"):
        """
        Initialize Phi model wrapper.
        
        Args:
            model_variant: Specific Phi model version
            source: 'huggingface' or 'azure_ml_registry'
        """
        super().__init__(model_name=model_variant, config={"source": source})
    
    def load(self) -> None:
        """Load Phi model using transformers library"""
        pass
    
    def generate(self, prompt: str, generation_config: Optional[GenerationConfig] = None, **kwargs) -> str:
        """Generate using Phi model"""
        pass
```

#### `src/models/finetuned/wrapper.py`

```python
from ..base import BaseModelWrapper, ModelInfo
from typing import Optional

class FinetunedModelWrapper(BaseModelWrapper):
    """
    Wrapper for finetuned models.
    
    Provides unified interface for models finetuned via:
    - Code-first approach
    - Azure AI Foundry
    """
    
    def __init__(self, 
                 base_model_name: str,
                 checkpoint_path: Optional[str] = None,
                 azure_model_name: Optional[str] = None):
        """
        Initialize finetuned model wrapper.
        
        Args:
            base_model_name: Original base model identifier
            checkpoint_path: Path to local checkpoint (for code-first)
            azure_model_name: Model name in Azure registry (for Foundry)
        """
        super().__init__(
            model_name=f"{base_model_name}-finetuned",
            config={
                "base_model": base_model_name,
                "checkpoint_path": checkpoint_path,
                "azure_model_name": azure_model_name
            }
        )
    
    def load_from_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model from local checkpoint.
        
        Args:
            checkpoint_path: Path to saved model checkpoint
        """
        pass
    
    def load_from_azure_registry(self, model_name: str) -> None:
        """
        Load model from Azure ML model registry.
        
        Args:
            model_name: Name of model in Azure registry
        """
        pass
    
    def load(self) -> None:
        """Load model based on available config"""
        pass
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate using finetuned model"""
        pass
```

---

### 3. Training Pipeline

#### `src/training/config.py`

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation)"""
    r: int = 8  # Rank
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

@dataclass
class QLoRAConfig:
    """Configuration for QLoRA (Quantized LoRA)"""
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    lora_config: LoRAConfig = field(default_factory=LoRAConfig)

@dataclass
class TrainingConfig:
    """Main training configuration"""
    # Model
    base_model: str
    model_max_length: int = 2048
    
    # Training hyperparameters
    learning_rate: float = 2e-5
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Optimization
    optimizer: str = "adamw"
    lr_scheduler: str = "cosine"
    
    # Strategy-specific
    strategy: str = "lora"  # 'full', 'lora', 'qlora'
    lora_config: Optional[LoRAConfig] = None
    qlora_config: Optional[QLoRAConfig] = None
    
    # Logging and checkpointing
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    output_dir: str = "./output"
    
    # Evaluation
    evaluation_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    
    # Hardware
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = False
    
    # Additional parameters
    seed: int = 42
    additional_params: Dict[str, Any] = field(default_factory=dict)
```

#### `src/training/trainer.py`

```python
from typing import Optional, Dict, Any
from .config import TrainingConfig, LoRAConfig
from ..data.preprocessors.preprocessor import DataPreprocessor
import pandas as pd

class FineTuner:
    """
    Code-first finetuning implementation.
    
    Supports multiple finetuning strategies:
    - Full finetuning
    - LoRA (Low-Rank Adaptation)
    - QLoRA (Quantized LoRA)
    """
    
    def __init__(self, 
                 base_model_name: str,
                 config: TrainingConfig,
                 preprocessor: Optional[DataPreprocessor] = None):
        """
        Initialize finetuner.
        
        Args:
            base_model_name: Name/path of base model
            config: TrainingConfig object
            preprocessor: Optional DataPreprocessor instance
        """
        self.base_model_name = base_model_name
        self.config = config
        self.preprocessor = preprocessor or DataPreprocessor()
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def prepare_model(self) -> None:
        """
        Load and prepare model for finetuning.
        
        Applies strategy-specific modifications:
        - Full: Load model as-is
        - LoRA: Add LoRA adapters
        - QLoRA: Quantize + add LoRA adapters
        """
        pass
    
    def prepare_data(self, 
                    train_dataset: pd.DataFrame,
                    val_dataset: pd.DataFrame,
                    task_type: str) -> Dict[str, Any]:
        """
        Prepare datasets for training.
        
        Args:
            train_dataset: Training data
            val_dataset: Validation data
            task_type: Type of task
            
        Returns:
            Dictionary with processed datasets
        """
        pass
    
    def train(self,
             train_dataset: pd.DataFrame,
             val_dataset: pd.DataFrame,
             task_type: str) -> Dict[str, Any]:
        """
        Execute training loop.
        
        Args:
            train_dataset: Training data
            val_dataset: Validation data
            task_type: Type of task
            
        Returns:
            Training metrics and results
        """
        pass
    
    def save_checkpoint(self, path: str) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Directory to save checkpoint
        """
        pass
    
    def load_checkpoint(self, path: str) -> None:
        """
        Load model from checkpoint.
        
        Args:
            path: Path to checkpoint directory
        """
        pass
    
    def evaluate(self, eval_dataset: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model on dataset.
        
        Args:
            eval_dataset: Evaluation data
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass
    
    def get_training_args(self) -> Any:
        """
        Convert TrainingConfig to framework-specific training arguments.
        
        Returns:
            Training arguments object (e.g., transformers.TrainingArguments)
        """
        pass
```

#### `src/training/strategies/base.py`

```python
from abc import ABC, abstractmethod
from typing import Any, Dict

class TrainingStrategy(ABC):
    """
    Abstract base class for training strategies.
    
    Different strategies (Full, LoRA, QLoRA) implement this interface.
    """
    
    @abstractmethod
    def prepare_model(self, model: Any, config: Dict[str, Any]) -> Any:
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
```

#### `src/training/strategies/lora.py`

```python
from .base import TrainingStrategy
from typing import Any, Dict

class LoRAFineTuning(TrainingStrategy):
    """
    LoRA (Low-Rank Adaptation) finetuning strategy.
    
    Uses PEFT library to add trainable low-rank adapters
    while keeping base model frozen.
    """
    
    def prepare_model(self, model: Any, config: Dict[str, Any]) -> Any:
        """
        Add LoRA adapters to model.
        
        Args:
            model: Base model
            config: LoRA configuration
            
        Returns:
            Model with LoRA adapters
        """
        pass
    
    def get_trainable_parameters(self, model: Any) -> int:
        """Get number of trainable LoRA parameters"""
        pass
```

---

### 4. Evaluation Framework

#### `src/evaluation/evaluator.py`

```python
from typing import List, Dict, Any
from ..models.base import BaseModelWrapper
from .metrics.base import Metric
import pandas as pd
from dataclasses import dataclass

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
```

#### `src/evaluation/metrics/base.py`

```python
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
```

#### `src/evaluation/metrics/accuracy.py`

```python
from .base import Metric
from typing import List

class AccuracyMetric(Metric):
    """Exact match accuracy for classification tasks"""
    
    def __init__(self):
        super().__init__(name="accuracy")
    
    def compute(self, predictions: List[str], references: List[str], **kwargs) -> float:
        """
        Compute accuracy as proportion of exact matches.
        
        Args:
            predictions: Model predictions
            references: Ground truth labels
            
        Returns:
            Accuracy score (0-1)
        """
        pass

class F1Metric(Metric):
    """F1 score for classification"""
    
    def __init__(self, average: str = "weighted"):
        """
        Initialize F1 metric.
        
        Args:
            average: Averaging strategy ('micro', 'macro', 'weighted')
        """
        super().__init__(name="f1")
        self.average = average
    
    def compute(self, predictions: List[str], references: List[str], **kwargs) -> float:
        """Compute F1 score"""
        pass
```

#### `src/evaluation/metrics/nlg_metrics.py`

```python
from .base import Metric
from typing import List

class BLEUMetric(Metric):
    """BLEU score for text generation"""
    
    def __init__(self, n_gram: int = 4):
        super().__init__(name=f"bleu-{n_gram}")
        self.n_gram = n_gram
    
    def compute(self, predictions: List[str], references: List[str], **kwargs) -> float:
        """Compute BLEU score"""
        pass

class ROUGEMetric(Metric):
    """ROUGE scores for summarization"""
    
    def __init__(self, rouge_type: str = "rouge-l"):
        """
        Initialize ROUGE metric.
        
        Args:
            rouge_type: Type of ROUGE ('rouge-1', 'rouge-2', 'rouge-l')
        """
        super().__init__(name=rouge_type)
        self.rouge_type = rouge_type
    
    def compute(self, predictions: List[str], references: List[str], **kwargs) -> float:
        """Compute ROUGE score"""
        pass
```

#### `src/evaluation/metrics/performance.py`

```python
from .base import Metric
from typing import List, Dict
import time

class LatencyMetric(Metric):
    """Measure generation latency"""
    
    def __init__(self):
        super().__init__(name="latency")
    
    def compute(self, predictions: List[str], references: List[str], **kwargs) -> float:
        """
        Compute average latency.
        
        Note: Requires 'generation_times' in kwargs
        
        Returns:
            Average latency in seconds
        """
        pass
    
    def get_statistics(self, generation_times: List[float]) -> Dict[str, float]:
        """
        Get detailed latency statistics.
        
        Args:
            generation_times: List of generation times
            
        Returns:
            Dict with mean, median, p95, p99
        """
        pass

class CostMetric(Metric):
    """Estimate generation cost"""
    
    def __init__(self, pricing: Dict[str, float]):
        """
        Initialize cost metric.
        
        Args:
            pricing: Dictionary with 'input_per_1k' and 'output_per_1k' costs
        """
        super().__init__(name="cost")
        self.pricing = pricing
    
    def compute(self, predictions: List[str], references: List[str], **kwargs) -> float:
        """
        Compute total cost.
        
        Requires 'input_tokens' and 'output_tokens' in kwargs
        
        Returns:
            Total cost in USD
        """
        pass
```

---

### 5. Azure AI Foundry Integration

#### `src/azure_foundry/client.py`

```python
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class AzureConfig:
    """Azure AI Foundry configuration"""
    subscription_id: str
    resource_group: str
    workspace_name: str
    endpoint: str
    api_key: str

class AzureFoundryClient:
    """
    Client for Azure AI Foundry operations.
    
    Provides programmatic access to:
    - Model catalog
    - Finetuning jobs
    - Model deployment
    - Endpoint management
    """
    
    def __init__(self, config: AzureConfig):
        """
        Initialize Azure Foundry client.
        
        Args:
            config: Azure configuration
        """
        self.config = config
        self.client = None
        
    def connect(self) -> None:
        """Establish connection to Azure AI Foundry"""
        pass
    
    def list_available_models(self, task_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available base models in catalog.
        
        Args:
            task_type: Optional filter by task type
            
        Returns:
            List of model metadata dictionaries
        """
        pass
    
    def create_finetuning_job(self,
                             base_model: str,
                             training_data_path: str,
                             validation_data_path: str,
                             config: Dict[str, Any]) -> str:
        """
        Create a finetuning job.
        
        Args:
            base_model: Base model identifier
            training_data_path: Path to training data
            validation_data_path: Path to validation data
            config: Finetuning configuration
            
        Returns:
            Job ID
        """
        pass
    
    def monitor_job(self, job_id: str) -> Dict[str, Any]:
        """
        Get status of finetuning job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job status dictionary
        """
        pass
    
    def deploy_model(self,
                    model_id: str,
                    deployment_name: str,
                    deployment_config: Dict[str, Any]) -> str:
        """
        Deploy finetuned model to endpoint.
        
        Args:
            model_id: Model to deploy
            deployment_name: Name for deployment
            deployment_config: Deployment configuration
            
        Returns:
            Deployment ID
        """
        pass
    
    def get_deployment_endpoint(self, deployment_id: str) -> str:
        """
        Get endpoint URL for deployment.
        
        Args:
            deployment_id: Deployment identifier
            
        Returns:
            Endpoint URL
        """
        pass
    
    def delete_deployment(self, deployment_id: str) -> None:
        """
        Delete a model deployment.
        
        Args:
            deployment_id: Deployment to delete
        """
        pass
```

#### `src/azure_foundry/job_manager.py`

```python
from typing import Dict, Any, Optional
from .client import AzureFoundryClient
import time

@dataclass
class JobSpec:
    """Specification for finetuning job"""
    job_name: str
    base_model: str
    training_data: str
    validation_data: str
    hyperparameters: Dict[str, Any]
    compute_config: Dict[str, Any]

class FineTuningJobManager:
    """
    Manage Azure AI Foundry finetuning jobs.
    
    Handles:
    - Job submission
    - Status tracking
    - Result retrieval
    - Resource cleanup
    """
    
    def __init__(self, client: AzureFoundryClient):
        """
        Initialize job manager.
        
        Args:
            client: AzureFoundryClient instance
        """
        self.client = client
        self.active_jobs = {}
    
    def submit_job(self, job_spec: JobSpec) -> str:
        """
        Submit finetuning job to Azure.
        
        Args:
            job_spec: Job specification
            
        Returns:
            Job ID
        """
        pass
    
    def track_job_status(self,
                        job_id: str,
                        poll_interval: int = 60,
                        timeout: int = 3600) -> Dict[str, Any]:
        """
        Track job status until completion.
        
        Args:
            job_id: Job identifier
            poll_interval: Seconds between status checks
            timeout: Maximum seconds to wait
            
        Returns:
            Final job status
        """
        pass
    
    def download_results(self,
                        job_id: str,
                        output_dir: str) -> None:
        """
        Download job results and logs.
        
        Args:
            job_id: Job identifier
            output_dir: Local directory for outputs
        """
        pass
    
    def cleanup_resources(self, job_id: str) -> None:
        """
        Clean up resources associated with job.
        
        Args:
            job_id: Job identifier
        """
        pass
    
    def get_job_metrics(self, job_id: str) -> Dict[str, Any]:
        """
        Get training metrics from completed job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Dictionary of training metrics
        """
        pass
```

---

### 6. Orchestration & Utilities

#### `src/orchestration/experiment.py`

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import yaml

@dataclass
class ExperimentConfig:
    """Configuration for an experiment"""
    experiment_name: str
    task_type: str
    models_to_test: List[str]
    finetuning_configs: List[Dict[str, Any]]
    dataset_config: Dict[str, Any]
    evaluation_metrics: List[str]
    use_azure_foundry: bool = False
    output_dir: str = "./experiments"
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file"""
        pass

class Experiment:
    """
    High-level experiment orchestration.
    
    Manages end-to-end experiment execution:
    1. Data generation
    2. Baseline evaluation
    3. Model finetuning
    4. Finetuned evaluation
    5. Results comparison
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.experiment_id = self._generate_experiment_id()
        self.results = {}
        
    def setup(self) -> None:
        """
        Setup experiment environment.
        
        - Create output directories
        - Initialize tracking
        - Validate configuration
        """
        pass
    
    def run(self) -> Dict[str, Any]:
        """
        Execute complete experiment.
        
        Returns:
            Dictionary with experiment results
        """
        pass
    
    def cleanup(self) -> None:
        """
        Cleanup experiment resources.
        
        - Delete temporary files
        - Release compute resources
        """
        pass
    
    def save_results(self, output_path: str) -> None:
        """
        Save experiment results.
        
        Args:
            output_path: Path to save results
        """
        pass
    
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment identifier"""
        pass
```

#### `src/orchestration/pipeline.py`

```python
from typing import Dict, Any, List
from ..data.generators.base import BaseDatasetGenerator
from ..models.base import BaseModelWrapper
from ..evaluation.evaluator import ModelEvaluator
from ..training.trainer import FineTuner

class ExperimentPipeline:
    """
    End-to-end experiment pipeline.
    
    Orchestrates all steps of the experimentation workflow.
    """
    
    def __init__(self, experiment_config: Dict[str, Any]):
        """
        Initialize pipeline.
        
        Args:
            experiment_config: Configuration dictionary
        """
        self.config = experiment_config
        self.data_generator = None
        self.models = {}
        self.evaluator = None
        self.finetuner = None
        
    def generate_data(self) -> Dict[str, Any]:
        """
        Generate synthetic dataset.
        
        Returns:
            Dictionary with train/val/test datasets
        """
        pass
    
    def run_baseline_evaluation(self,
                               models: List[BaseModelWrapper],
                               test_dataset: Any) -> Dict[str, Any]:
        """
        Evaluate pretrained models (baseline).
        
        Args:
            models: List of model wrappers
            test_dataset: Test dataset
            
        Returns:
            Dictionary of baseline results
        """
        pass
    
    def run_finetuning(self,
                      base_models: List[str],
                      train_dataset: Any,
                      val_dataset: Any) -> Dict[str, str]:
        """
        Finetune base models.
        
        Args:
            base_models: List of base model names
            train_dataset: Training dataset
            val_dataset: Validation dataset
            
        Returns:
            Dictionary mapping model name to checkpoint path
        """
        pass
    
    def run_finetuned_evaluation(self,
                                finetuned_models: Dict[str, BaseModelWrapper],
                                test_dataset: Any) -> Dict[str, Any]:
        """
        Evaluate finetuned models.
        
        Args:
            finetuned_models: Dictionary of finetuned model wrappers
            test_dataset: Test dataset
            
        Returns:
            Dictionary of evaluation results
        """
        pass
    
    def generate_report(self,
                       baseline_results: Dict[str, Any],
                       finetuned_results: Dict[str, Any]) -> str:
        """
        Generate comparison report.
        
        Args:
            baseline_results: Results from baseline evaluation
            finetuned_results: Results from finetuned evaluation
            
        Returns:
            Path to generated report
        """
        pass
```

#### `src/utils/tracking/tracker.py`

```python
from typing import Dict, Any, Optional
import mlflow

class ExperimentTracker:
    """
    Track experiments using MLflow or Azure ML.
    
    Logs:
    - Parameters
    - Metrics
    - Artifacts
    - Models
    """
    
    def __init__(self,
                 experiment_name: str,
                 tracking_uri: Optional[str] = None):
        """
        Initialize tracker.
        
        Args:
            experiment_name: Name of experiment
            tracking_uri: Optional tracking URI (MLflow/Azure ML)
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.run_id = None
        
    def start_run(self, run_name: Optional[str] = None) -> None:
        """
        Start a new tracking run.
        
        Args:
            run_name: Optional name for the run
        """
        pass
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters.
        
        Args:
            params: Dictionary of parameters
        """
        pass
    
    def log_metrics(self,
                   metrics: Dict[str, float],
                   step: Optional[int] = None) -> None:
        """
        Log metrics.
        
        Args:
            metrics: Dictionary of metric values
            step: Optional step number
        """
        pass
    
    def log_artifact(self, artifact_path: str, artifact_type: str = "file") -> None:
        """
        Log artifact (file, model, etc.).
        
        Args:
            artifact_path: Path to artifact
            artifact_type: Type of artifact
        """
        pass
    
    def log_model(self,
                 model: Any,
                 artifact_path: str,
                 **kwargs) -> None:
        """
        Log model.
        
        Args:
            model: Model object
            artifact_path: Path for artifact
            **kwargs: Additional arguments
        """
        pass
    
    def end_run(self) -> None:
        """End the current tracking run"""
        pass
```

---

## Configuration File Examples

### `config/models.yaml`

```yaml
# Model specifications
models:
  pretrained:
    gpt-5:
      type: "azure_openai"
      deployment: "gpt-5-deployment"
      endpoint: "${AZURE_OPENAI_ENDPOINT}"
      api_version: "2024-02-01"
      
    gpt-5-mini:
      type: "azure_openai"
      deployment: "gpt-5-mini-deployment"
      endpoint: "${AZURE_OPENAI_ENDPOINT}"
      
    gpt-5-nano:
      type: "azure_openai"
      deployment: "gpt-5-nano-deployment"
      endpoint: "${AZURE_OPENAI_ENDPOINT}"
      
    phi-4:
      type: "huggingface"
      model_id: "microsoft/phi-4"
      trust_remote_code: true
      
  finetuning_bases:
    - gpt-5-mini
    - phi-4
    
  pricing:
    gpt-5:
      input_per_1k: 0.01
      output_per_1k: 0.03
    gpt-5-mini:
      input_per_1k: 0.003
      output_per_1k: 0.01
    phi-4:
      input_per_1k: 0.0
      output_per_1k: 0.0
```

### `config/experiments.yaml`

```yaml
experiments:
  - name: "sentiment_classification_baseline"
    description: "Baseline evaluation on sentiment classification"
    task: "classification"
    dataset:
      num_samples: 5000
      difficulty: "medium"
      num_classes: 3
    models:
      - gpt-5
      - gpt-5-mini
      - phi-4
    metrics:
      - accuracy
      - f1
      - latency
      - cost
    output_dir: "./experiments/sentiment_baseline"
    
  - name: "sentiment_classification_finetuned"
    description: "Finetuned models on sentiment classification"
    task: "classification"
    base_experiments:
      - "sentiment_classification_baseline"
    base_models:
      - gpt-5-mini
      - phi-4
    finetuning:
      strategy: "lora"
      hyperparameters:
        learning_rate: 2e-5
        batch_size: 8
        num_epochs: 3
        lora_r: 8
        lora_alpha: 16
    use_azure_foundry: false
    output_dir: "./experiments/sentiment_finetuned"
```

### `config/tasks.yaml`

```yaml
tasks:
  classification:
    type: "classification"
    generator: "ClassificationDataGenerator"
    preprocessor: "classification_preprocessor"
    metrics:
      - accuracy
      - f1
      - precision
      - recall
    prompt_template: |
      Classify the following text into one of these categories: {categories}
      
      Text: {text}
      
      Category:
      
  summarization:
    type: "summarization"
    generator: "SummarizationDataGenerator"
    preprocessor: "summarization_preprocessor"
    metrics:
      - rouge-1
      - rouge-2
      - rouge-l
      - bleu
    prompt_template: |
      Summarize the following document in 2-3 sentences:
      
      {document}
      
      Summary:
      
  qa:
    type: "question_answering"
    generator: "QADataGenerator"
    preprocessor: "qa_preprocessor"
    metrics:
      - exact_match
      - f1
      - bleu
    prompt_template: |
      Context: {context}
      
      Question: {question}
      
      Answer:
```

---

## Development Roadmap

### Phase 1: Foundation (Week 1)
- [ ] Setup project structure
- [ ] Implement `ConfigManager`
- [ ] Create base classes (`BaseModelWrapper`, `Metric`, `BaseDatasetGenerator`)
- [ ] Setup `ExperimentTracker`
- [ ] Write unit tests for base classes

### Phase 2: Data Generation (Week 2)
- [ ] Implement `ClassificationDataGenerator`
- [ ] Implement `SummarizationDataGenerator`
- [ ] Implement `QADataGenerator`
- [ ] Implement `DataPreprocessor`
- [ ] Add data validation and quality checks
- [ ] Test data generation pipeline

### Phase 3: Model Integration (Week 3)
- [ ] Implement `GPT5Wrapper` (all variants)
- [ ] Implement `PhiWrapper`
- [ ] Test model loading and generation
- [ ] Implement `FinetunedModelWrapper`
- [ ] Add error handling and retries

### Phase 4: Evaluation (Week 4)
- [ ] Implement core metrics (Accuracy, F1, BLEU, ROUGE)
- [ ] Implement `LatencyMetric` and `CostMetric`
- [ ] Implement `ModelEvaluator`
- [ ] Implement `ModelComparator`
- [ ] Add visualization utilities

### Phase 5: Training (Week 5-6)
- [ ] Implement `FineTuner` class
- [ ] Implement `FullFineTuning` strategy
- [ ] Implement `LoRAFineTuning` strategy
- [ ] Implement `QLoRAFineTuning` strategy
- [ ] Add checkpoint management
- [ ] Test training pipeline

### Phase 6: Azure Foundry (Week 7)
- [ ] Implement `AzureFoundryClient`
- [ ] Implement `FineTuningJobManager`
- [ ] Test job submission and monitoring
- [ ] Add deployment capabilities

### Phase 7: Orchestration (Week 8)
- [ ] Implement `Experiment` class
- [ ] Implement `ExperimentPipeline`
- [ ] Add CLI scripts
- [ ] Create end-to-end tests

### Phase 8: Polish & Documentation (Week 9-10)
- [ ] Add comprehensive documentation
- [ ] Create example notebooks
- [ ] Performance optimization
- [ ] Integration testing
- [ ] User guide and tutorials

---

## Key Dependencies

```txt
# Core
python>=3.10
pyyaml>=6.0
pandas>=2.0.0
numpy>=1.24.0

# ML & Training
torch>=2.0.0
transformers>=4.35.0
peft>=0.7.0
bitsandbytes>=0.41.0
accelerate>=0.24.0
datasets>=2.14.0

# Azure
azure-ai-ml>=1.11.0
azure-identity>=1.14.0
openai>=1.3.0

# Evaluation
scikit-learn>=1.3.0
nltk>=3.8.0
rouge-score>=0.1.2
sacrebleu>=2.3.0

# Tracking
mlflow>=2.8.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.17.0

# Utilities
tqdm>=4.66.0
python-dotenv>=1.0.0
```

---

## Usage Examples

### CLI Usage

```bash
# Generate synthetic dataset
python scripts/generate_data.py \
  --task classification \
  --num-samples 5000 \
  --output-dir ./data/sentiment

# Run baseline evaluation
python scripts/evaluate_models.py \
  --config config/experiments.yaml \
  --experiment sentiment_classification_baseline

# Run finetuning
python scripts/run_experiment.py \
  --config config/experiments.yaml \
  --experiment sentiment_classification_finetuned

# Compare results
python scripts/compare_results.py \
  --baseline ./experiments/sentiment_baseline/results.json \
  --finetuned ./experiments/sentiment_finetuned/results.json
```

### Python API Usage

```python
from src.orchestration.experiment import Experiment, ExperimentConfig
from src.utils.tracking.tracker import ExperimentTracker

# Load configuration
config = ExperimentConfig.from_yaml("config/experiments.yaml")

# Create experiment
experiment = Experiment(config)

# Setup tracking
tracker = ExperimentTracker(experiment_name=config.experiment_name)

# Run experiment
experiment.setup()
results = experiment.run()

# Save results
experiment.save_results("./experiments/results")
```

---

## Notes for LLM Agents

When implementing this specification:

1. **Follow the structure exactly** - maintain the directory hierarchy
2. **Implement all abstract methods** - ensure base classes are fully inherited
3. **Add comprehensive docstrings** - follow Google/NumPy docstring format
4. **Include type hints** - use typing module for all function signatures
5. **Add error handling** - wrap external API calls in try-except blocks
6. **Log extensively** - use Python logging module
7. **Write unit tests** - create tests for each module
8. **Keep it modular** - each component should be independently testable
9. **Use configuration files** - avoid hardcoding values
10. **Document assumptions** - add comments for non-obvious logic

This specification provides the complete blueprint for building the LLM finetuning experimentation framework. Each section can be implemented incrementally by LLM agents following the development roadmap.
