from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation)"""
    r: int = 8  # Rank
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
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
    additional_params: dict[str, Any] = field(default_factory=dict)
