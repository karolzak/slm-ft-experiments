import unittest
from dataclasses import asdict

from src.data.generators.base import BaseDatasetGenerator, DatasetConfig
from src.models.base import BaseModelWrapper, ModelInfo, GenerationConfig
from src.training.config import TrainingConfig, LoRAConfig, QLoRAConfig
from src.training.strategies.base import TrainingStrategy
from src.evaluation.metrics.base import Metric
from src.evaluation.evaluator import ModelEvaluator, EvaluationResults
from src.data.preprocessors.preprocessor import DataPreprocessor
from src.azure_foundry.client import AzureFoundryClient, AzureConfig


class TestDatasetConfig(unittest.TestCase):
    def test_dataset_config_creation(self):
        config = DatasetConfig(
            num_samples=100,
            task_type="classification",
            difficulty_level="medium"
        )
        self.assertEqual(config.num_samples, 100)
        self.assertEqual(config.task_type, "classification")
        self.assertEqual(config.difficulty_level, "medium")
        self.assertEqual(config.train_split, 0.8)
        self.assertEqual(config.seed, 42)


class TestModelInfo(unittest.TestCase):
    def test_model_info_creation(self):
        info = ModelInfo(
            name="gpt-5",
            type="pretrained",
            base_model=None,
            parameters=1000000000,
            context_length=128000,
            deployment_endpoint="https://example.com"
        )
        self.assertEqual(info.name, "gpt-5")
        self.assertEqual(info.type, "pretrained")
        self.assertEqual(info.parameters, 1000000000)


class TestGenerationConfig(unittest.TestCase):
    def test_generation_config_defaults(self):
        config = GenerationConfig()
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.max_tokens, 512)
        self.assertEqual(config.top_p, 1.0)
    
    def test_generation_config_custom(self):
        config = GenerationConfig(temperature=0.5, max_tokens=1024)
        self.assertEqual(config.temperature, 0.5)
        self.assertEqual(config.max_tokens, 1024)


class TestLoRAConfig(unittest.TestCase):
    def test_lora_config_defaults(self):
        config = LoRAConfig()
        self.assertEqual(config.r, 8)
        self.assertEqual(config.lora_alpha, 16)
        self.assertEqual(config.lora_dropout, 0.1)
        self.assertIn("q_proj", config.target_modules)
        self.assertIn("v_proj", config.target_modules)


class TestQLoRAConfig(unittest.TestCase):
    def test_qlora_config_defaults(self):
        config = QLoRAConfig()
        self.assertTrue(config.load_in_4bit)
        self.assertEqual(config.bnb_4bit_compute_dtype, "float16")
        self.assertIsInstance(config.lora_config, LoRAConfig)


class TestTrainingConfig(unittest.TestCase):
    def test_training_config_creation(self):
        config = TrainingConfig(base_model="phi-4")
        self.assertEqual(config.base_model, "phi-4")
        self.assertEqual(config.learning_rate, 2e-5)
        self.assertEqual(config.strategy, "lora")
        self.assertEqual(config.seed, 42)


class TestAzureConfig(unittest.TestCase):
    def test_azure_config_creation(self):
        config = AzureConfig(
            subscription_id="sub-123",
            resource_group="rg-test",
            workspace_name="ws-test",
            endpoint="https://example.com",
            api_key="key123"
        )
        self.assertEqual(config.subscription_id, "sub-123")
        self.assertEqual(config.resource_group, "rg-test")


class TestDataPreprocessor(unittest.TestCase):
    def test_preprocessor_creation(self):
        preprocessor = DataPreprocessor()
        self.assertIsNone(preprocessor.tokenizer)
    
    def test_preprocessor_with_tokenizer(self):
        tokenizer = "mock_tokenizer"
        preprocessor = DataPreprocessor(tokenizer=tokenizer)
        self.assertEqual(preprocessor.tokenizer, tokenizer)


class TestAzureFoundryClient(unittest.TestCase):
    def test_client_creation(self):
        config = AzureConfig(
            subscription_id="sub-123",
            resource_group="rg-test",
            workspace_name="ws-test",
            endpoint="https://example.com",
            api_key="key123"
        )
        client = AzureFoundryClient(config)
        self.assertEqual(client.config, config)
        self.assertIsNone(client.client)


class TestEvaluationResults(unittest.TestCase):
    def test_evaluation_results_creation(self):
        results = EvaluationResults(
            model_name="test-model",
            task_type="classification",
            metrics={"accuracy": 0.95},
            predictions=["pred1", "pred2"],
            references=["ref1", "ref2"],
            latency_stats={"mean": 0.5},
            cost_estimate=0.01,
            metadata={"info": "test"}
        )
        self.assertEqual(results.model_name, "test-model")
        self.assertEqual(results.metrics["accuracy"], 0.95)
        self.assertEqual(len(results.predictions), 2)


if __name__ == "__main__":
    unittest.main()
