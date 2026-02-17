"""
Tests for KYC/SOW Data Generator.
"""

import unittest
import json
import os
from unittest.mock import Mock, patch, MagicMock

from src.data.generators.base import DatasetConfig
from src.data.generators.kyc_sow_generator import KYCSOWDataGenerator


class TestKYCSOWDataGenerator(unittest.TestCase):
    """Test cases for KYCSOWDataGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = DatasetConfig(
            num_samples=10,
            task_type="kyc_sow",
            difficulty_level="medium",
            train_split=0.7,
            val_split=0.15,
            test_split=0.15,
            seed=42
        )
    
    @patch.dict(os.environ, {
        'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com/',
        'AZURE_OPENAI_API_KEY': 'test-key'
    })
    @patch('src.data.generators.kyc_sow_generator.AzureOpenAI')
    def test_generator_initialization(self, mock_client_class):
        """Test generator initialization."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        generator = KYCSOWDataGenerator(self.config)
        self.assertEqual(generator.config, self.config)
        self.assertEqual(generator.seed, 42)
        self.assertIsNotNone(generator.client)
    
    def test_generator_initialization_without_credentials(self):
        """Test that generator fails without Azure credentials."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as context:
                KYCSOWDataGenerator(self.config)
            self.assertIn("Azure OpenAI credentials are required", str(context.exception))
    
    @patch.dict(os.environ, {
        'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com/',
        'AZURE_OPENAI_API_KEY': 'test-key'
    })
    @patch('src.data.generators.kyc_sow_generator.AzureOpenAI')
    def test_scenario_templates(self, mock_client_class):
        """Test that scenario templates are properly loaded."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        generator = KYCSOWDataGenerator(self.config)
        scenarios = generator.scenarios  # Now loaded from config file
        
        # Check we have sufficient scenario variety
        self.assertGreaterEqual(len(scenarios), 10, "Should have at least 10 scenario types")
        
        # Check each scenario has required fields
        for scenario in scenarios:
            self.assertIn("scenario_type", scenario)
            self.assertIn("description", scenario)
            self.assertIn("typical_amounts", scenario)
            self.assertIn("currencies", scenario)
            self.assertIn("risk_profile", scenario)
            
            # Check difficulty levels in amounts
            self.assertIn("easy", scenario["typical_amounts"])
            self.assertIn("medium", scenario["typical_amounts"])
            self.assertIn("hard", scenario["typical_amounts"])
    
    @patch.dict(os.environ, {
        'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com/',
        'AZURE_OPENAI_API_KEY': 'test-key'
    })
    @patch('src.data.generators.kyc_sow_generator.AzureOpenAI')
    def test_generate_with_mock_llm(self, mock_client_class):
        """Test dataset generation with mocked LLM."""
        # Mock the OpenAI client
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock note generation response
        mock_note_response = MagicMock()
        mock_note_response.choices = [MagicMock()]
        mock_note_response.choices[0].message.content = """Meeting with Sarah Thompson, 34, Senior Software Engineer.
        
Deposit of GBP 45,000 from employment savings. Salary £95,000 per annum. Provided payslips and P60. Low risk."""
        
        # Mock extraction response
        mock_extract_response = MagicMock()
        mock_extract_response.choices = [MagicMock()]
        mock_extract_response.choices[0].message.content = json.dumps({
            "customer_name": "Sarah Thompson",
            "occupation": "Senior Software Engineer",
            "wealth_sources": [
                {
                    "source_type": "employment",
                    "amount": 45000,
                    "currency": "GBP",
                    "description": "Employment savings"
                }
            ],
            "total_amount": 45000,
            "risk_level": "low",
            "documents_provided": ["payslips", "P60"],
            "flags": []
        })
        
        # Alternate between note and extraction responses
        mock_client.chat.completions.create.side_effect = [
            mock_note_response, mock_extract_response
        ] * 10  # For 10 samples
        
        # Create generator with small sample size
        small_config = DatasetConfig(
            num_samples=2,
            task_type="kyc_sow",
            difficulty_level="medium",
            seed=42
        )
        generator = KYCSOWDataGenerator(small_config)
        generator.client = mock_client
        
        # Generate dataset
        dataset = generator.generate()
        
        # Verify structure
        self.assertIn("train", dataset)
        self.assertIn("val", dataset)
        self.assertIn("test", dataset)
        
        # Verify columns
        for split in dataset.values():
            self.assertIn("note_text", split.columns)
            self.assertIn("structured_output", split.columns)
            self.assertIn("scenario_type", split.columns)
            self.assertIn("difficulty", split.columns)
    
    def test_validate_good_dataset(self):
        """Test validation with a properly structured dataset."""
        import pandas as pd
        
        # Create valid test data
        valid_data = {
            "note_text": "Meeting with John Smith. Employment income of GBP 50,000.",
            "structured_output": json.dumps({
                "customer_name": "John Smith",
                "occupation": "Engineer",
                "wealth_sources": [
                    {
                        "source_type": "employment",
                        "amount": 50000,
                        "currency": "GBP",
                        "description": "Salary savings"
                    }
                ],
                "total_amount": 50000,
                "risk_level": "low",
                "documents_provided": ["payslips"],
                "flags": []
            }),
            "scenario_type": "employment",
            "difficulty": "easy"
        }
        
        # Create dataset with variety
        data_samples = []
        scenario_types = ["employment", "business_sale", "inheritance", "property_sale", "investment", "crypto"]
        risk_levels = ["low", "medium", "high"]
        
        for i in range(20):
            sample = valid_data.copy()
            sample["note_text"] = f"Meeting {i}. Different content for variety."
            
            # Vary scenario and risk
            scenario = scenario_types[i % len(scenario_types)]
            risk = risk_levels[i % len(risk_levels)]
            
            structured = json.loads(sample["structured_output"])
            structured["risk_level"] = risk
            structured["wealth_sources"][0]["source_type"] = scenario
            sample["structured_output"] = json.dumps(structured)
            sample["scenario_type"] = scenario
            
            data_samples.append(sample)
        
        df = pd.DataFrame(data_samples)
        
        # Split dataset
        dataset = {
            "train": df[:14],
            "val": df[14:17],
            "test": df[17:]
        }
        
        with patch.dict(os.environ, {
            'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com/',
            'AZURE_OPENAI_API_KEY': 'test-key'
        }):
            with patch('src.data.generators.kyc_sow_generator.AzureOpenAI'):
                generator = KYCSOWDataGenerator(self.config)
                self.assertTrue(generator.validate(dataset))
    
    def test_validate_missing_columns(self):
        """Test validation fails with missing columns."""
        import pandas as pd
        
        # Create dataset with missing column
        df = pd.DataFrame({
            "note_text": ["test"],
            "structured_output": [json.dumps({"customer_name": "test", "wealth_sources": [{}], "risk_level": "low"})]
            # Missing scenario_type and difficulty
        })
        
        dataset = {"train": df, "val": df, "test": df}
        
        with patch.dict(os.environ, {
            'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com/',
            'AZURE_OPENAI_API_KEY': 'test-key'
        }):
            with patch('src.data.generators.kyc_sow_generator.AzureOpenAI'):
                generator = KYCSOWDataGenerator(self.config)
                self.assertFalse(generator.validate(dataset))
    
    def test_validate_invalid_json(self):
        """Test validation fails with invalid JSON."""
        import pandas as pd
        
        df = pd.DataFrame({
            "note_text": ["test"],
            "structured_output": ["invalid json {{}"],
            "scenario_type": ["employment"],
            "difficulty": ["easy"]
        })
        
        dataset = {"train": df, "val": df, "test": df}
        
        with patch.dict(os.environ, {
            'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com/',
            'AZURE_OPENAI_API_KEY': 'test-key'
        }):
            with patch('src.data.generators.kyc_sow_generator.AzureOpenAI'):
                generator = KYCSOWDataGenerator(self.config)
                self.assertFalse(generator.validate(dataset))
    
    def test_validate_missing_required_fields(self):
        """Test validation fails with missing required JSON fields."""
        import pandas as pd
        
        # Missing customer_name
        df = pd.DataFrame({
            "note_text": ["test"],
            "structured_output": [json.dumps({"occupation": "test"})],
            "scenario_type": ["employment"],
            "difficulty": ["easy"]
        })
        
        dataset = {"train": df, "val": df, "test": df}
        
        with patch.dict(os.environ, {
            'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com/',
            'AZURE_OPENAI_API_KEY': 'test-key'
        }):
            with patch('src.data.generators.kyc_sow_generator.AzureOpenAI'):
                generator = KYCSOWDataGenerator(self.config)
                self.assertFalse(generator.validate(dataset))
    
    def test_validate_empty_wealth_sources(self):
        """Test validation fails with empty wealth_sources."""
        import pandas as pd
        
        df = pd.DataFrame({
            "note_text": ["test"],
            "structured_output": [json.dumps({
                "customer_name": "test",
                "wealth_sources": [],  # Empty array
                "risk_level": "low"
            })],
            "scenario_type": ["employment"],
            "difficulty": ["easy"]
        })
        
        dataset = {"train": df, "val": df, "test": df}
        
        with patch.dict(os.environ, {
            'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com/',
            'AZURE_OPENAI_API_KEY': 'test-key'
        }):
            with patch('src.data.generators.kyc_sow_generator.AzureOpenAI'):
                generator = KYCSOWDataGenerator(self.config)
                self.assertFalse(generator.validate(dataset))
    
    def test_save_and_load(self):
        """Test save and load functionality."""
        import pandas as pd
        import tempfile
        import shutil
        
        # Create test dataset
        df = pd.DataFrame({
            "note_text": ["test note 1", "test note 2"],
            "structured_output": [
                json.dumps({"customer_name": "test1", "wealth_sources": [{}], "risk_level": "low"}),
                json.dumps({"customer_name": "test2", "wealth_sources": [{}], "risk_level": "medium"})
            ],
            "scenario_type": ["employment", "business_sale"],
            "difficulty": ["easy", "medium"]
        })
        
        dataset = {
            "train": df[:1],
            "val": df[1:2],
            "test": df[1:2]
        }
        
        # Save to temp directory
        temp_dir = tempfile.mkdtemp()
        try:
            with patch.dict(os.environ, {
                'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com/',
                'AZURE_OPENAI_API_KEY': 'test-key'
            }):
                with patch('src.data.generators.kyc_sow_generator.AzureOpenAI'):
                    generator = KYCSOWDataGenerator(self.config)
                    generator.save(dataset, temp_dir)
            
            # Verify files exist
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "train.csv")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "val.csv")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "test.csv")))
            
            # Load back
            loaded = generator.load(temp_dir)
            
            # Verify loaded data
            self.assertIn("train", loaded)
            self.assertIn("val", loaded)
            self.assertIn("test", loaded)
            self.assertEqual(len(loaded["train"]), 1)
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_get_statistics(self):
        """Test statistics computation."""
        import pandas as pd
        
        df = pd.DataFrame({
            "note_text": ["note1", "note2", "note3", "note4"],
            "structured_output": [
                json.dumps({"customer_name": "test", "wealth_sources": [{}], "risk_level": "low"}),
                json.dumps({"customer_name": "test", "wealth_sources": [{}], "risk_level": "low"}),
                json.dumps({"customer_name": "test", "wealth_sources": [{}], "risk_level": "high"}),
                json.dumps({"customer_name": "test", "wealth_sources": [{}], "risk_level": "medium"})
            ],
            "scenario_type": ["employment", "employment", "crypto", "business_sale"],
            "difficulty": ["easy", "medium", "hard", "medium"]
        })
        
        dataset = {
            "train": df[:2],
            "val": df[2:3],
            "test": df[3:]
        }
        
        with patch.dict(os.environ, {
            'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com/',
            'AZURE_OPENAI_API_KEY': 'test-key'
        }):
            with patch('src.data.generators.kyc_sow_generator.AzureOpenAI'):
                generator = KYCSOWDataGenerator(self.config)
                stats = generator.get_statistics(dataset)
        
        self.assertEqual(stats["total_samples"], 4)
        self.assertEqual(stats["splits"]["train"], 2)
        self.assertEqual(stats["splits"]["val"], 1)
        self.assertEqual(stats["splits"]["test"], 1)
        self.assertIn("scenario_distribution", stats)
        self.assertIn("difficulty_distribution", stats)
        self.assertIn("risk_level_distribution", stats)


if __name__ == "__main__":
    unittest.main()
