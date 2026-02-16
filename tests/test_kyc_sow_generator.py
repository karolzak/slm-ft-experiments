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
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        generator = KYCSOWDataGenerator(self.config)
        self.assertEqual(generator.config, self.config)
        self.assertEqual(generator.seed, 42)
        self.assertIsNotNone(generator.client)
    
    def test_scenario_templates(self):
        """Test that scenario templates are properly defined."""
        generator = KYCSOWDataGenerator(self.config)
        scenarios = generator._get_scenario_templates()
        
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
    
    def test_fallback_note_generation(self):
        """Test fallback note generation."""
        generator = KYCSOWDataGenerator(self.config)
        scenarios = generator._get_scenario_templates()
        scenario = scenarios[0]
        
        # Test easy difficulty
        note = generator._generate_fallback_note(scenario, "easy", 50000, "GBP")
        self.assertIsInstance(note, str)
        self.assertGreater(len(note), 50)
        self.assertIn("GBP", note)
        self.assertIn("50,000", note)
        
        # Test medium difficulty
        note = generator._generate_fallback_note(scenario, "medium", 100000, "USD")
        self.assertIsInstance(note, str)
        self.assertIn("USD", note)
        
        # Test hard difficulty
        note = generator._generate_fallback_note(scenario, "hard", 200000, "EUR")
        self.assertIsInstance(note, str)
        self.assertIn("EUR", note)
    
    def test_fallback_data_extraction(self):
        """Test fallback data extraction."""
        generator = KYCSOWDataGenerator(self.config)
        
        # Test with low risk notes
        notes = "Meeting with John Smith. Low risk assessment."
        data = generator._extract_fallback_data(notes)
        
        self.assertIsInstance(data, dict)
        self.assertIn("customer_name", data)
        self.assertIn("wealth_sources", data)
        self.assertIn("risk_level", data)
        self.assertIsInstance(data["wealth_sources"], list)
        self.assertGreater(len(data["wealth_sources"]), 0)
        
        # Test with high risk notes
        notes = "Complex case with high risk assessment."
        data = generator._extract_fallback_data(notes)
        self.assertEqual(data["risk_level"], "high")
    
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
            self.assertIn("notes", split.columns)
            self.assertIn("structured_output", split.columns)
            self.assertIn("scenario_type", split.columns)
            self.assertIn("difficulty", split.columns)
    
    def test_validate_good_dataset(self):
        """Test validation with a properly structured dataset."""
        import pandas as pd
        
        # Create valid test data
        valid_data = {
            "notes": "Meeting with John Smith. Employment income of GBP 50,000.",
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
            sample["notes"] = f"Meeting {i}. Different content for variety."
            
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
        
        generator = KYCSOWDataGenerator(self.config)
        self.assertTrue(generator.validate(dataset))
    
    def test_validate_missing_columns(self):
        """Test validation fails with missing columns."""
        import pandas as pd
        
        # Create dataset with missing column
        df = pd.DataFrame({
            "notes": ["test"],
            "structured_output": [json.dumps({"customer_name": "test", "wealth_sources": [{}], "risk_level": "low"})]
            # Missing scenario_type and difficulty
        })
        
        dataset = {"train": df, "val": df, "test": df}
        
        generator = KYCSOWDataGenerator(self.config)
        self.assertFalse(generator.validate(dataset))
    
    def test_validate_invalid_json(self):
        """Test validation fails with invalid JSON."""
        import pandas as pd
        
        df = pd.DataFrame({
            "notes": ["test"],
            "structured_output": ["invalid json {{}"],
            "scenario_type": ["employment"],
            "difficulty": ["easy"]
        })
        
        dataset = {"train": df, "val": df, "test": df}
        
        generator = KYCSOWDataGenerator(self.config)
        self.assertFalse(generator.validate(dataset))
    
    def test_validate_missing_required_fields(self):
        """Test validation fails with missing required JSON fields."""
        import pandas as pd
        
        # Missing customer_name
        df = pd.DataFrame({
            "notes": ["test"],
            "structured_output": [json.dumps({"occupation": "test"})],
            "scenario_type": ["employment"],
            "difficulty": ["easy"]
        })
        
        dataset = {"train": df, "val": df, "test": df}
        
        generator = KYCSOWDataGenerator(self.config)
        self.assertFalse(generator.validate(dataset))
    
    def test_validate_empty_wealth_sources(self):
        """Test validation fails with empty wealth_sources."""
        import pandas as pd
        
        df = pd.DataFrame({
            "notes": ["test"],
            "structured_output": [json.dumps({
                "customer_name": "test",
                "wealth_sources": [],  # Empty array
                "risk_level": "low"
            })],
            "scenario_type": ["employment"],
            "difficulty": ["easy"]
        })
        
        dataset = {"train": df, "val": df, "test": df}
        
        generator = KYCSOWDataGenerator(self.config)
        self.assertFalse(generator.validate(dataset))
    
    def test_save_and_load(self):
        """Test save and load functionality."""
        import pandas as pd
        import tempfile
        import shutil
        
        # Create test dataset
        df = pd.DataFrame({
            "notes": ["test note 1", "test note 2"],
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
            "notes": ["note1", "note2", "note3", "note4"],
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
