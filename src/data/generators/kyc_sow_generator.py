"""
KYC/SOW Data Generator for financial services ML applications.

Generates synthetic account manager notes and structured JSON extractions
for Know Your Customer (KYC) and Source of Wealth (SOW) extraction tasks.
"""

import os
import json
import random
from typing import Any
from collections import Counter
import pandas as pd
from openai import AzureOpenAI

from .base import BaseDatasetGenerator, DatasetConfig


class KYCSOWDataGenerator(BaseDatasetGenerator):
    """
    Generator for KYC/SOW synthetic datasets.
    
    Creates realistic account manager notes documenting customer interactions
    about the origin of funds, then extracts structured JSON information.
    
    Uses a two-stage LLM generation approach:
    1. Generate free-text notes based on scenario templates
    2. Extract structured JSON from the generated notes
    """
    
    def __init__(self, config: DatasetConfig):
        """
        Initialize KYC/SOW data generator.
        
        Args:
            config: DatasetConfig object with generation parameters
        """
        super().__init__(config)
        
        # Check if Azure OpenAI is configured
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        
        # Initialize Azure OpenAI client only if credentials are provided
        if endpoint and api_key:
            self.client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
            )
            self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
            self.use_llm = True
        else:
            self.client = None
            self.deployment_name = None
            self.use_llm = False
        
        # Set random seed for reproducibility
        random.seed(self.seed)
    
    def _get_scenario_templates(self) -> list[dict[str, Any]]:
        """
        Define scenario templates for various wealth source types.
        
        Returns:
            List of scenario template dictionaries
        """
        return [
            {
                "scenario_type": "employment",
                "description": "Employed professional with salary-based income",
                "typical_amounts": {"easy": (10000, 50000), "medium": (50000, 150000), "hard": (150000, 500000)},
                "currencies": ["GBP", "USD", "EUR"],
                "risk_profile": "low"
            },
            {
                "scenario_type": "business_sale",
                "description": "Business owner who sold their company",
                "typical_amounts": {"easy": (100000, 500000), "medium": (500000, 2000000), "hard": (2000000, 10000000)},
                "currencies": ["GBP", "USD", "EUR"],
                "risk_profile": "medium"
            },
            {
                "scenario_type": "inheritance",
                "description": "Inherited wealth from estate transfer",
                "typical_amounts": {"easy": (50000, 200000), "medium": (200000, 1000000), "hard": (1000000, 5000000)},
                "currencies": ["GBP", "USD", "EUR", "CHF"],
                "risk_profile": "low"
            },
            {
                "scenario_type": "property_sale",
                "description": "Real estate sale or property investment",
                "typical_amounts": {"easy": (100000, 300000), "medium": (300000, 1000000), "hard": (1000000, 5000000)},
                "currencies": ["GBP", "USD", "EUR"],
                "risk_profile": "low"
            },
            {
                "scenario_type": "investment",
                "description": "Investment returns from stocks, bonds, or funds",
                "typical_amounts": {"easy": (20000, 100000), "medium": (100000, 500000), "hard": (500000, 3000000)},
                "currencies": ["GBP", "USD", "EUR"],
                "risk_profile": "low"
            },
            {
                "scenario_type": "pension",
                "description": "Pension lump sum or retirement funds",
                "typical_amounts": {"easy": (50000, 150000), "medium": (150000, 500000), "hard": (500000, 2000000)},
                "currencies": ["GBP", "USD", "EUR"],
                "risk_profile": "low"
            },
            {
                "scenario_type": "crypto",
                "description": "Cryptocurrency trading or investment gains",
                "typical_amounts": {"easy": (10000, 50000), "medium": (50000, 500000), "hard": (500000, 5000000)},
                "currencies": ["GBP", "USD", "EUR"],
                "risk_profile": "high"
            },
            {
                "scenario_type": "startup_exit",
                "description": "Entrepreneur who exited a startup or received venture funding",
                "typical_amounts": {"easy": (200000, 500000), "medium": (500000, 3000000), "hard": (3000000, 20000000)},
                "currencies": ["GBP", "USD", "EUR"],
                "risk_profile": "medium"
            },
            {
                "scenario_type": "gift",
                "description": "Gift from family member or third party",
                "typical_amounts": {"easy": (5000, 30000), "medium": (30000, 100000), "hard": (100000, 500000)},
                "currencies": ["GBP", "USD", "EUR"],
                "risk_profile": "medium"
            },
            {
                "scenario_type": "international_transfer",
                "description": "Cross-border funds with multiple jurisdictions",
                "typical_amounts": {"easy": (50000, 200000), "medium": (200000, 1000000), "hard": (1000000, 5000000)},
                "currencies": ["GBP", "USD", "EUR", "CHF", "SGD"],
                "risk_profile": "high"
            },
            {
                "scenario_type": "rental_income",
                "description": "Accumulated rental income from property portfolio",
                "typical_amounts": {"easy": (20000, 80000), "medium": (80000, 300000), "hard": (300000, 1000000)},
                "currencies": ["GBP", "USD", "EUR"],
                "risk_profile": "low"
            },
            {
                "scenario_type": "complex_portfolio",
                "description": "High-net-worth individual with multiple wealth sources",
                "typical_amounts": {"easy": (500000, 1000000), "medium": (1000000, 5000000), "hard": (5000000, 50000000)},
                "currencies": ["GBP", "USD", "EUR", "CHF"],
                "risk_profile": "high"
            }
        ]
    
    def _generate_note_with_llm(
        self,
        scenario: dict[str, Any],
        difficulty: str,
        amount: int,
        currency: str
    ) -> str:
        """
        Generate realistic account manager notes using LLM.
        
        Args:
            scenario: Scenario template dictionary
            difficulty: Difficulty level ('easy', 'medium', 'hard')
            amount: Amount of funds
            currency: Currency code
            
        Returns:
            Generated free-text account manager notes
        """
        # Adjust complexity based on difficulty
        if difficulty == "easy":
            complexity_instruction = """
- Single, clear wealth source
- Complete documentation provided
- Straightforward explanation
- No concerns or red flags
- Professional, structured writing style
"""
        elif difficulty == "medium":
            complexity_instruction = """
- 2-3 wealth sources
- Some missing details or documentation gaps
- Moderately complex situation
- Minor concerns but manageable
- Mix of formal and casual tone
"""
        else:  # hard
            complexity_instruction = """
- Multiple wealth sources (3+)
- Incomplete information or documentation
- Inconsistencies in amounts or dates
- Red flags or concerns present
- Multiple jurisdictions or offshore elements
- Vague or evasive explanations
"""
        
        prompt = f"""You are an experienced bank account manager writing notes about a customer meeting regarding their source of wealth.

Scenario: {scenario['description']}
Amount: {currency} {amount:,}
Difficulty level: {difficulty}

Write realistic, natural account manager notes that include:
- Customer name and background
- Occupation or business details
- Source(s) of wealth with specific amounts
- Documents provided or requested
- Your observations and assessment
- Any concerns or red flags (if applicable)

Style guidelines for {difficulty} difficulty:
{complexity_instruction}

Write the notes in a natural, authentic style as if you're a real account manager documenting this meeting. Do not use markdown formatting or headers - just write the notes naturally.
"""
        
        # Use fallback if LLM is not available
        if not self.use_llm or not self.client:
            return self._generate_fallback_note(scenario, difficulty, amount, currency)
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.8
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            # Fallback to template-based generation if API fails
            return self._generate_fallback_note(scenario, difficulty, amount, currency)
    
    def _generate_fallback_note(
        self,
        scenario: dict[str, Any],
        difficulty: str,
        amount: int,
        currency: str
    ) -> str:
        """
        Generate a simple template-based note as fallback.
        
        Args:
            scenario: Scenario template dictionary
            difficulty: Difficulty level
            amount: Amount of funds
            currency: Currency code
            
        Returns:
            Template-based account manager notes
        """
        names = ["John Smith", "Sarah Thompson", "Michael Chen", "Emma Wilson", "David Brown"]
        name = random.choice(names)
        
        if difficulty == "easy":
            note = f"""Meeting with {name} regarding deposit of {currency} {amount:,}.

Source of funds: {scenario['description']}. All documentation provided and verified. Clear explanation of source. No concerns identified. Risk assessment: Low. Approved for standard processing."""
        elif difficulty == "medium":
            note = f"""Call with {name} about {currency} {amount:,} deposit.

{scenario['description']}. Some documentation pending. Need to follow up on additional verification. Minor questions but overall acceptable. Risk assessment: Medium."""
        else:
            note = f"""Complex case: {name}, {currency} {amount:,}.

{scenario['description']}. Multiple sources, incomplete documentation. Some inconsistencies noted. Requires enhanced due diligence. Risk assessment: High. Hold pending further review."""
        
        return note
    
    def _extract_structured_data_with_llm(self, notes: str) -> dict[str, Any]:
        """
        Extract structured JSON from account manager notes using LLM.
        
        Args:
            notes: Free-text account manager notes
            
        Returns:
            Dictionary with structured extracted data
        """
        schema = """{
  "customer_name": "string",
  "occupation": "string or null",
  "wealth_sources": [
    {
      "source_type": "string (e.g., 'employment', 'business_sale', 'inheritance', 'investment', 'property_sale', 'pension', 'crypto', 'gift', 'rental_income', 'startup_exit')",
      "amount": "number or null",
      "currency": "string (e.g., 'GBP', 'USD', 'EUR')",
      "description": "string - brief explanation"
    }
  ],
  "total_amount": "number or null",
  "risk_level": "string ('low', 'medium', or 'high')",
  "documents_provided": ["array of strings"],
  "flags": ["array of strings - concerns or red flags"]
}"""
        
        prompt = f"""You are an expert at extracting structured information from account manager notes for compliance and risk assessment.

Extract information from these notes into the JSON schema below. Use null for missing information. Be precise and accurate.

NOTES:
{notes}

JSON SCHEMA:
{schema}

Extract the information and return ONLY valid JSON (no markdown, no explanations, just the JSON object):"""
        
        # Use fallback if LLM is not available
        if not self.use_llm or not self.client:
            return self._extract_fallback_data(notes)
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.3
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean up markdown code blocks if present
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            # Parse and validate JSON
            extracted_data = json.loads(content)
            
            # Ensure required fields exist
            if "customer_name" not in extracted_data:
                extracted_data["customer_name"] = "Unknown"
            if "wealth_sources" not in extracted_data or not extracted_data["wealth_sources"]:
                extracted_data["wealth_sources"] = [{"source_type": "unknown", "amount": None, "currency": "GBP", "description": "Not specified"}]
            if "risk_level" not in extracted_data:
                extracted_data["risk_level"] = "medium"
            if "documents_provided" not in extracted_data:
                extracted_data["documents_provided"] = []
            if "flags" not in extracted_data:
                extracted_data["flags"] = []
                
            return extracted_data
            
        except Exception as e:
            # Fallback to basic extraction
            return self._extract_fallback_data(notes)
    
    def _extract_fallback_data(self, notes: str) -> dict[str, Any]:
        """
        Extract basic structured data as fallback.
        
        Args:
            notes: Free-text notes
            
        Returns:
            Basic structured data dictionary
        """
        # Simple extraction logic
        risk_level = "low"
        if "high" in notes.lower() and "risk" in notes.lower():
            risk_level = "high"
        elif "medium" in notes.lower() and "risk" in notes.lower():
            risk_level = "medium"
        
        return {
            "customer_name": "Unknown Client",
            "occupation": None,
            "wealth_sources": [
                {
                    "source_type": "unknown",
                    "amount": None,
                    "currency": "GBP",
                    "description": "Source not clearly specified"
                }
            ],
            "total_amount": None,
            "risk_level": risk_level,
            "documents_provided": [],
            "flags": []
        }
    
    def generate(self) -> dict[str, pd.DataFrame]:
        """
        Generate synthetic KYC/SOW dataset.
        
        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames containing:
            - notes: Free-text account manager notes
            - structured_output: JSON string with extracted data
            - scenario_type: Type of wealth source scenario
            - difficulty: Difficulty level
        """
        scenarios = self._get_scenario_templates()
        difficulty_levels = ["easy", "medium", "hard"]
        
        # Generate samples
        all_samples = []
        
        for i in range(self.config.num_samples):
            # Select scenario and difficulty
            scenario = random.choice(scenarios)
            difficulty = random.choice(difficulty_levels)
            
            # Get amount range for this difficulty
            amount_range = scenario["typical_amounts"][difficulty]
            amount = random.randint(amount_range[0], amount_range[1])
            
            # Select currency
            currency = random.choice(scenario["currencies"])
            
            # Generate note
            notes = self._generate_note_with_llm(scenario, difficulty, amount, currency)
            
            # Extract structured data
            structured_data = self._extract_structured_data_with_llm(notes)
            
            # Create sample
            sample = {
                "notes": notes,
                "structured_output": json.dumps(structured_data),
                "scenario_type": scenario["scenario_type"],
                "difficulty": difficulty
            }
            
            all_samples.append(sample)
        
        # Create DataFrame
        df = pd.DataFrame(all_samples)
        
        # Shuffle data
        df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        # Split into train/val/test
        n = len(df)
        train_end = int(n * self.config.train_split)
        val_end = train_end + int(n * self.config.val_split)
        
        dataset = {
            "train": df[:train_end].reset_index(drop=True),
            "val": df[train_end:val_end].reset_index(drop=True),
            "test": df[val_end:].reset_index(drop=True)
        }
        
        return dataset
    
    def validate(self, dataset: dict[str, pd.DataFrame]) -> bool:
        """
        Validate generated dataset quality.
        
        Checks:
        - All required DataFrame columns present
        - All JSON strings are valid and parseable
        - Required JSON fields present
        - wealth_sources is non-empty array
        - Good variety in scenario types
        - Good variety in risk levels
        - No duplicate notes
        
        Args:
            dataset: Dictionary of DataFrames to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_columns = ["notes", "structured_output", "scenario_type", "difficulty"]
        
        try:
            # Check all splits exist
            if not all(split in dataset for split in ["train", "val", "test"]):
                print("ERROR: Missing required splits (train/val/test)")
                return False
            
            # Check each split
            all_scenario_types = set()
            all_risk_levels = set()
            all_notes = set()
            
            for split_name, df in dataset.items():
                # Check columns
                if not all(col in df.columns for col in required_columns):
                    print(f"ERROR: Missing required columns in {split_name} split")
                    return False
                
                # Check non-empty
                if len(df) == 0:
                    print(f"ERROR: {split_name} split is empty")
                    return False
                
                # Validate each row
                for idx, row in df.iterrows():
                    # Check JSON validity
                    try:
                        structured = json.loads(row["structured_output"])
                    except json.JSONDecodeError:
                        print(f"ERROR: Invalid JSON in {split_name} row {idx}")
                        return False
                    
                    # Check required JSON fields
                    required_fields = ["customer_name", "wealth_sources", "risk_level"]
                    if not all(field in structured for field in required_fields):
                        print(f"ERROR: Missing required JSON fields in {split_name} row {idx}")
                        return False
                    
                    # Check wealth_sources is non-empty array
                    if not isinstance(structured["wealth_sources"], list) or len(structured["wealth_sources"]) == 0:
                        print(f"ERROR: wealth_sources must be non-empty array in {split_name} row {idx}")
                        return False
                    
                    # Collect variety metrics
                    all_scenario_types.add(row["scenario_type"])
                    all_risk_levels.add(structured["risk_level"])
                    all_notes.add(row["notes"])
            
            # Check for good variety
            if len(all_scenario_types) < 5:
                print(f"WARNING: Low scenario variety - only {len(all_scenario_types)} unique types (should be at least 5)")
                return False
            
            if len(all_risk_levels) < 2:
                print(f"WARNING: Low risk level variety - only {len(all_risk_levels)} unique levels (should be at least 2)")
                return False
            
            # Check for duplicates
            total_samples = sum(len(df) for df in dataset.values())
            if len(all_notes) < total_samples:
                print(f"WARNING: Found duplicate notes ({total_samples - len(all_notes)} duplicates)")
                return False
            
            print(f"✓ Validation passed: {len(all_scenario_types)} scenario types, {len(all_risk_levels)} risk levels, no duplicates")
            return True
            
        except Exception as e:
            print(f"ERROR during validation: {str(e)}")
            return False
    
    def save(self, dataset: dict[str, pd.DataFrame], output_dir: str) -> None:
        """
        Save dataset to disk as CSV files.
        
        Args:
            dataset: Dictionary of DataFrames
            output_dir: Directory to save files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for split_name, df in dataset.items():
            filepath = os.path.join(output_dir, f"{split_name}.csv")
            df.to_csv(filepath, index=False)
            print(f"Saved {split_name} split to {filepath} ({len(df)} samples)")
    
    def load(self, input_dir: str) -> dict[str, pd.DataFrame]:
        """
        Load dataset from disk.
        
        Args:
            input_dir: Directory containing dataset files
            
        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames
        """
        dataset = {}
        for split_name in ["train", "val", "test"]:
            filepath = os.path.join(input_dir, f"{split_name}.csv")
            if os.path.exists(filepath):
                dataset[split_name] = pd.read_csv(filepath)
            else:
                print(f"WARNING: {split_name}.csv not found in {input_dir}")
        
        return dataset
    
    def get_statistics(self, dataset: dict[str, pd.DataFrame]) -> dict[str, Any]:
        """
        Compute dataset statistics.
        
        Args:
            dataset: Dictionary of DataFrames
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_samples": sum(len(df) for df in dataset.values()),
            "splits": {
                split_name: len(df)
                for split_name, df in dataset.items()
            },
            "scenario_distribution": {},
            "difficulty_distribution": {},
            "risk_level_distribution": {}
        }
        
        # Combine all data for overall statistics
        all_data = pd.concat(dataset.values(), ignore_index=True)
        
        # Scenario type distribution
        stats["scenario_distribution"] = all_data["scenario_type"].value_counts().to_dict()
        
        # Difficulty distribution
        stats["difficulty_distribution"] = all_data["difficulty"].value_counts().to_dict()
        
        # Risk level distribution (from JSON)
        risk_levels = []
        for _, row in all_data.iterrows():
            try:
                structured = json.loads(row["structured_output"])
                risk_levels.append(structured.get("risk_level", "unknown"))
            except:
                pass
        
        stats["risk_level_distribution"] = dict(Counter(risk_levels))
        
        return stats
