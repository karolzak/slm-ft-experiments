"""
KYC/SOW Data Generator for financial services ML applications.

Generates synthetic account manager notes and structured JSON extractions
for Know Your Customer (KYC) and Source of Wealth (SOW) extraction tasks.
"""

import os
import json
import random
import re
import yaml
from pathlib import Path
from typing import Any
from collections import Counter
import pandas as pd
from openai import AzureOpenAI, BadRequestError
from pydantic import ValidationError

try:
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
except ImportError:
    DefaultAzureCredential = None
    get_bearer_token_provider = None

from .base import BaseDatasetGenerator, DatasetConfig
from .kyc_sow_schema import KYCSOWOutput, get_kyc_sow_schema


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
            
        Raises:
            ValueError: If Azure OpenAI credentials are not configured
        """
        super().__init__(config)
        
        # Check if Azure OpenAI is configured
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
        api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
        auth_mode = os.getenv("AZURE_OPENAI_AUTH_MODE", "auto").strip().lower()

        if not endpoint:
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT is required for KYC/SOW data generation."
            )

        if auth_mode not in {"auto", "api_key", "azure_ad"}:
            raise ValueError(
                "AZURE_OPENAI_AUTH_MODE must be one of: auto, api_key, azure_ad"
            )

        # Initialize Azure OpenAI client with dual authentication support:
        # 1) API key auth when mode=api_key, or mode=auto with AZURE_OPENAI_API_KEY provided
        # 2) Microsoft Entra ID auth when mode=azure_ad, or mode=auto without API key
        use_api_key = auth_mode == "api_key" or (auth_mode == "auto" and bool(api_key))

        if use_api_key:
            if not api_key:
                raise ValueError(
                    "AZURE_OPENAI_AUTH_MODE is 'api_key' but AZURE_OPENAI_API_KEY is not set."
                )

            self.client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version
            )
            self.auth_mode = "api_key"
        else:
            if DefaultAzureCredential is None or get_bearer_token_provider is None:
                raise ValueError(
                    "AZURE_OPENAI_API_KEY is not set and azure-identity is not installed. "
                    "Install azure-identity or provide AZURE_OPENAI_API_KEY."
                )

            credential = DefaultAzureCredential()
            token_provider = get_bearer_token_provider(
                credential,
                "https://cognitiveservices.azure.com/.default"
            )

            self.client = AzureOpenAI(
                azure_endpoint=endpoint,
                azure_ad_token_provider=token_provider,
                api_version=api_version
            )
            self.auth_mode = "azure_ad"

        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
        
        # Load scenario templates from config file
        self._load_scenario_templates()
        
        # Set random seed for reproducibility
        random.seed(self.seed)

    def _get_raw_response_text(self, response: Any) -> str:
        """
        Extract raw text content from a Responses API response.

        Args:
            response: Response object returned by Azure OpenAI Responses API

        Returns:
            Raw text content as a string

        Raises:
            ValueError: If no text content can be extracted
        """
        # Fast path supported by latest SDKs
        output_text = getattr(response, "output_text", None)
        if output_text and output_text.strip():
            return output_text.strip()

        # Fallback: manually walk response.output blocks
        output = getattr(response, "output", []) or []
        text_chunks: list[str] = []

        for item in output:
            content_blocks = getattr(item, "content", []) or []
            for block in content_blocks:
                block_text = getattr(block, "text", None)
                if isinstance(block_text, str) and block_text.strip():
                    text_chunks.append(block_text)

        content = "\n".join(text_chunks).strip()
        if content:
            return content

        raise ValueError("No text content returned by Responses API")

    def _create_response(self, prompt: str, max_output_tokens: int, temperature: float) -> Any:
        """
        Create a response via Responses API with model compatibility fallback.

        Retries once without temperature if the model does not support it.
        """
        request_args = {
            "model": self.deployment_name,
            "input": prompt,
            "max_output_tokens": max_output_tokens,
            "temperature": temperature
        }

        try:
            return self.client.responses.create(**request_args)
        except BadRequestError as exc:
            error_message = str(exc).lower()
            if "temperature" in error_message and "not supported" in error_message:
                request_args.pop("temperature", None)
                return self.client.responses.create(**request_args)
            raise

    def _parse_json_from_response_text(self, content: str) -> dict[str, Any]:
        """
        Parse JSON content from a model response, handling wrapped text/code blocks.
        """
        cleaned = content.strip()

        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", cleaned)
            if match:
                return json.loads(match.group(0))
            raise
    
    def _load_scenario_templates(self) -> None:
        """
        Load scenario templates from YAML configuration file.
        
        Raises:
            FileNotFoundError: If scenario config file not found
            ValueError: If scenario config is invalid
        """
        # Find the config file relative to this module
        module_dir = Path(__file__).parent
        config_path = module_dir.parent.parent.parent / "config" / "kyc_sow_scenarios.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"KYC/SOW scenario configuration file not found at {config_path}"
            )
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        if 'scenarios' not in config_data:
            raise ValueError("Invalid scenario config: 'scenarios' key not found")
        
        # Convert YAML format to internal format
        self.scenarios = []
        for scenario in config_data['scenarios']:
            # Convert amount ranges from lists to tuples
            typical_amounts = {}
            for difficulty, amounts in scenario['typical_amounts'].items():
                typical_amounts[difficulty] = tuple(amounts)
            
            self.scenarios.append({
                "scenario_type": scenario['scenario_type'],
                "description": scenario['description'],
                "typical_amounts": typical_amounts,
                "currencies": scenario['currencies'],
                "risk_profile": scenario['risk_profile']
            })
    
    def _generate_note_text_with_llm(
        self,
        scenario: dict[str, Any],
        difficulty: str,
        amount: int,
        currency: str
    ) -> str:
        """
        Generate realistic account manager note text using LLM.
        
        Args:
            scenario: Scenario template dictionary
            difficulty: Difficulty level ('easy', 'medium', 'hard')
            amount: Amount of funds
            currency: Currency code
            
        Returns:
            Generated free-text account manager note text
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
        
        response = self._create_response(
            prompt=prompt,
            max_output_tokens=1000,
            temperature=0.8
        )

        return self._get_raw_response_text(response)
    
    def _extract_structured_data_with_llm(self, note_text: str) -> dict[str, Any]:
        """
        Extract structured JSON from account manager note text using LLM.
        
        Args:
            note_text: Free-text account manager note text
            
        Returns:
            Dictionary with structured extracted data
            
        Raises:
            json.JSONDecodeError: If LLM returns invalid JSON
            ValidationError: If extracted data doesn't match schema
        """
        # Get schema from Pydantic model
        schema = json.dumps(get_kyc_sow_schema(), indent=2)
        
        prompt = f"""You are an expert at extracting structured information from account manager notes for compliance and risk assessment.

    This is for a synthetic ML dataset used in financial compliance training and evaluation.
    Do not provide advice on evasion or wrongdoing. Only extract facts from the notes.

Extract information from these notes into the JSON schema below. Use null for missing information. Be precise and accurate.

NOTES:
{note_text}

JSON SCHEMA:
{schema}

Extract the information and return ONLY valid JSON (no markdown, no explanations, just the JSON object):"""
        
        try:
            response = self._create_response(
                prompt=prompt,
                max_output_tokens=800,
                temperature=0.3
            )

            content = self._get_raw_response_text(response)

            # Parse JSON with resilient text cleanup
            extracted_data = self._parse_json_from_response_text(content)
            
            # Validate against Pydantic schema
            validated_output = KYCSOWOutput(**extracted_data)
            
            # Return as dictionary
            return validated_output.model_dump()
            
        except json.JSONDecodeError as e:
            # Retry once with stricter formatting instructions
            retry_prompt = f"""Extract structured data from the notes below and return ONLY a valid JSON object matching the schema.
If any field is unknown, use null (or [] for arrays). No explanations.

NOTES:
{notes}

JSON SCHEMA:
{schema}
"""
            retry_response = self._create_response(
                prompt=retry_prompt,
                max_output_tokens=800,
                temperature=0.1
            )
            retry_content = self._get_raw_response_text(retry_response)

            try:
                extracted_data = self._parse_json_from_response_text(retry_content)
                validated_output = KYCSOWOutput(**extracted_data)
                return validated_output.model_dump()
            except Exception:
                raise json.JSONDecodeError(
                    f"Failed to parse JSON from LLM response. Content: {retry_content[:200]}...",
                    e.doc,
                    e.pos
                )
        except ValidationError as e:
            # If validation fails, try to fix common issues and return best-effort result
            print(f"WARNING: Schema validation failed: {e}")
            
            # Ensure required fields exist with defaults
            if "customer_name" not in extracted_data:
                extracted_data["customer_name"] = "Unknown"
            if "wealth_sources" not in extracted_data or not extracted_data["wealth_sources"]:
                extracted_data["wealth_sources"] = [{
                    "source_type": "unknown",
                    "amount": None,
                    "currency": "GBP",
                    "description": "Not specified"
                }]
            if "risk_level" not in extracted_data:
                extracted_data["risk_level"] = "medium"
            if "documents_provided" not in extracted_data:
                extracted_data["documents_provided"] = []
            if "flags" not in extracted_data:
                extracted_data["flags"] = []
            
            return extracted_data
    
    def generate(self) -> dict[str, pd.DataFrame]:
        """
        Generate synthetic KYC/SOW dataset.
        
        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames containing:
            - note_text: Free-text account manager note text
            - structured_output: JSON string with extracted data
            - scenario_type: Type of wealth source scenario
            - difficulty: Difficulty level
        """
        difficulty_levels = ["easy", "medium", "hard"]
        
        # Generate samples
        all_samples = []
        
        for i in range(self.config.num_samples):
            # Select scenario and difficulty
            scenario = random.choice(self.scenarios)
            difficulty = random.choice(difficulty_levels)
            
            # Get amount range for this difficulty
            amount_range = scenario["typical_amounts"][difficulty]
            amount = random.randint(amount_range[0], amount_range[1])
            
            # Select currency
            currency = random.choice(scenario["currencies"])
            
            # Generate note text
            note_text = self._generate_note_text_with_llm(scenario, difficulty, amount, currency)
            
            # Extract structured data
            structured_data = self._extract_structured_data_with_llm(note_text)
            
            # Create sample
            sample = {
                "note_text": note_text,
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
        required_columns = ["note_text", "structured_output", "scenario_type", "difficulty"]
        
        try:
            # Check all splits exist
            if not all(split in dataset for split in ["train", "val", "test"]):
                print("ERROR: Missing required splits (train/val/test)")
                return False
            
            # Check each split
            all_scenario_types = set()
            all_risk_levels = set()
            all_note_texts = set()
            
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
                    all_note_texts.add(row["note_text"])
            
            # Check for good variety
            if len(all_scenario_types) < 5:
                print(f"WARNING: Low scenario variety - only {len(all_scenario_types)} unique types (should be at least 5)")
                return False
            
            if len(all_risk_levels) < 2:
                print(f"WARNING: Low risk level variety - only {len(all_risk_levels)} unique levels (should be at least 2)")
                return False
            
            # Check for duplicates
            total_samples = sum(len(df) for df in dataset.values())
            if len(all_note_texts) < total_samples:
                print(f"WARNING: Found duplicate note texts ({total_samples - len(all_note_texts)} duplicates)")
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
