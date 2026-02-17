#!/usr/bin/env python3
"""
Example usage of KYCSOWDataGenerator.

This script demonstrates how to use the KYC/SOW data generator to create
synthetic datasets for financial compliance ML tasks.

Before running:
1. Set up Azure OpenAI credentials in .env file:
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_OPENAI_API_KEY=your-api-key
   AZURE_OPENAI_DEPLOYMENT=gpt-4  # or your deployment name

2. Install dependencies:
   pip install pandas openai python-dotenv
"""

import os
import sys

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
from src.data.generators import KYCSOWDataGenerator, DatasetConfig


def main():
    """Generate and validate a KYC/SOW dataset."""
    
    # Load environment variables
    load_dotenv()
    
    # Check if Azure OpenAI is configured
    if not os.getenv("AZURE_OPENAI_ENDPOINT") or not os.getenv("AZURE_OPENAI_API_KEY"):
        print("❌ ERROR: Azure OpenAI credentials not configured")
        print()
        print("The KYC/SOW generator requires Azure OpenAI credentials.")
        print("Please set the following environment variables in your .env file:")
        print("  - AZURE_OPENAI_ENDPOINT")
        print("  - AZURE_OPENAI_API_KEY")
        print("  - AZURE_OPENAI_DEPLOYMENT (optional, defaults to 'gpt-4')")
        print()
        return
    
    # Create configuration
    config = DatasetConfig(
        num_samples=20,  # Generate 20 samples (start small for testing)
        task_type="kyc_sow",
        difficulty_level="mixed",  # Will generate all difficulty levels
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        seed=42
    )
    
    print("=" * 80)
    print("KYC/SOW Data Generator - Example Usage")
    print("=" * 80)
    print()
    print(f"Configuration:")
    print(f"  - Total samples: {config.num_samples}")
    print(f"  - Train/Val/Test split: {config.train_split}/{config.val_split}/{config.test_split}")
    print(f"  - Random seed: {config.seed}")
    print()
    
    # Initialize generator
    print("Initializing generator...")
    generator = KYCSOWDataGenerator(config)
    print("✓ Generator initialized")
    print()
    
    # Generate dataset
    print("Generating synthetic KYC/SOW dataset...")
    print("(This may take a few minutes with real LLM calls)")
    dataset = generator.generate()
    print("✓ Dataset generated")
    print()
    
    # Display statistics
    print("Dataset Statistics:")
    print("-" * 80)
    stats = generator.get_statistics(dataset)
    print(f"Total samples: {stats['total_samples']}")
    print(f"  - Train: {stats['splits']['train']}")
    print(f"  - Val: {stats['splits']['val']}")
    print(f"  - Test: {stats['splits']['test']}")
    print()
    
    print("Scenario Type Distribution:")
    for scenario, count in sorted(stats['scenario_distribution'].items()):
        print(f"  - {scenario}: {count}")
    print()
    
    print("Difficulty Distribution:")
    for difficulty, count in sorted(stats['difficulty_distribution'].items()):
        print(f"  - {difficulty}: {count}")
    print()
    
    print("Risk Level Distribution:")
    for risk, count in sorted(stats['risk_level_distribution'].items()):
        print(f"  - {risk}: {count}")
    print()
    
    # Validate dataset
    print("Validating dataset...")
    is_valid = generator.validate(dataset)
    if is_valid:
        print("✓ Dataset validation passed!")
    else:
        print("✗ Dataset validation failed")
        return
    print()
    
    # Display sample from training set
    print("Sample from Training Set:")
    print("-" * 80)
    sample = dataset['train'].iloc[0]
    print(f"Scenario Type: {sample['scenario_type']}")
    print(f"Difficulty: {sample['difficulty']}")
    print()
    print("Account Manager Notes:")
    print(sample['notes'][:300] + "..." if len(sample['notes']) > 300 else sample['notes'])
    print()
    print("Structured Output (JSON):")
    import json
    structured = json.loads(sample['structured_output'])
    print(json.dumps(structured, indent=2))
    print()
    
    # Save dataset
    output_dir = "./kyc_sow_dataset"
    print(f"Saving dataset to {output_dir}...")
    generator.save(dataset, output_dir)
    print("✓ Dataset saved successfully!")
    print()
    
    print("=" * 80)
    print("Example complete!")
    print()
    print("Next steps:")
    print("  1. Review the generated data in ./kyc_sow_dataset/")
    print("  2. Use the dataset for model training or evaluation")
    print("  3. Adjust num_samples in config for larger datasets")
    print("  4. Customize scenario templates for specific use cases")
    print("=" * 80)


if __name__ == "__main__":
    main()
