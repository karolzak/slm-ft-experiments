# KYC/SOW Data Generator

## Overview

The KYC/SOW Data Generator creates synthetic datasets for **Know Your Customer (KYC)** and **Source of Wealth (SOW)** extraction tasks in financial services. It uses LLMs to generate realistic account manager notes and extract structured JSON data for compliance and risk assessment applications.

## Features

- ✅ **Two-Stage LLM Generation**: Generates realistic notes, then extracts structured data
- ✅ **12+ Scenario Types**: Employment, business sale, inheritance, crypto, property, investments, etc.
- ✅ **3 Difficulty Levels**: Easy, medium, and hard samples with varying complexity
- ✅ **Risk Assessment**: Automatic categorization into low, medium, and high risk levels
- ✅ **Fallback Mode**: Works without Azure credentials using template-based generation
- ✅ **Comprehensive Validation**: 12 validation rules ensure dataset quality
- ✅ **Train/Val/Test Splits**: Automatic dataset splitting with configurable ratios

## Installation

```bash
# Install dependencies
pip install pandas openai python-dotenv

# Or install the package
pip install -e .
```

## Quick Start

### 1. Configure Azure OpenAI (Optional)

Create a `.env` file with your Azure credentials:

```bash
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=gpt-4  # or your deployment name
```

**Note**: The generator works without Azure credentials in fallback mode, but LLM-powered generation produces more realistic and varied data.

### 2. Basic Usage

```python
from src.data.generators import KYCSOWDataGenerator, DatasetConfig

# Configure dataset generation
config = DatasetConfig(
    num_samples=100,
    task_type="kyc_sow",
    difficulty_level="mixed",
    train_split=0.7,
    val_split=0.15,
    test_split=0.15,
    seed=42
)

# Initialize generator
generator = KYCSOWDataGenerator(config)

# Generate dataset
dataset = generator.generate()

# Validate dataset
if generator.validate(dataset):
    print("✓ Dataset is valid!")
    
# Save to disk
generator.save(dataset, "./kyc_sow_data")

# Get statistics
stats = generator.get_statistics(dataset)
print(f"Generated {stats['total_samples']} samples")
```

### 3. Run Example Script

```bash
python examples/kyc_sow_example.py
```

## Dataset Structure

Each dataset contains train, validation, and test splits as pandas DataFrames with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `notes` | string | Free-text account manager notes |
| `structured_output` | JSON string | Extracted structured data |
| `scenario_type` | string | Type of wealth source scenario |
| `difficulty` | string | Difficulty level (easy/medium/hard) |

### Structured Output Schema

```json
{
  "customer_name": "string",
  "occupation": "string or null",
  "wealth_sources": [
    {
      "source_type": "string",
      "amount": "number or null",
      "currency": "string",
      "description": "string"
    }
  ],
  "total_amount": "number or null",
  "risk_level": "string (low/medium/high)",
  "documents_provided": ["array of strings"],
  "flags": ["array of concerns/red flags"]
}
```

## Scenario Types

The generator supports 12 different wealth source scenarios:

1. **Employment** - Salary-based income
2. **Business Sale** - Company sale proceeds
3. **Inheritance** - Estate transfers
4. **Property Sale** - Real estate transactions
5. **Investment** - Stock/bond returns
6. **Pension** - Retirement funds
7. **Crypto** - Cryptocurrency gains
8. **Startup Exit** - Venture-backed exits
9. **Gift** - Family gifts
10. **International Transfer** - Cross-border funds
11. **Rental Income** - Property portfolio
12. **Complex Portfolio** - Multiple sources

## Difficulty Levels

### Easy
- Single wealth source
- Complete documentation
- Clear explanations
- Low risk profile
- Professional writing style

### Medium
- 2-3 wealth sources
- Some documentation gaps
- Moderate complexity
- Medium risk factors
- Mixed writing style

### Hard
- Multiple wealth sources (3+)
- Incomplete information
- Inconsistencies in details
- High risk indicators
- Multiple jurisdictions
- Vague explanations

## Examples

### Example 1: Low Risk - Employment Income

**Generated Notes**:
```
Meeting with Sarah Thompson, 34, Senior Software Engineer at TechCorp Ltd.

Sarah is opening an investment account with an initial deposit of £45,000. 
Source of funds is accumulated savings from her employment over the past 5 years. 
Current salary is £95,000 per annum plus annual bonus typically around £15,000.

She provided recent payslips (last 3 months) and her latest P60 showing gross 
pay of £110,000 for the previous tax year. Employment confirmed via HR letter 
from TechCorp.

No concerns raised. All documentation in order. Approved for standard onboarding.
```

**Extracted JSON**:
```json
{
  "customer_name": "Sarah Thompson",
  "occupation": "Senior Software Engineer",
  "wealth_sources": [{
    "source_type": "employment",
    "amount": 45000,
    "currency": "GBP",
    "description": "Accumulated savings from employment"
  }],
  "total_amount": 45000,
  "risk_level": "low",
  "documents_provided": ["payslips", "P60", "employment_letter"],
  "flags": []
}
```

### Example 2: High Risk - Multiple Sources

**Generated Notes**:
```
Call with Mr. Alessandro Vittorio regarding substantial deposit request.

Client states funds (~£2.3M) coming from "various business interests" - mentioned 
property development in Malta, consulting fees from UAE-based company, and some 
inheritance from deceased uncle in Switzerland.

When asked for breakdown, he became somewhat vague. Said approximately £800k from 
property sale but couldn't provide exact date. Mentioned £1.2M from consulting but 
unclear on nature of services.

Documents: Has provided sale agreement for Malta property (£650k, not £800k as 
stated) and bank statement showing wire transfer from Zurich (€320k). Still 
awaiting full documentation on consulting arrangement.

Some concerns:
- Discrepancies in amounts mentioned vs. documented
- Vague on consulting services details
- Multiple jurisdictions (Malta, UAE, Switzerland)
- Reluctant to provide full documentation

HOLD pending enhanced due diligence.
```

**Extracted JSON**:
```json
{
  "customer_name": "Alessandro Vittorio",
  "occupation": "Consultant / Property Developer",
  "wealth_sources": [
    {
      "source_type": "property_sale",
      "amount": 650000,
      "currency": "GBP",
      "description": "Property sale in Malta"
    },
    {
      "source_type": "business_income",
      "amount": 1200000,
      "currency": "GBP",
      "description": "Consulting fees from UAE"
    },
    {
      "source_type": "inheritance",
      "amount": 320000,
      "currency": "EUR",
      "description": "Inheritance from Switzerland"
    }
  ],
  "total_amount": 2300000,
  "risk_level": "high",
  "documents_provided": ["property_sale_agreement", "bank_statement_zurich"],
  "flags": [
    "amount_discrepancies",
    "vague_consulting_details",
    "multiple_offshore_jurisdictions",
    "incomplete_documentation"
  ]
}
```

## Validation

The generator includes comprehensive validation that checks:

- ✅ All required DataFrame columns present
- ✅ All JSON strings valid and parseable
- ✅ Required JSON fields present (customer_name, wealth_sources, risk_level)
- ✅ wealth_sources is non-empty array
- ✅ Good scenario variety (5+ types)
- ✅ Good risk level variety (2+ levels)
- ✅ No duplicate notes
- ✅ Proper train/val/test splits

## Advanced Usage

### Custom Configuration

```python
config = DatasetConfig(
    num_samples=500,
    task_type="kyc_sow",
    difficulty_level="mixed",
    train_split=0.8,
    val_split=0.1,
    test_split=0.1,
    seed=123,
    additional_params={
        "include_pep_checks": True,
        "enhanced_due_diligence": True
    }
)
```

### Load Existing Dataset

```python
generator = KYCSOWDataGenerator(config)
dataset = generator.load("./kyc_sow_data")
stats = generator.get_statistics(dataset)
```

### Batch Processing

For large datasets, consider:
- Using cheaper models (GPT-3.5, GPT-4-mini) for simple scenarios
- Implementing rate limiting for Azure API calls
- Generating in smaller batches
- Monitoring token usage and costs

## Testing

Run the test suite:

```bash
# Run all KYC/SOW generator tests
python -m unittest tests.test_kyc_sow_generator -v

# Run all project tests
python -m unittest discover -q
```

## Performance Considerations

- **With LLM**: ~2-5 seconds per sample (depends on Azure response time)
- **Fallback Mode**: ~instant per sample
- **Recommended**: Start with 20-50 samples for testing, scale to 500+ for production
- **Cost**: Azure OpenAI has per-token pricing - monitor usage for large datasets

## Troubleshooting

### Issue: "No module named 'openai'"
**Solution**: Install dependencies with `pip install openai pandas python-dotenv`

### Issue: Generator seems slow or hangs
**Solution**: Check Azure API status and rate limits. Consider using fallback mode for testing.

### Issue: Low data variety in fallback mode
**Solution**: Configure Azure OpenAI credentials for LLM-powered generation.

### Issue: Validation fails
**Solution**: Check console output for specific validation errors. Common issues:
- Missing required columns
- Invalid JSON in structured_output
- Empty wealth_sources arrays

## License

See project LICENSE file.

## Contributing

Contributions welcome! Please:
1. Add tests for new features
2. Follow existing code style
3. Update documentation
4. Ensure all tests pass

## Support

For issues or questions, please open a GitHub issue in the repository.
