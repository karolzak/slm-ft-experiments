"""
KYC/SOW data schema definitions using Pydantic.

Defines the structured output schema for KYC (Know Your Customer) and 
SOW (Source of Wealth) extraction tasks.
"""

from pydantic import BaseModel, Field


class WealthSource(BaseModel):
    """Individual wealth source information."""
    
    source_type: str = Field(
        ...,
        description="Type of wealth source (e.g., 'employment', 'business_sale', 'inheritance', 'investment', 'property_sale', 'pension', 'crypto', 'gift', 'rental_income', 'startup_exit')"
    )
    amount: int | None = Field(
        None,
        description="Amount in the specified currency"
    )
    currency: str = Field(
        ...,
        description="Currency code (e.g., 'GBP', 'USD', 'EUR')"
    )
    description: str = Field(
        ...,
        description="Brief explanation of the wealth source"
    )


class KYCSOWOutput(BaseModel):
    """Structured output for KYC/SOW extraction."""
    
    customer_name: str = Field(
        ...,
        description="Full name of the customer"
    )
    occupation: str | None = Field(
        None,
        description="Customer's occupation or business description"
    )
    wealth_sources: list[WealthSource] = Field(
        ...,
        description="List of wealth sources (must have at least one)",
        min_length=1
    )
    total_amount: int | None = Field(
        None,
        description="Total amount across all wealth sources"
    )
    risk_level: str = Field(
        ...,
        description="Risk assessment level: 'low', 'medium', or 'high'"
    )
    documents_provided: list[str] = Field(
        default_factory=list,
        description="List of documents provided or requested"
    )
    flags: list[str] = Field(
        default_factory=list,
        description="List of concerns or red flags identified"
    )
    
    def model_dump_json_schema(self) -> str:
        """
        Generate JSON schema string suitable for LLM prompts.
        
        Returns:
            JSON schema as a formatted string
        """
        import json
        schema = self.model_json_schema()
        return json.dumps(schema, indent=2)


def get_kyc_sow_schema() -> str:
    """
    Get the KYC/SOW JSON schema as a string for LLM prompts.
    
    Returns:
        JSON schema string
    """
    return KYCSOWOutput.model_json_schema()
