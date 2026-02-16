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
                    compute_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deploy a finetuned model.
        
        Args:
            model_id: Model identifier
            deployment_name: Name for deployment
            compute_config: Compute configuration
            
        Returns:
            Deployment information
        """
        pass
    
    def delete_deployment(self, deployment_name: str) -> None:
        """
        Delete a model deployment.
        
        Args:
            deployment_name: Name of deployment to delete
        """
        pass
