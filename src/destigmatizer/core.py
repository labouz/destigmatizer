"""Core functionality for the destigmatizer package using pipeline pattern."""

from typing import Dict, Any, Optional
from .pipeline import create_pipeline
from .filters import (
    classify_drug_filter, 
    classify_stigma_filter, 
    analyze_style_filter,
    rewrite_text_filter
)
from .clients import get_client

def initialize(api_key: Optional[str] = None, client: Optional[Any] = None, 
              client_type: Optional[str] = None) -> Any:
    """
    Initialize and return a client for the destigmatizer library.
    
    Args:
        api_key: API key for the language model service
        client: Pre-configured client instance
        client_type: Type of client ("openai", "together", or "claude")
        
    Returns:
        Any: Client instance
    """
    if client:
        return client
    return get_client(client_type, api_key)

def analyze_and_rewrite_text(text: str, client: Any, model: Optional[str] = None, 
                           verbose: bool = False) -> str:
    """
    Analyze and rewrite text using a preconfigured pipeline.
    
    Args:
        text: Text to process
        client: LLM client
        model: Model to use
        verbose: Whether to print pipeline progress
        
    Returns:
        str: Processed text (rewritten if necessary)
    """
    if verbose:
        print("Step 1: Creating pipeline...")
        
    # Create pipeline
    pipeline = create_pipeline(client=client, model=model)
    
    # Add filters in sequence
    pipeline.add_filter(classify_drug_filter)
    
    if verbose:
        print("Step 2: Classifying drug-related content...")
        
    # Process text with just the drug filter
    intermediate_result = pipeline.process(text)
    
    # If not drug-related, return original text
    if not intermediate_result["metadata"].get("is_drug_related", False):
        if verbose:
            print("Text is not drug-related. Skipping further analysis.")
        return text
    
    # Add remaining filters
    if verbose:
        print("Step 3: Checking for stigmatizing language...")
        
    pipeline.add_filter(classify_stigma_filter)
    
    # Process with the stigma filter
    intermediate_result = pipeline.process(text)
    
    # If not stigmatizing, return original text
    if not intermediate_result["metadata"].get("is_stigmatizing", False):
        if verbose:
            print("No stigmatizing content detected. Skipping further analysis.")
        return text
    
    if verbose:
        print("Step 4: Analyzing text style...")
        
    pipeline.add_filter(analyze_style_filter)
    
    if verbose:
        print("Step 5: Rewriting stigmatizing content...")
        
    pipeline.add_filter(rewrite_text_filter)
    
    # Process through the full pipeline
    result = pipeline.process(text)
    
    return result["processed_text"]

# Keep these functions for backward compatibility but implement using pipeline
def classify_if_drug(text: str, client: Any, model: Optional[str] = None) -> str:
    """Backward compatibility for drug classification."""
    pipeline = create_pipeline(client=client, model=model)
    pipeline.add_filter(classify_drug_filter)
    result = pipeline.process(text)
    is_drug = result["metadata"].get("is_drug_related", False)
    return "d" if is_drug else "nd"

def classify_if_stigma(text: str, client: Any, model: Optional[str] = None) -> str:
    """Backward compatibility for stigma classification."""
    pipeline = create_pipeline(client=client, model=model)
    pipeline.add_filter(classify_stigma_filter)
    result = pipeline.process(text)
    is_stigma = result["metadata"].get("is_stigmatizing", False)
    explanation = result["metadata"].get("stigma_explanation", "")
    return f"s, {explanation}" if is_stigma else "ns"

def analyze_text_llm(text: str, client: Any, model: Optional[str] = None) -> Dict[str, Any]:
    """Backward compatibility for text style analysis."""
    pipeline = create_pipeline(client=client, model=model)
    pipeline.add_filter(analyze_style_filter)
    result = pipeline.process(text)
    return result["metadata"].get("style_analysis", {})

def rewrite_to_destigma(text, explanation, style_instruct, model, client):
    # Create pipeline
    pipeline = create_pipeline(client=client, model=model)
    
    # Manipulate the pipeline data directly
    pipeline.add_filter(
        rewrite_text_filter,
        rewrite_data = {
            "explanation": explanation,
            "style_instruct": style_instruct,
            "client": client,
            "model": model
        }
        
    )
    
    result = pipeline.process(text)
    return result["processed_text"]

