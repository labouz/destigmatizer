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
        
    # Create pipeline with all filters and their conditions
    pipeline = create_pipeline(client=client, model=model)
    
    # Add drug filter (always executes)
    pipeline.add_filter(classify_drug_filter)
    
    # Add stigma filter (only if drug-related)
    pipeline.add_filter(classify_stigma_filter, condition="is_drug_related")
    
    # Add style analyzer (only if stigmatizing)
    pipeline.add_filter(analyze_style_filter, condition="is_stigmatizing")
    
    # Add rewriter (only if stigmatizing)
    pipeline.add_filter(rewrite_text_filter, condition="is_stigmatizing")
    
    if verbose:
        print("Processing text through pipeline...")
    
    # Process through the complete pipeline
    result = pipeline.process(text)
    
    # Log progress if verbose
    if verbose:
        if not result["metadata"].get("is_drug_related", False):
            print("Text is not drug-related. Skipping further analysis.")
        elif not result["metadata"].get("is_stigmatizing", False):
            print("No stigmatizing content detected. Skipping further analysis.")
        elif result["metadata"].get("was_rewritten", False):
            print("Text was successfully rewritten.")
    
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
    """Rewrite stigmatizing text using provided explanation and style instructions."""
    # Create a pipeline with pre-populated metadata
    pipeline = create_pipeline(client=client, model=model)
    
    # Custom filter that uses the externally provided data instead of pipeline metadata
    def custom_rewrite_filter(text, client, model, **kwargs):
        from .rewriters import DestigmatizingRewriter
        
        rewriter = DestigmatizingRewriter(client)
        rewritten_text = rewriter.rewrite(
            text=text,
            explanation=explanation,
            style_instruct=style_instruct,
            model=model
        )
        
        return {
            "text": rewritten_text,
            "metadata": {
                "was_rewritten": True
            }
        }
    
    # Add our custom filter
    pipeline.add_filter(custom_rewrite_filter)
    
    # Process and return
    result = pipeline.process(text)
    return result["processed_text"]

