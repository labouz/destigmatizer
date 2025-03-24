"""
Pipeline implementation for text processing using the pipe and filter pattern.
"""
from typing import List, Dict, Any, Optional, Callable, Union
from .clients import get_client

class TextPipeline:
    """
    A pipeline for processing and transforming text using a series of filters.
    """
    
    def __init__(self, client: Any, model: Optional[str] = None):
        """
        Initialize the pipeline with a client and model.
        
        Args:
            client: LLM client instance
            model: Model name to use for LLM operations
        """
        self.client = client
        self.model = model
        self.filters = []
        
    def add_filter(self, filter_func: Callable, **kwargs) -> 'TextPipeline':
        """
        Add a filter function to the pipeline.
        
        Args:
            filter_func: Function that takes text and returns processed text
            **kwargs: Additional parameters to pass to the filter function
            
        Returns:
            self: For method chaining
        """
        self.filters.append((filter_func, kwargs))
        return self
    
    def process(self, text: str) -> Dict[str, Any]:
        """
        Process text through the pipeline.
        
        Args:
            text: Input text
            
        Returns:
            dict: Result containing processed text and metadata
        """
        result = {
            "original_text": text,
            "processed_text": text,
            "metadata": {}
        }
        
        for filter_func, kwargs in self.filters:
            # Pass the current result to each filter
            filter_result = filter_func(
                text=result["processed_text"],
                client=self.client,
                model=self.model,
                pipeline_data=result,
                **kwargs
            )
            
            # Update result with filter's output
            if isinstance(filter_result, dict):
                if "text" in filter_result:
                    result["processed_text"] = filter_result["text"]
                
                # Merge metadata
                result["metadata"].update(filter_result.get("metadata", {}))
            elif isinstance(filter_result, str):
                # If filter just returns text
                result["processed_text"] = filter_result
        
        return result


def create_pipeline(api_key: Optional[str] = None, client_type: str = "openai", 
                   model: Optional[str] = None, client: Optional[Any] = None) -> TextPipeline:
    """
    Create a text processing pipeline with the specified client.
    
    Args:
        api_key: API key for the LLM provider
        client_type: Type of client to use
        model: Model name to use
        client: Pre-configured client instance
        
    Returns:
        TextPipeline: Configured pipeline instance
    """
    if client is None:
        client = get_client(client_type, api_key)
    
    return TextPipeline(client, model)