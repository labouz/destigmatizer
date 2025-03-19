"""
Destigmatizer: A Python package for destigmatizing language related to drug use.

This package provides tools to identify, analyze, and rewrite text
containing stigmatizing language.
"""

# Core functions
from .core import (
    initialize,
    analyze_and_rewrite_text,
    classify_if_drug,
    classify_if_stigma,
    analyze_text_llm,
    rewrite_to_destigma
)

# Pipeline components
from .pipeline import create_pipeline, TextPipeline
from .filters import (
    classify_drug_filter,
    classify_stigma_filter,
    analyze_style_filter,
    rewrite_text_filter
)

# Original components for backward compatibility
from .clients import get_client, LLMClient, OpenAIClient, TogetherClient, ClaudeClient
from .utils import get_model_mapping, get_default_model

__all__ = [
    # Core functions
    'initialize',
    'analyze_and_rewrite_text',
    'classify_if_drug',
    'classify_if_stigma',
    'analyze_text_llm',
    'rewrite_to_destigma',
    
    # Pipeline components
    'create_pipeline',
    'TextPipeline',
    'classify_drug_filter',
    'classify_stigma_filter',
    'analyze_style_filter',
    'rewrite_text_filter',
    
    # Client utilities
    'get_client',
    'LLMClient',
    'OpenAIClient',
    'TogetherClient',
    'ClaudeClient',
    
    # Model utilities
    'get_model_mapping',
    'get_default_model'
]