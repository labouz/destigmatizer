"""
Reframe: A Python package for destigmatizing language related to drug use.

This package provides tools to identify, analyze, and rewrite text
containing stigmatizing language.
"""

# Import from core for backward compatibility
from .core import (
    initialize,
    classify_if_drug,
    classify_if_stigma,
    analyze_text_llm,
    rewrite_to_destigma,
    get_emotion
)

# Import main classes for direct access
from .clients import LLMClient, OpenAIClient, TogetherClient, ClaudeClient, get_client
from .classifiers import BaseClassifier, DrugClassifier, StigmaClassifier
from .analyzers import TextAnalyzer, StyleAnalyzer, EmotionAnalyzer, LLMBasedAnalyzer
from .rewriters import TextRewriter, DestigmatizingRewriter

__all__ = [
    # Core functions (backward compatibility)
    'initialize',
    'classify_if_drug',
    'classify_if_stigma',
    'analyze_text_llm',
    'rewrite_to_destigma',
    'get_emotion',
    
    # Client classes
    'LLMClient',
    'OpenAIClient',
    'TogetherClient',
    'ClaudeClient',
    'get_client',
    
    # Classifier classes
    'BaseClassifier',
    'DrugClassifier',
    'StigmaClassifier',
    
    # Analyzer classes
    'TextAnalyzer',
    'StyleAnalyzer',
    'EmotionAnalyzer',
    'LLMBasedAnalyzer',
    
    # Rewriter classes
    'TextRewriter',
    'DestigmatizingRewriter'
]