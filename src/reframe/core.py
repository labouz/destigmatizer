"""Core functionality for the reframe package."""

from typing import Tuple, Dict, Any, Optional, Union
from .clients import get_client
from .classifiers import DrugClassifier, StigmaClassifier
from .analyzers import StyleAnalyzer, EmotionAnalyzer, LLMBasedAnalyzer
from .rewriters import DestigmatizingRewriter


def initialize(api_key: Optional[str] = None, client: Optional[Any] = None, 
              client_type: Optional[str] = None) -> Any:
    """
    Initialize and return a client for the Reframe library.
    
    Args:
        api_key: API key for the language model service
        client: Pre-configured client instance
        client_type: Type of client ("openai", "together", or "claude")
        
    Returns:
        Any: Client instance
        
    Raises:
        ValueError: If neither api_key nor client is provided, or if client_type is unsupported
    """
    if client:
        return client
    elif api_key:
        return get_client(client_type, api_key)
    else:
        raise ValueError("Either api_key or client must be provided")


def classify_if_drug(text: str, client: Any, model: Optional[str] = None,
                    retries: int = 2) -> str:
    """
    Classify if text contains drug-related content.
    
    Args:
        text: Text content to classify
        client: Client instance
        model: Model to use
        retries: Number of retries on failure
        
    Returns:
        str: 'D' for drug-related, 'ND' for non-drug-related, 'skipped' on error
    """
    drug_classifier = DrugClassifier(client)
    return drug_classifier.classify(text, model=model, retries=retries)


def classify_if_stigma(text: str, client: Any, model: Optional[str] = None,
                      retries: int = 2) -> str:
    """
    Classify if text contains stigmatizing language related to drug use.
    
    Args:
        text: Text content to classify
        client: Client instance
        model: Model to use
        retries: Number of retries on failure
        
    Returns:
        str: Classification result with explanation if stigmatizing
    """
    stigma_classifier = StigmaClassifier(client)
    return stigma_classifier.classify(text, model=model, retries=retries)


def analyze_text_llm(text: str, client: Any, model: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze text style and emotion.
    
    Args:
        text: Text to analyze
        client: Client instance
        model: Model to use
        
    Returns:
        dict: Analysis results
    """
    style_analyzer = StyleAnalyzer()
    emotion_analyzer = EmotionAnalyzer(client)
    analyzer = LLMBasedAnalyzer(client, emotion_analyzer, style_analyzer)
    return analyzer.analyze(text, model=model)


def get_emotion(text: str, client: Any, model: Optional[str] = None,
               temperature: float = 0, retries: int = 2) -> str:
    """
    Detect the primary emotion in text.
    
    Args:
        text: Text to analyze
        client: Client instance
        model: Model to use
        temperature: Sampling temperature
        retries: Number of retries on failure
        
    Returns:
        str: Detected emotion
    """
    emotion_analyzer = EmotionAnalyzer(client)
    result = emotion_analyzer.analyze(text, model=model)
    return result.get("primary_emotion", "unknown")


def rewrite_to_destigma(text: str, explanation: str, style_instruct: str, step: int,
                        model: Optional[str] = None, client: Any = None, 
                        retries: int = 2) -> str:
    """
    Rewrite text to remove stigmatizing language.
    
    Args:
        text: Text to rewrite
        explanation: Explanation of stigma from classifier
        style_instruct: Style instructions to maintain
        step: Rewriting step (1 or 2)
        model: Model to use
        client: Client instance
        retries: Number of retries on failure
        
    Returns:
        str: Rewritten text
    """
    rewriter = DestigmatizingRewriter(client)
    return rewriter.rewrite(
        text=text,
        explanation=explanation,
        style_instruct=style_instruct,
        step=step,
        model=model,
        retries=retries
    )
